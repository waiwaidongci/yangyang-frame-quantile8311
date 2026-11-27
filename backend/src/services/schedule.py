from fastapi import HTTPException
from sqlalchemy import select

from src.constants.workflow import RunStatus
from src.domain.boosting import model_from_dict, predict_shot
from src.domain.records import normalize_rows
from src.domain.reports import live_schedule
from src.domain.scheduling import apply_confirmation, latest_confirmations, overlaps, range_within
from src.models.confirmation import Confirmation
from src.models.run import Run
from src.models.slot_lock import SlotLock
from src.services.audit import record
from src.services.catalog import get_dataset
from src.services.runs import get_run
from src.utils.time import utc_stamp


def pending_index(dataset):
    return {row['id']: row for row in normalize_rows(dataset.rows) if row['split'] == 'pending'}


def run_for_update(session, run_id):
    # Serializes concurrent confirmations and locks on the same run.
    run = session.scalar(select(Run).where(Run.id == run_id).with_for_update())
    if run is None:
        raise LookupError('找不到运行')
    if run.status != RunStatus.COMPLETED:
        raise HTTPException(409, '计算成功完成后才能登记复核或锁定')
    return run


def confirmation_info(entry):
    return {'shot_id': entry.shot_id, 'version': entry.version,
            'occlusion_low': entry.occlusion_low, 'occlusion_high': entry.occlusion_high,
            'rig_low': entry.rig_low, 'rig_high': entry.rig_high, 'created_at': entry.created_at}


def lock_info(entry):
    return {'shot_id': entry.shot_id, 'start_minute': entry.start_minute,
            'finish_minute': entry.finish_minute, 'created_at': entry.created_at}


def run_confirmations(session, run_id):
    return session.scalars(select(Confirmation).where(Confirmation.run_id == run_id)
                           .order_by(Confirmation.shot_id, Confirmation.version)).all()


def run_locks(session, run_id):
    return session.scalars(select(SlotLock).where(SlotLock.run_id == run_id)
                           .order_by(SlotLock.start_minute, SlotLock.shot_id)).all()


def current_schedule(session, run):
    if run.status != RunStatus.COMPLETED:
        raise HTTPException(409, '计算完成后可查看重排工时单')
    snapshot = (run.result or {}).get('models')
    if snapshot is None:
        raise HTTPException(409, '该运行没有模型快照，请重新提交运行')
    dataset = get_dataset(session, run.dataset_id)
    effective = latest_confirmations([confirmation_info(entry) for entry in run_confirmations(session, run.id)])
    models = (model_from_dict(snapshot['p50']), model_from_dict(snapshot['p90']))
    shots = [predict_shot(apply_confirmation(row, effective.get(shot_id)), models)
             for shot_id, row in pending_index(dataset).items()]
    locks = [lock_info(entry) for entry in run_locks(session, run.id)]
    versions = {shot_id: entry['version'] for shot_id, entry in effective.items()}
    return live_schedule(shots, run.settings['shift_minutes'], locks, versions)


def confirm(session, run_id, payload, actor):
    run = run_for_update(session, run_id)
    dataset = get_dataset(session, run.dataset_id)
    shot = pending_index(dataset).get(payload.shot_id)
    if shot is None:
        raise LookupError('找不到该运行的待排镜头')
    if payload.occlusion_low > payload.occlusion_high or payload.rig_low > payload.rig_high:
        raise ValueError('复核范围的下限不能超过上限。')
    incoming = ((payload.occlusion_low, payload.occlusion_high), (payload.rig_low, payload.rig_high))
    ledger = ((shot['occlusion_pct'], shot['occlusion_pct_high']), (shot['rig_marks'], shot['rig_marks_high']))
    if not all(range_within(new, old) for new, old in zip(incoming, ledger)):
        raise ValueError('复核范围须落在台账记录的范围内。')
    records = [entry for entry in run_confirmations(session, run_id) if entry.shot_id == payload.shot_id]
    current = max(records, key=lambda entry: entry.version, default=None)
    effective = current is None or payload.version > current.version
    if effective and current is not None:
        standing = ((current.occlusion_low, current.occlusion_high), (current.rig_low, current.rig_high))
        if not all(range_within(new, old) for new, old in zip(incoming, standing)):
            raise ValueError('复核只会收窄范围，新范围须落在当前生效范围内。')
    entry = Confirmation(run_id=run_id, shot_id=payload.shot_id, version=payload.version,
                         occlusion_low=payload.occlusion_low, occlusion_high=payload.occlusion_high,
                         rig_low=payload.rig_low, rig_high=payload.rig_high,
                         created_by=actor.id, created_at=utc_stamp())
    session.add(entry)
    record(session, 'schedule.confirmed', run_id, actor.id)
    session.commit()
    return {'confirmation': confirmation_info(entry), 'effective': effective,
            'schedule': current_schedule(session, run)}


def lock(session, run_id, payload, actor):
    run = run_for_update(session, run_id)
    dataset = get_dataset(session, run.dataset_id)
    shot = pending_index(dataset).get(payload.shot_id)
    if shot is None:
        raise LookupError('找不到该运行的待排镜头')
    start, finish = payload.start_minute, payload.finish_minute
    if not start < finish:
        raise ValueError('锁定时段须满足开始分钟早于完成分钟。')
    if finish > run.settings['shift_minutes']:
        raise ValueError('锁定时段须在下班前收工。')
    if finish > shot['due_minutes']:
        raise ValueError('锁定时段须赶上该镜头的审片交接。')
    existing = run_locks(session, run_id)
    if any(entry.shot_id == payload.shot_id for entry in existing):
        raise HTTPException(409, '该镜头已锁定，锁定时段原样交接。')
    if overlaps(start, finish, [lock_info(entry) for entry in existing]):
        raise HTTPException(409, '同一工位时段只能排一个镜头。')
    entry = SlotLock(run_id=run_id, shot_id=payload.shot_id, start_minute=start,
                     finish_minute=finish, created_by=actor.id, created_at=utc_stamp())
    session.add(entry)
    record(session, 'schedule.locked', run_id, actor.id)
    session.commit()
    return {'lock': lock_info(entry), 'schedule': current_schedule(session, run)}


def get_schedule(session, run_id, actor):
    run, _ = get_run(session, run_id, actor)
    return current_schedule(session, run)


def list_confirmations(session, run_id, actor):
    get_run(session, run_id, actor)
    return [confirmation_info(entry) for entry in run_confirmations(session, run_id)]


def list_locks(session, run_id, actor):
    get_run(session, run_id, actor)
    return [lock_info(entry) for entry in run_locks(session, run_id)]
