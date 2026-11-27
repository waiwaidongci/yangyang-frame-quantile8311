from fastapi import APIRouter

from src.controllers.dependencies import Actor, Admin, Db, Writer
from src.controllers.report import report_csv, report_detail
from src.schemas.reviews import ReviewCreate, ReviewInfo
from src.schemas.runs import RunCreate, RunInfo
from src.schemas.schedule import ConfirmationCreate, LockCreate
from src.services import reviews, runs, schedule

router = APIRouter(prefix='/runs')


@router.get('', response_model=list[RunInfo])
def list_all(session: Db, actor: Actor):
    return runs.list_runs(session, actor)


@router.post('', response_model=RunInfo, status_code=202)
def create(payload: RunCreate, session: Db, actor: Writer):
    return runs.create_run(session, payload, actor)


@router.get('/{run_id}')
def detail(run_id: int, session: Db, actor: Actor):
    return report_detail(session, run_id, actor)


@router.post('/{run_id}/review', response_model=ReviewInfo, status_code=201)
def review(run_id: int, payload: ReviewCreate, session: Db, actor: Admin):
    return reviews.submit(session, run_id, payload, actor)


@router.get('/{run_id}/schedule')
def live(run_id: int, session: Db, actor: Actor):
    return schedule.get_schedule(session, run_id, actor)


@router.post('/{run_id}/confirmations', status_code=201)
def confirm(run_id: int, payload: ConfirmationCreate, session: Db, actor: Writer):
    return schedule.confirm(session, run_id, payload, actor)


@router.get('/{run_id}/confirmations')
def confirmations(run_id: int, session: Db, actor: Actor):
    return schedule.list_confirmations(session, run_id, actor)


@router.post('/{run_id}/locks', status_code=201)
def lock(run_id: int, payload: LockCreate, session: Db, actor: Writer):
    return schedule.lock(session, run_id, payload, actor)


@router.get('/{run_id}/locks')
def locks(run_id: int, session: Db, actor: Actor):
    return schedule.list_locks(session, run_id, actor)


@router.get('/{run_id}/export/{key}')
def export(run_id: int, key: str, session: Db, actor: Actor):
    return report_csv(session, run_id, key, actor)
