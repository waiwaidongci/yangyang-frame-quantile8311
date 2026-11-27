"""Stdlib regression tests for ranges, box-max prediction and locked scheduling.

Run from the repository root:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=backend python3 backend/tests/test_domain.py
"""

import json
import sys

from src.domain.boosting import (QuantileModel, fit_quantile, model_from_dict, model_max,
                                 model_to_dict, predict_shot, shot_features)
from src.domain.project import analyze, demo_rows, normalize_rows
from src.domain.records import parse_measurement
from src.domain.reports import live_schedule
from src.domain.scheduling import (arrange, earliest_start, latest_confirmations, overlaps,
                                   range_within)
from src.domain.settings import parse_settings
from src.domain.trees import Branch, Leaf

PASSED = []


def check(name, condition):
    assert condition, name
    PASSED.append(name)


def base_rows():
    rows = []
    for scene in range(1, 4):
        for number in range(1, 11):
            rows.append({'id': f'H{scene}-{number}', 'scene_id': f'H{scene}', 'split': 'train',
                         'frames': 50 + number, 'occlusion_pct': 20 + number,
                         'overlapping_characters': 1, 'rig_marks': 2,
                         'due_minutes': None, 'actual_minutes': 30.0 + number})
    for number in range(1, 5):
        rows.append({'id': f'V1-{number}', 'scene_id': 'V1', 'split': 'holdout',
                     'frames': 60, 'occlusion_pct': 30, 'overlapping_characters': 0,
                     'rig_marks': 1, 'due_minutes': None, 'actual_minutes': 40.0})
    return rows


def pending_row(**overrides):
    row = {'id': 'P1-1', 'scene_id': 'P1', 'split': 'pending', 'frames': 100,
           'occlusion_pct': '40-60', 'overlapping_characters': 2, 'rig_marks': '2-5',
           'due_minutes': 120, 'actual_minutes': None}
    row.update(overrides)
    return row


def test_demo_regression():
    result = analyze(demo_rows(), {})
    metrics = result['metrics']
    check('演示 P50 Pinball', metrics['P50 留出 Pinball'] == 2.95042)
    check('演示 P90 Pinball', metrics['P90 留出 Pinball'] == 1.996761)
    check('演示 P50 常量', metrics['P50 常量 Pinball'] == 12.839357)
    check('演示 P90 常量', metrics['P90 常量 Pinball'] == 6.453071)
    check('演示覆盖率', (metrics['P50 留出覆盖率'], metrics['P90 留出覆盖率']) == (0.557143, 0.828571))
    check('演示排班计数', (metrics['已分配镜头数'], metrics['待顺延镜头数'], metrics['本班已预留分钟']) == (2, 10, 166))
    queue = next(table for table in result['tables'] if table['key'] == 'shift_queue')
    check('工时单 12 行', len(queue['rows']) == 12)
    first = queue['rows'][0]
    check('RA-P01-01 顺延', first['id'] == 'RA-P01-01' and first['p90_minutes'] == 98.280247
          and first['start_minute'] is None)
    scheduled = {row['id']: row for row in queue['rows'] if row['action'] == '本班可排'}
    check('本班预留 69/97', [row['reserve_minutes'] for row in scheduled.values()] == [69, 97])
    check('精确镜头上下限相等', all(row['occlusion_pct'] == row['occlusion_pct_high']
          and row['rig_marks'] == row['rig_marks_high'] for row in queue['rows']))
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    check('结果含模型快照', set(result['models']) == {'p50', 'p90'})


def test_range_parsing():
    rows = normalize_rows(base_rows() + [pending_row()])
    shot = next(row for row in rows if row['id'] == 'P1-1')
    check('范围解析', (shot['occlusion_pct'], shot['occlusion_pct_high']) == (40.0, 60.0)
          and (shot['rig_marks'], shot['rig_marks_high']) == (2.0, 5.0))
    exact = normalize_rows(base_rows() + [pending_row(id='P1-2', occlusion_pct='43', rig_marks=3)])
    shot = next(row for row in exact if row['id'] == 'P1-2')
    check('精确值折叠为点', (shot['occlusion_pct'], shot['occlusion_pct_high']) == (43.0, 43.0)
          and (shot['rig_marks'], shot['rig_marks_high']) == (3.0, 3.0))
    again = normalize_rows(rows)
    check('归一化幂等', again == rows)
    for bad, message in [
        (pending_row(occlusion_pct='60-40'), '下限超过上限应拒绝'),
        (pending_row(rig_marks='5-2'), '痕迹下限超过上限应拒绝'),
        (pending_row(occlusion_pct='40-160'), '范围越界应拒绝'),
    ]:
        try:
            normalize_rows(base_rows() + [bad])
        except ValueError:
            check(message, True)
        else:
            raise AssertionError(message)
    train_range = base_rows()
    train_range[0]['occlusion_pct'] = '40-60'
    try:
        normalize_rows(train_range)
    except ValueError:
        check('历史镜头拒绝范围', True)
    else:
        raise AssertionError('历史镜头拒绝范围')
    low, high = parse_measurement({'occlusion_pct': '1e-3'}, 'occlusion_pct', 1, 0, 100, allow_range=True)
    check('科学计数法不误判为范围', (low, high) == (0.001, 0.001))


def test_model_max():
    tree = Branch(0, 10.0, Leaf(1.0), Branch(1, 5.0, Leaf(2.0), Leaf(4.0)))
    other = Branch(1, 3.0, Leaf(0.5), Leaf(-1.0))
    model = QuantileModel(constant=2.0, rate=0.5, trees=(tree, other))
    boxes = ((0.0, 20.0), (0.0, 8.0), (1.0, 1.0), (2.0, 2.0))
    grid = [x / 20 for x in range(0, 161)]
    grid_y = [x / 20 for x in range(0, 81)]
    samples = set(grid) | {t + delta for t in (10.0,) for delta in (-0.001, 0.001)}
    samples_y = set(grid_y) | {t + delta for t in (3.0, 5.0) for delta in (-0.001, 0.001)}
    brute = max(model.predict((x, y, 1.0, 2.0)) for x in samples for y in samples_y
                if 0 <= x <= 20 and 0 <= y <= 8)
    check('盒内最大值等于暴力枚举', model_max(model, boxes) == brute)
    point = ((7.0, 7.0), (2.0, 2.0), (1.0, 1.0), (2.0, 2.0))
    check('点盒等于单点预测', model_max(model, point) == model.predict((7.0, 2.0, 1.0, 2.0)))


def trained_models():
    rows = normalize_rows(demo_rows())
    train = [row for row in rows if row['split'] == 'train']
    options = parse_settings({})
    features = [shot_features(row) for row in train]
    actual = [row['actual_minutes'] for row in train]
    return (fit_quantile(features, actual, 50, options),
            fit_quantile(features, actual, 90, options))


def test_ranged_prediction():
    models = trained_models()
    rows = normalize_rows(demo_rows())
    base = next(row for row in rows if row['id'] == 'RA-P01-02')
    ranged = {**base, 'occlusion_pct': max(0.0, base['occlusion_pct'] - 10),
              'occlusion_pct_high': min(100.0, base['occlusion_pct'] + 10),
              'rig_marks': max(0.0, base['rig_marks'] - 1),
              'rig_marks_high': base['rig_marks'] + 2}
    widened = predict_shot(ranged, models)
    exact = predict_shot(base, models)
    check('范围 P90 不低于精确 P90', widened['p90_minutes'] >= exact['p90_minutes'])
    for occ in (ranged['occlusion_pct'], base['occlusion_pct'], ranged['occlusion_pct_high']):
        for rig in (ranged['rig_marks'], ranged['rig_marks_high']):
            point = predict_shot({**ranged, 'occlusion_pct': occ, 'occlusion_pct_high': occ,
                                  'rig_marks': rig, 'rig_marks_high': rig}, models)
            check('盒内任一点不超过范围 P90', point['p90_minutes'] <= widened['p90_minutes'] + 1e-9)
    restored = model_from_dict(json.loads(json.dumps(model_to_dict(models[1]))))
    check('模型快照往返一致', all(restored.predict(shot_features(row)) == models[1].predict(shot_features(row))
          for row in rows[:40]))


def test_earliest_start():
    check('容量不足返回空', earliest_start([], 10, 5) is None)
    check('空工位从零开始', earliest_start([], 10, 20) == 0)
    check('绕过占用时段', earliest_start([(0, 10)], 5, 20) == 10)
    check('恰好嵌入间隙', earliest_start([(0, 10), (20, 30)], 10, 100) == 10)
    check('间隙不足顺延', earliest_start([(0, 10), (20, 30)], 11, 100) == 30)
    check('交接时限截断', earliest_start([(0, 10)], 12, 20) is None)


def test_arrange():
    shots = [
        {'id': 'A', 'due_minutes': 100, 'p90_minutes': 50.0},
        {'id': 'B', 'due_minutes': 200, 'p90_minutes': 60.0},
        {'id': 'C', 'due_minutes': 150, 'p90_minutes': 40.0},
    ]
    placements = arrange(shots, 240, [])
    check('无锁顺序填充', (placements['A']['start_minute'], placements['A']['finish_minute']) == (0, 50)
          and (placements['C']['start_minute'], placements['C']['finish_minute']) == (50, 90)
          and (placements['B']['start_minute'], placements['B']['finish_minute']) == (90, 150))
    locks = [{'shot_id': 'B', 'start_minute': 0, 'finish_minute': 60}]
    placements = arrange(shots, 240, locks)
    check('锁定原样交接', placements['B']['action'] == 'locked'
          and (placements['B']['start_minute'], placements['B']['finish_minute']) == (0, 60))
    check('锁定后赶不上交接则顺延', placements['A']['action'] == 'deferred')
    check('未锁定绕开锁定', (placements['C']['start_minute'], placements['C']['finish_minute']) == (60, 100))
    shots[0]['due_minutes'] = 130
    placements = arrange(shots, 240, locks)
    check('赶上交接的绕锁重排', (placements['A']['start_minute'], placements['A']['finish_minute']) == (60, 110)
          and (placements['C']['start_minute'], placements['C']['finish_minute']) == (110, 150))
    shots[0]['due_minutes'] = 100
    full = arrange(shots, 240, [{'shot_id': 'B', 'start_minute': 0, 'finish_minute': 200}])
    check('工位排满顺延', full['A']['action'] == 'deferred' and full['C']['action'] == 'deferred')
    gap = arrange([{'id': 'D', 'due_minutes': 300, 'p90_minutes': 20.0}], 240,
                  [{'shot_id': 'E', 'start_minute': 0, 'finish_minute': 100}])
    check('锁定后排到其后', (gap['D']['start_minute'], gap['D']['finish_minute']) == (100, 120))


def test_arrange_invariants():
    import random
    rng = random.Random(7)
    shots = [{'id': f'S{index}', 'due_minutes': rng.randint(30, 300),
              'p90_minutes': rng.uniform(5, 80)} for index in range(15)]
    locks = [{'shot_id': 'S0', 'start_minute': 10, 'finish_minute': 60},
             {'shot_id': 'S1', 'start_minute': 100, 'finish_minute': 150}]
    for lock in locks:
        shot = next(row for row in shots if row['id'] == lock['shot_id'])
        shot['due_minutes'] = max(shot['due_minutes'], lock['finish_minute'])
    placements = arrange(shots, 240, locks)
    intervals = sorted((p['start_minute'], p['finish_minute']) for p in placements.values()
                       if p['start_minute'] is not None)
    check('同一工位时段只排一个镜头', all(first[1] <= second[0] for first, second in zip(intervals, intervals[1:])))
    check('排完赶上交接且下班前收工', all(
        placements[shot['id']]['finish_minute'] is None
        or placements[shot['id']]['finish_minute'] <= min(240, shot['due_minutes'])
        for shot in shots))


def test_confirmations():
    records = [
        {'shot_id': 'A', 'version': 2, 'occlusion_low': 45, 'occlusion_high': 55, 'rig_low': 2, 'rig_high': 4},
        {'shot_id': 'A', 'version': 1, 'occlusion_low': 40, 'occlusion_high': 60, 'rig_low': 1, 'rig_high': 5},
        {'shot_id': 'B', 'version': 1, 'occlusion_low': 10, 'occlusion_high': 20, 'rig_low': 0, 'rig_high': 1},
    ]
    latest = latest_confirmations(records)
    check('乱序到达按最高版本生效', latest['A']['version'] == 2 and latest['B']['version'] == 1)
    check('收窄合法', range_within((45, 55), (40, 60)))
    check('扩宽非法', not range_within((35, 55), (40, 60)))
    check('重叠占用', overlaps(5, 15, [{'start_minute': 10, 'finish_minute': 20}]))
    check('相邻不算重叠', not overlaps(20, 30, [{'start_minute': 10, 'finish_minute': 20}]))


def test_live_schedule():
    shots = [
        {'id': 'A', 'scene_id': 'S', 'frames': 10, 'occlusion_pct': 40.0, 'occlusion_pct_high': 60.0,
         'overlapping_characters': 0, 'rig_marks': 2.0, 'rig_marks_high': 5.0, 'due_minutes': 200,
         'p50_minutes': 40.0, 'p90_minutes': 50.0, 'crossing_adjusted': False},
        {'id': 'B', 'scene_id': 'S', 'frames': 10, 'occlusion_pct': 10.0, 'occlusion_pct_high': 10.0,
         'overlapping_characters': 0, 'rig_marks': 1.0, 'rig_marks_high': 1.0, 'due_minutes': 240,
         'p50_minutes': 55.0, 'p90_minutes': 60.0, 'crossing_adjusted': False},
        {'id': 'C', 'scene_id': 'S', 'frames': 10, 'occlusion_pct': 5.0, 'occlusion_pct_high': 5.0,
         'overlapping_characters': 0, 'rig_marks': 1.0, 'rig_marks_high': 1.0, 'due_minutes': 30,
         'p50_minutes': 40.0, 'p90_minutes': 45.0, 'crossing_adjusted': False},
    ]
    locks = [{'shot_id': 'B', 'start_minute': 0, 'finish_minute': 60}]
    table = live_schedule(shots, 240, locks, {'A': 3})
    rows = {row['id']: row for row in table['rows']}
    check('锁定行原样交接', rows['B']['action'] == '已锁定' and rows['B']['start_minute'] == 0
          and rows['B']['finish_minute'] == 60 and rows['B']['reserve_minutes'] == 60)
    check('未锁定重排', rows['A']['action'] == '本班可排'
          and (rows['A']['start_minute'], rows['A']['finish_minute']) == (60, 110))
    check('顺延排最后', table['rows'][-1]['id'] == 'C' and rows['C']['action'] == '顺延复核')
    check('生效版本随排单展示', rows['A']['confirmed_version'] == 3 and rows['B']['confirmed_version'] is None)
    check('时间轴有序', [row['id'] for row in table['rows']] == ['B', 'A', 'C'])


def test_analyze_with_ranges():
    exact = analyze(demo_rows(), {})
    exact_queue = next(table for table in exact['tables'] if table['key'] == 'shift_queue')
    exact_row = next(row for row in exact_queue['rows'] if row['id'] == 'RA-P01-02')
    rows = demo_rows()
    target = next(row for row in rows if row['id'] == 'RA-P01-02')
    target['occlusion_pct'] = f"{max(0, target['occlusion_pct'] - 10)}-{target['occlusion_pct'] + 10}"
    ranged = analyze(rows, {})
    ranged_queue = next(table for table in ranged['tables'] if table['key'] == 'shift_queue')
    ranged_row = next(row for row in ranged_queue['rows'] if row['id'] == 'RA-P01-02')
    check('范围镜头按最高 P90 预留', ranged_row['reserve_minutes'] >= exact_row['reserve_minutes'])
    check('范围上下限进入工时单', ranged_row['occlusion_pct_high'] == target['occlusion_pct'].split('-')[1]
          or float(ranged_row['occlusion_pct_high']) == float(target['occlusion_pct'].split('-')[1]))
    untouched = next(row for row in ranged_queue['rows'] if row['id'] == 'RA-P01-01')
    check('未改镜头结果不变', untouched['p90_minutes'] == 98.280247)


def reschedule(result, pending, confirmations, locks, shift_minutes):
    """Mirror the service pipeline: snapshot models + latest ranges + locks."""
    from src.domain.boosting import model_from_dict
    from src.domain.scheduling import apply_confirmation, latest_confirmations
    models = (model_from_dict(result['models']['p50']), model_from_dict(result['models']['p90']))
    effective = latest_confirmations(confirmations)
    shots = [predict_shot(apply_confirmation(row, effective.get(row['id'])), models) for row in pending]
    versions = {shot_id: entry['version'] for shot_id, entry in effective.items()}
    return live_schedule(shots, shift_minutes, locks, versions)


def test_reschedule_pipeline():
    rows = demo_rows()
    target = next(row for row in rows if row['id'] == 'RA-P01-02')
    exact_occ = target['occlusion_pct']
    target['occlusion_pct'] = f"{exact_occ - 10}-{exact_occ + 10}"
    result = analyze(rows, {})
    pending = [row for row in normalize_rows(rows) if row['split'] == 'pending']
    shift = 240

    initial = reschedule(result, pending, [], [], shift)
    widened = {row['id']: row for row in initial['rows']}
    check('初始按台账范围预留', widened['RA-P01-02']['reserve_minutes'] >= 69)

    v1 = {'shot_id': 'RA-P01-02', 'version': 1, 'occlusion_low': exact_occ - 5,
          'occlusion_high': exact_occ + 5, 'rig_low': 2.0, 'rig_high': 2.0}
    narrowed = reschedule(result, pending, [v1], [], shift)
    after_v1 = {row['id']: row for row in narrowed['rows']}
    check('复核收窄后预留回落', after_v1['RA-P01-02']['reserve_minutes'] <= widened['RA-P01-02']['reserve_minutes'])
    check('生效版本为 1', after_v1['RA-P01-02']['confirmed_version'] == 1)

    v3 = {'shot_id': 'RA-P01-02', 'version': 3, 'occlusion_low': exact_occ - 2,
          'occlusion_high': exact_occ + 2, 'rig_low': 2.0, 'rig_high': 2.0}
    v2 = {'shot_id': 'RA-P01-02', 'version': 2, 'occlusion_low': exact_occ - 4,
          'occlusion_high': exact_occ + 4, 'rig_low': 2.0, 'rig_high': 2.0}
    reordered = reschedule(result, pending, [v1, v3, v2], [], shift)
    latest = {row['id']: row for row in reordered['rows']}
    check('乱序到达按最高版本重排', latest['RA-P01-02']['confirmed_version'] == 3)
    without_v2 = {row['id']: row for row in reschedule(result, pending, [v1, v3], [], shift)['rows']}
    check('旧版本不改状态', latest['RA-P01-02']['reserve_minutes'] == without_v2['RA-P01-02']['reserve_minutes'])

    locks = [{'shot_id': 'RA-P02-03', 'start_minute': 0, 'finish_minute': 90}]
    locked = reschedule(result, pending, [v1, v3, v2], locks, shift)
    final = {row['id']: row for row in locked['rows']}
    check('锁定时段原样交接', final['RA-P02-03']['action'] == '已锁定'
          and (final['RA-P02-03']['start_minute'], final['RA-P02-03']['finish_minute']) == (0, 90))
    intervals = sorted((row['start_minute'], row['finish_minute']) for row in locked['rows']
                       if row['start_minute'] is not None)
    check('重排后同一工位时段只排一个镜头',
          all(first[1] <= second[0] for first, second in zip(intervals, intervals[1:])))
    due = {row['id']: row['due_minutes'] for row in pending}
    check('重排后仍赶上各自交接且下班前收工', all(
        row['finish_minute'] is None or row['finish_minute'] <= min(shift, due[row['id']])
        for row in locked['rows']))


TESTS = [test_demo_regression, test_range_parsing, test_model_max, test_ranged_prediction,
         test_earliest_start, test_arrange, test_arrange_invariants, test_confirmations,
         test_live_schedule, test_analyze_with_ranges, test_reschedule_pipeline]


def main():
    for test in TESTS:
        test()
        print(f'通过 {test.__name__}')
    print(f'{len(PASSED)} 项断言全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
