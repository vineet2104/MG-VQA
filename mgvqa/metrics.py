"""Answer-only accuracy shared by worker and merged summaries (no runtime imports)."""
from collections import Counter


def is_answer_correct(record):
    if record.get('status') != 'answered':
        return False
    if record.get('question_type') == 'relative_size':
        return bool(record.get('letter_correct', record.get('is_correct')))
    return bool(record.get('is_correct'))


def answer_metrics(records, scale=1.0, digits=3):
    """Exclude every non-answer from accuracy; retain counts/reasons separately.

    ``total`` is inventory only, not an accuracy denominator. Legacy workers
    use percentage units, whereas merged summaries use fractions.
    """
    per_type = {}
    statuses = Counter()
    failures = Counter()
    for record in records:
        status = record.get('status', 'unknown')
        statuses[status] += 1
        stats = per_type.setdefault(record.get('question_type', 'unknown'), {
            'total': 0, 'answered': 0, 'correct': 0, 'non_answer': 0,
            'status_counts': {},
        })
        stats['total'] += 1
        stats['status_counts'][status] = stats['status_counts'].get(status, 0) + 1
        if status == 'answered':
            stats['answered'] += 1
            stats['correct'] += int(is_answer_correct(record))
        else:
            stats['non_answer'] += 1
            if status not in {'skipped', 'snapshot_saved'}:
                failures[record.get('fail_reason') or record.get('error') or status] += 1
    for stats in per_type.values():
        stats['accuracy'] = (round(scale * stats['correct'] / stats['answered'], digits)
                             if stats['answered'] else 0.0)
        stats['accuracy_answered'] = stats['accuracy']
    return {'per_type': per_type, 'status_counts': dict(statuses),
            'non_answer_failures_by_reason': dict(failures)}
