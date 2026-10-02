"""Dependency-free research protocol checks and image-alert slice summaries."""
import csv
from collections import Counter, defaultdict

SLICE_FIELDS = ('lighting', 'weather', 'source')


def load_metadata(path, records):
    """Require one row per audited image, keyed by canonical split and filename.

    group_id must identify the original scene/video BEFORE augmentation. It is
    user-supplied provenance, not something inferred from similar filenames.
    """
    expected = {(split, r['image'].name) for split, rows in records.items() for r in rows}
    metadata = {}
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'split', 'image', 'group_id'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Metadata requires split,image,group_id columns')
        for row in reader:
            key = (row['split'].strip(), row['image'].strip())
            if key not in expected or key in metadata:
                raise ValueError(f'Unknown or duplicate metadata image: {key}')
            group = (row.get('group_id') or '').strip()
            if not group:
                raise ValueError(f'Missing original scene/video group_id: {key}')
            metadata[key] = dict(group_id=group, **{
                field: (row.get(field) or '').strip() or 'unknown' for field in SLICE_FIELDS})
    missing = expected - metadata.keys()
    if missing:
        raise ValueError(f'Metadata missing {len(missing)} images, e.g. {sorted(missing)[:3]}')
    groups = defaultdict(set)
    for (split, _), row in metadata.items():
        groups[row['group_id']].add(split)
    overlaps = {group: sorted(splits) for group, splits in sorted(groups.items()) if len(splits) > 1}
    return metadata, overlaps


def protocol_issues(records, split, targets, exact_overlaps, metadata, group_overlaps):
    selected = records[split]
    positives = sum(bool(set(r['classes']) & targets) for r in selected)
    issues = []
    if positives == 0:
        issues.append('Selected split has no target-positive images; recall cannot be estimated.')
    if positives == len(selected):
        issues.append('Selected split has no target-negative images; false-positive rate cannot be estimated.')
    if exact_overlaps:
        issues.append('Exact image duplicates cross splits.')
    if metadata is None:
        issues.append('Original scene/video metadata is missing; group independence is unverified.')
    if group_overlaps:
        issues.append('Original scene/video groups cross splits.')
    return issues


def summarize_rows(rows, metric_fn):
    counts = Counter(row['outcome'] for row in rows)
    return dict(images=len(rows), positive_images=counts['TP'] + counts['FN'],
                negative_images=counts['FP'] + counts['TN'],
                metrics=metric_fn(*(counts[key] for key in ('TP', 'FP', 'FN', 'TN'))))


def slice_metrics(rows, metric_fn):
    """Descriptive image-alert metrics, not object AP or causal shift estimates."""
    result = {}
    for field in SLICE_FIELDS:
        buckets = defaultdict(list)
        for row in rows:
            buckets[row.get(field, 'unknown')].append(row)
        result[field] = {value: summarize_rows(bucket, metric_fn)
                         for value, bucket in sorted(buckets.items())}
    return result


def presentation_wording(split, count, recall, mode, targets):
    measured = 'undefined (no target-positive images)' if recall is None else f'{100 * recall:.2f}%'
    return (f'Evaluated a {mode}-mode image-alert rule on {count} annotated {split} images '
            f'for dataset target IDs {sorted(targets)}; image-level recall was {measured}. '
            'Interpret with the negative-image counts and split-provenance audit. '
            'This does not establish active-emergency recognition, deployment safety, '
            'or reduced response time. Report object-detection mAP separately when available.')
