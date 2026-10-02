"""Reproducible ClearPath-AI evaluation. Run with --help; never trains a model."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from evaluation_protocol import load_metadata, protocol_issues, slice_metrics, presentation_wording

KEYWORDS = ('ambulance', 'police', 'fire', 'truck')  # controllers/detection.py
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def ratio(a, b):
    return a / b if b else None


def binary_metrics(tp, fp, fn, tn):
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                accuracy=ratio(tp + tn, tp + fp + fn + tn),
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                f1=ratio(2 * tp, 2 * tp + fp + fn),
                specificity=ratio(tn, tn + fp), false_positive_rate=ratio(fp, fp + tn))


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit_dataset(root, names):
    """Fail on missing/malformed labels; audit exact file duplicates across splits."""
    summary, records, hashes = {}, {}, {}
    for split, folder in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:
        images = sorted(p for p in (root / folder / 'images').iterdir()
                        if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f'No images in {folder}')
        if len({p.stem for p in images}) != len(images):
            raise ValueError(f'Ambiguous annotation filenames in {folder}: repeated image stems')
        counts, rows, formats = Counter(), [], Counter()
        for image in images:
            label = root / folder / 'labels' / (image.stem + '.txt')
            if not label.is_file():
                raise ValueError(f'Missing annotation: {label}; not assumed negative')
            boxes = []
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                fields = line.split()
                values = list(map(float, fields))
                cls, coordinates = values[0], values[1:]
                if not cls.is_integer() or int(cls) not in names or not all(0 <= v <= 1 for v in coordinates):
                    raise ValueError(f'Invalid annotation in {label}: {line}')
                if len(coordinates) == 4:
                    if coordinates[2] <= 0 or coordinates[3] <= 0:
                        raise ValueError(f'Degenerate box in {label}')
                    formats['boxes'] += 1
                elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
                    if max(coordinates[::2]) <= min(coordinates[::2]) or max(coordinates[1::2]) <= min(coordinates[1::2]):
                        raise ValueError(f'Degenerate polygon in {label}')
                    formats['polygons'] += 1
                else:
                    raise ValueError(f'Expected YOLO box or polygon: {label}')
                boxes.append(int(cls)); counts[int(cls)] += 1
            digest = sha256(image)
            hashes.setdefault(digest, []).append(f'{split}/{image.name}')
            rows.append(dict(image=image, label=label, classes=boxes, image_sha256=digest,
                             label_sha256=sha256(label)))
        records[split] = rows
        summary[split] = dict(images=len(rows), annotated_objects=sum(counts.values()), annotation_formats=dict(formats),
                              positive_images=sum(bool(r['classes']) for r in rows),
                              negative_images=sum(not r['classes'] for r in rows),
                              objects_by_class={f'{k}:{names[k]}': v for k,v in sorted(counts.items())})
    overlaps = [v for v in hashes.values() if len({x.split('/')[0] for x in v}) > 1]
    return summary, records, overlaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument('--model', type=Path, default=base/'models/yolov8n.pt')
    parser.add_argument('--data', type=Path, default=base/'emergency-dataset/data.yaml')
    parser.add_argument('--split', choices=['test', 'val'], default='test')
    parser.add_argument('--output', type=Path, default=base/'evaluation_results')
    parser.add_argument('--conf', type=float, default=0.25, help='Fixed image-alert confidence threshold; do not tune on test')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--metadata', type=Path, help='CSV with split,image,group_id and optional lighting,weather,source')
    parser.add_argument('--target-ids', type=int, nargs='+', help='Dataset class IDs defining an image alert; default: all IDs')
    parser.add_argument('--strict-protocol', action='store_true', help='Require positive/negative controls, provenance, and no cross-split overlap')
    parser.add_argument('--audit-only', action='store_true', help='Audit data and protocol without loading weights or running inference')
    parser.add_argument('--mode', choices=['auto', 'app', 'detector'], default='auto',
                        help='app mirrors existing keyword rule; detector requires exact label mapping')
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error('--conf must be between 0 and 1')
    if args.imgsz <= 0:
        parser.error('--imgsz must be positive')
    import yaml
    cfg = yaml.safe_load(args.data.read_text())
    names = cfg['names']
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k):v for k,v in names.items()}
    root = args.data.resolve().parent
    summary, records, overlaps = audit_dataset(root, names)
    targets = set(names if args.target_ids is None else args.target_ids)
    if not targets or not targets.issubset(names):
        parser.error('--target-ids must be nonempty and present in data.yaml names')
    metadata, group_overlaps = load_metadata(args.metadata, records) if args.metadata else (None, {})
    issues = protocol_issues(records, args.split, targets, overlaps, metadata, group_overlaps)
    protocol = dict(target_ids=sorted(targets), issues=issues,
                    checks_passed=not issues, cross_split_groups=group_overlaps,
                    metadata_sha256=sha256(args.metadata) if args.metadata else None,
                    note='Checks depend on supplied provenance; passing is not proof of scene independence or deployment readiness.')
    if args.audit_only:
        print(json.dumps(dict(dataset_summary=summary, cross_split_exact_duplicates=overlaps,
                              protocol=protocol), indent=2))
    if args.strict_protocol and issues:
        parser.error('Protocol checks failed: ' + ' '.join(issues))
    if args.audit_only:
        return
    if not args.model.is_file():
        raise FileNotFoundError(args.model)  # no implicit download
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    model = YOLO(str(args.model.resolve()))
    compatible = dict(model.names) == names
    if args.mode == 'detector' and not compatible:
        raise ValueError(f'Model labels do not match dataset IDs. Model: {model.names}; dataset: {names}')
    mode = ('detector' if compatible else 'app') if args.mode == 'auto' else args.mode
    out = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+args.split)
    out.mkdir(parents=True, exist_ok=False)
    selected = records[args.split]
    print(f'Evaluating {len(selected)} {args.split} images in {mode} mode. Output: {out}', flush=True)
    manifest = [{**r, 'image': str(r['image'].relative_to(root)), 'label': str(r['label'].relative_to(root)),
                 'metadata': metadata[(args.split, r['image'].name)] if metadata else None} for r in selected]
    (out/'test_manifest.json').write_text(json.dumps(manifest, indent=2))
    warnings = ['Dataset annotations include boxes/polygons; native detection validation evaluates enclosing boxes for polygon labels, not segmentation masks.']
    if not compatible:
        warnings.append('COCO and dataset class IDs have different meanings. Ambulance detection mAP and per-class detection scores are unavailable for this checkpoint. Native model.val was intentionally not run.')
    if all(set(r['classes']) & targets for r in selected):
        warnings.append('No negative images: image-level accuracy equals recall, precision is trivial when alerts occur, and specificity/false-positive rate cannot be estimated. These are NOT deployment accuracy claims.')
    if overlaps:
        warnings.append(f'{len(overlaps)} exact-image duplicate groups span dataset splits; the supplied test set is not fully independent. Re-split by original scene before evaluating a trained model.')
    warnings.extend(issues)
    warnings.append(f'Image-level ground truth is positive for dataset target IDs {sorted(targets)}. Other annotated classes are target-negative. This is a visual-object proxy, not proof of an active emergency or siren audio.')
    warnings.append('Condition slices are descriptive image-alert metrics, not per-condition detection mAP. Correlated frames and small slices limit inference; no confidence intervals are claimed.')
    warnings.append('Exact-byte duplicate checks do not detect resized/augmented copies or shared scenes. No external generalization or confidence-interval claim is made.')
    model.predict(str(selected[0]['image']), imgsz=args.imgsz, conf=args.conf,
                  iou=0.7, device=args.device, verbose=False, save=False)  # excluded warm-up
    rows, latencies = [], []
    tp = fp = fn = tn = 0
    start = time.perf_counter()
    for i, record in enumerate(selected, 1):
        tick = time.perf_counter()
        result = model.predict(str(record['image']), imgsz=args.imgsz, conf=args.conf,
                               iou=0.7, device=args.device, verbose=False, save=False)[0]
        elapsed = (time.perf_counter()-tick)*1000
        latencies.append(elapsed)
        predicted_ids = [int(c) for c in result.boxes.cls.cpu().tolist()]
        classes = [result.names[c] for c in predicted_ids]
        triggers = ([c for c in classes if any(k in c.lower() for k in KEYWORDS)] if mode == 'app'
                    else [result.names[c] for c in predicted_ids if c in targets])
        predicted, actual = bool(triggers), bool(set(record['classes']) & targets)
        outcome = 'TP' if actual and predicted else 'FN' if actual else 'FP' if predicted else 'TN'
        tp += outcome == 'TP'; fp += outcome == 'FP'; fn += outcome == 'FN'; tn += outcome == 'TN'
        rows.append(dict(image=record['image'].name, actual_positive=actual, predicted_positive=predicted,
                         outcome=outcome, detected_classes=json.dumps(classes), triggers=json.dumps(triggers),
                         latency_ms=round(elapsed,3)))
        rows[-1].update(metadata[(args.split, record['image'].name)] if metadata else
                        dict(group_id='unknown', lighting='unknown', weather='unknown', source='unknown'))
        if i % 20 == 0 or i == len(selected):
            print(f'{i}/{len(selected)} images evaluated', flush=True)
    duration = time.perf_counter()-start
    metrics = binary_metrics(tp,fp,fn,tn)
    detection = None
    if compatible and mode == 'detector':
        # Copy annotations into the output area so Ultralytics caches cannot alter source data.
        eval_root = out/'dataset'
        for split, source_rows in records.items():
            folder = {'val':'valid'}.get(split,split)
            for kind in ['images','labels']:
                (eval_root/folder/kind).mkdir(parents=True)
            for r in source_rows:
                (eval_root/folder/'images'/r['image'].name).symlink_to(r['image'].resolve())
                shutil.copy2(r['label'], eval_root/folder/'labels'/r['label'].name)
        resolved = dict(path=str(eval_root), train='train/images', val='valid/images', test='test/images', names=names)
        resolved_path = out/'resolved_data.yaml'
        resolved_path.write_text(yaml.safe_dump(resolved))
        val = model.val(data=str(resolved_path), split=args.split, imgsz=args.imgsz,
                        device=args.device, batch=1, workers=0, conf=0.001, iou=0.7,
                        plots=True, project=str(out), name='detection', exist_ok=True)
        detection = dict(precision=float(val.box.mp), recall=float(val.box.mr),
                         macro_f1=float(np.mean(val.box.f1)), map50=float(val.box.map50),
                         map50_95=float(val.box.map),
                         note='Native Ultralytics PR/F1 operating point uses its F1 curve; differs from the fixed image-alert threshold. mAP integrates confidence rankings.',
                         per_class=[dict(class_id=int(c),name=names[int(c)],precision=float(val.box.p[j]),
                                         recall=float(val.box.r[j]),f1=float(val.box.f1[j]),
                                         ap50=float(val.box.ap50[j]),ap50_95=float(val.box.ap[j]))
                                    for j,c in enumerate(val.box.ap_class_index)])
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), mode=mode,
                  model=str(args.model.resolve()), model_sha256=sha256(args.model), model_classes=model.names,
                  dataset_classes=names, class_mapping_compatible=compatible, split=args.split,
                  conf=args.conf, nms_iou=0.7, imgsz=args.imgsz, device=args.device,
                  versions=dict(python=sys.version, ultralytics=ultralytics.__version__, torch=torch.__version__, platform=platform.platform()),
                  dataset_summary=summary, cross_split_exact_duplicates=overlaps,
                  protocol=protocol, image_alert_slices=slice_metrics(rows, binary_metrics),
                  image_level_metrics=metrics, detection_metrics=detection,
                  timing=dict(mean_ms=float(np.mean(latencies)), median_ms=float(np.median(latencies)),
                              p95_ms=float(np.percentile(latencies,95)), images_per_second=len(selected)/duration,
                              note='Sequential batch=1 wall time, image decoding/preprocessing/inference/postprocessing included; warm-up excluded. Not video-stream FPS.'),
                  limitations=warnings)
    (out/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    with (out/'per_image_results.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'failure_cases.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        w.writerows(row for row in rows if row['outcome'] in ('FP', 'FN'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4,4.8))
    matrix=np.array([[tn,fp],[fn,tp]])
    ax.imshow(matrix,cmap='Blues')
    for (y,x),value in np.ndenumerate(matrix):
        ax.text(x,y,str(value),ha='center',va='center',fontsize=22,color='white' if value>matrix.max()/2 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['No alert','Alert'],yticklabels=['Negative','Positive'],
           xlabel='Predicted alert',ylabel='Annotated image',title=f'ClearPath-AI: image-level {args.split} confusion matrix')
    fig.text(.5,.02,'Emergency-associated images; not bounding-box detection accuracy',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'confusion_matrix.png',dpi=180);plt.close(fig)
    def pct(v): return 'N/A' if v is None else f'{100*v:.2f}%'
    lines=['# ClearPath-AI — measured evaluation results','',f'Evaluated {len(selected)} supplied {args.split} images. Mode: **{mode}**.',
           '', '**Scope:** image-level emergency-alert proxy. These results do not establish ambulance detection accuracy.',
           '', '| Metric | Result |','|---|---|']
    lines += [f'| {k.replace("_"," ").title()} | {pct(metrics[k])} |' for k in ['accuracy','precision','recall','f1','specificity','false_positive_rate']]
    lines += ['',f'TP={tp}; FP={fp}; FN={fn}; TN={tn}. Confidence={args.conf}; image size={args.imgsz}; device={args.device}.',
              '',f'Mean image processing time: {np.mean(latencies):.1f} ms; throughput: {len(selected)/duration:.2f} images/s (warm-up excluded).',
              '', '## Object detection metrics']
    if detection:
        lines += ['',f'mAP@0.50: {pct(detection["map50"])}; mAP@0.50:0.95: {pct(detection["map50_95"])}.',
                  'Per-class precision, recall, F1 and AP are in metrics.json; detection/ contains native plots.']
    else:
        lines += ['', '**Not available:** this model has no compatible ambulance/siren class mapping. Reporting native class-ID scores would compare different categories.']
    lines += ['', '## Interpretation and limitations',''] + ['- '+w for w in warnings]
    lines += ['', '## Condition slices (image alerts only)', '',
              '| Field | Value | Images | Positives | Negatives | Recall | False-positive rate |',
              '|---|---|---:|---:|---:|---:|---:|']
    for field, buckets in result['image_alert_slices'].items():
        for value, bucket in buckets.items():
            value = value.replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')
            lines.append(f'| {field} | {value} | {bucket["images"]} | {bucket["positive_images"]} | {bucket["negative_images"]} | {pct(bucket["metrics"]["recall"])} | {pct(bucket["metrics"]["false_positive_rate"])} |')
    lines += ['', '## Presentation wording','',
              presentation_wording(args.split, len(selected), metrics['recall'], mode, targets),
              '', '## Reproducibility','',f'- Model SHA-256: `{result["model_sha256"]}`',
              '- metrics.json records settings, environment, dataset audit and exact duplicates.',
              '- test_manifest.json records evaluated paths and image/annotation hashes.',
              '- per_image_results.csv records every decision, label and processing time.',
              '- No training or test-threshold optimization was performed by this script.',
              '', 'Metric references: https://docs.ultralytics.com/modes/val/ and https://docs.ultralytics.com/guides/yolo-performance-metrics/']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(metrics,indent=2));print(f'Report: {out / "REPORT.md"}',flush=True)


if __name__ == '__main__':
    main()
