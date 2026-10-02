# Evaluating ClearPath-AI

`evaluate_model.py` evaluates saved weights; it never starts training. It creates a new timestamped directory under `evaluation_results/` so previous runs are preserved.

## Run in PyCharm

Open `evaluate_model.py` and select **Run**. Use the same Python interpreter as `main.py` (the current installation has the required libraries). The default run uses the project's `models/yolov8n.pt`, the supplied test split, CPU, 640-pixel inference size, and confidence 0.25. Stop the video demo first for a less noisy timing measurement.

Or, from this project's terminal:

```sh
python3.12 evaluate_model.py
```

Dependencies: ultralytics, torch, numpy, PyYAML, matplotlib. The measured environment is recorded in each run's `metrics.json`; the initial run used Ultralytics 8.3.152.

## What the current checkpoint can measure

The existing application uses the generic COCO YOLOv8n checkpoint and treats any class containing `ambulance`, `police`, `fire`, or `truck` as an emergency. The evaluator mirrors that rule exactly in `app` mode. A labelled image is considered emergency-associated if it contains any annotated ambulance/siren object. This is a binary **image-alert** evaluation, not a bounding-box or ambulance recognition evaluation.

Outputs:

- `REPORT.md`: results, interpretation, limitations, and presentation wording.
- `metrics.json`: accuracy, precision, recall, F1, specificity, false-positive rate, counts, timings, versions, hashes, and dataset audit.
- `per_image_results.csv`: every prediction and its detected/triggering classes.
- `confusion_matrix.png`: actual versus predicted image-level alert counts.
- `test_manifest.json`: evaluated image and annotation paths and SHA-256 hashes (also used for validation runs).

Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2TP/(2TP+FP+FN); accuracy = (TP+TN)/N. Undefined metrics are `null` in JSON and `N/A` in the report, never silently zero.

**The provided dataset has no negative images.** Any nonzero alert count therefore yields 100% measured precision even if the rule is poor. Accuracy equals recall. Neither value establishes reliable discrimination in traffic. Do not present these as validated deployment accuracy, and do not describe this checkpoint as custom trained.

## Evaluate a genuinely trained detector

Provide the weights produced by training on the matching dataset taxonomy:

```sh
python3.12 evaluate_model.py --model runs/detect/train/weights/best.pt --mode detector
```

The script checks that every model class ID/name exactly matches `data.yaml`. It refuses incompatible labels. The current dataset has three distinct IDs: `Ambulance`, `ambulance`, and `siren`; even though the first two names differ only in case, they are separate categories in this dataset. It does not silently merge them. If labels are cleaned, retrain and evaluate with the corresponding dataset.

For compatible weights it runs native Ultralytics validation and saves detection precision, recall, macro F1, mAP@0.50, mAP@0.50:0.95, per-class AP, and native diagnostic plots. Detection accuracy is not used because background true negatives are not naturally defined for bounding-box detection. Native detection PR/F1 uses the library's F1-curve operating point; the separate image-alert table uses fixed confidence 0.25. mAP uses predictions down to confidence 0.001 and IoU thresholds 0.50 through 0.95. NMS IoU is 0.70; this is different from ground-truth matching IoU.

The script copies labels to the output folder and links source images for native validation so generated caches do not modify the supplied annotations. Polygon annotations are evaluated as enclosing bounding boxes in detection validation, not as masks.

## Before citing results on a CV or in a presentation

1. Train an ambulance-specific model and preserve its training configuration and split provenance.
2. Resolve duplicate ambulance IDs if they represent the same category; define whether sirens are a separate detection target.
3. Split by original image/scene/video before augmentation and remove cross-split duplicates. The script audits exact-byte duplicates only; it cannot establish scene independence.
4. Add a separately annotated negative test set containing ordinary cars, trucks, fire hydrants, and other challenging scenes.
5. Choose thresholds on validation data, then freeze them and evaluate the held-out test set once. Use `--split val` while developing.
6. Report test size, class distribution, mAP, per-class precision/recall/F1, threshold, device, and limitations. The saved report supplies wording appropriate to the current baseline.

No additional training, external data downloads, or test-set tuning are performed automatically.

References: [Ultralytics validation](https://docs.ultralytics.com/modes/val/) and [performance metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/).
