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

## Research protocol and condition analysis

The evaluator now supports explicit alert targets, original-scene provenance checks,
condition summaries, and a CSV of false-positive/false-negative images. These are
evaluation capabilities, not new measured results or improved model weights.
The published 121-image baseline is unchanged.

Create a metadata CSV with **one row for every image in train, val, and test**:

```csv
split,image,group_id,lighting,weather,source
train,example_train.jpg,original_video_01,day,clear,camera_a
val,example_val.jpg,original_video_02,night,rain,camera_b
test,example_test.jpg,original_video_03,day,fog,camera_c
```

These filenames are examples only. Use `val` in the CSV even though its directory
is `valid`. A group identifies an original scene/video **before augmentation**;
all its frames and derived images must have the same group ID. Do not generate
unique groups per frame to make checks pass. Assign conditions by inspecting
source data, not by guessing from filenames. Unspecified conditions are `unknown`.
Group IDs must be globally unique across independent sources.

Audit without model weights (the current train/valid/test directory layout remains required):

```sh
python evaluate_model.py --data emergency-dataset/data.yaml \
  --metadata scene_metadata.csv --target-ids 0 1 --audit-only --strict-protocol
```

For the supplied taxonomy, IDs 0 and 1 are the two ambulance categories; confirm
the IDs against your own `data.yaml`. `--target-ids` affects image-level ground
truth and, in detector mode, alert predictions. It does **not** merge categories
or restrict native object-detection mAP, which still covers the full taxonomy.
In app mode the original keyword prediction rule stays unchanged for a faithful
baseline comparison. Omitting `--target-ids` preserves the old all-object proxy.
An empty, reviewed annotation file is a negative image; a missing label is an error.
Siren-only images are ambulance-negative when siren is excluded, not necessarily
ordinary-traffic negatives. Include genuinely diverse hard negatives as well.

Develop on validation data, then freeze the operating threshold and configuration:

```sh
python evaluate_model.py --model runs/detect/train/weights/best.pt \
  --mode detector --split val --target-ids 0 1 --metadata scene_metadata.csv \
  --strict-protocol --conf 0.25
```

After selecting settings on validation only, run the same frozen configuration
with `--split test`. The flag is a check, not an enforcement of your experimental
history: it cannot detect previous test-set tuning or fabricated provenance.
It rejects missing provenance, any cross-split scene or exact-byte overlap, and
an evaluated split lacking either target-positive or target-negative images.
Without strict mode, these issues remain visible warnings for legacy baseline runs.

New outputs within each run:

- `metrics.json`: target IDs, provenance hash, cross-split groups, protocol issues,
  and lighting/weather/source image-alert metrics with positive/negative counts.
- `REPORT.md`: condition tables and wording using the actual evaluation split.
- `failure_cases.csv`: FP/FN rows for manual error analysis; header only if none.
- `test_manifest.json`: dataset-relative image/label paths and selected metadata.

Condition results are descriptive and cannot isolate causal effects of weather or
lighting. They are not per-condition detection mAP. Missing denominators remain
null/N/A. Correlated frames and small slices do not support independent-image
confidence claims. Passing checks does not prove scene independence, robustness,
emergency status, or operational safety.

### A useful experiment sequence

1. Review label taxonomy and source provenance; remove cross-split source overlap
   before retraining. Keep an untouched independent test set with hard negatives.
2. Compare the existing COCO keyword baseline and a custom detector on identical
   held-out images, target semantics, and recorded inference settings.
3. Preserve seeds, training config/logs, checkpoint hashes and validation-selected
   thresholds. Compare several training seeds if resources permit.
4. Report detection mAP alongside fixed-threshold image recall, false-positive
   rate, condition support, latency, and a manually reviewed error taxonomy.
5. Write a short technical report describing hypotheses, comparisons, failures,
   and limitations. Claim measured findings only after running the experiments.

Run regression checks with `python -m unittest discover -s tests -v`. The pipeline
test uses synthetic predictions: passing tests establishes evaluator behavior,
not detector quality. Real checkpoint/dataset inference must be validated separately.

References: [Ultralytics validation](https://docs.ultralytics.com/modes/val/) and [performance metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/).
