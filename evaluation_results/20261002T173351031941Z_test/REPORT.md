# ClearPath-AI — measured evaluation results

Evaluated 121 supplied test images. Mode: **app**.

**Scope:** image-level emergency-alert proxy. These results do not establish ambulance detection accuracy.

| Metric | Result |
|---|---|
| Accuracy | 60.33% |
| Precision | 100.00% |
| Recall | 60.33% |
| F1 | 75.26% |
| Specificity | N/A |
| False Positive Rate | N/A |

TP=73; FP=0; FN=48; TN=0. Confidence=0.25; image size=640; device=cpu.

Mean image processing time: 48.6 ms; throughput: 20.55 images/s (warm-up excluded).

## Object detection metrics

**Not available:** this model has no compatible ambulance/siren class mapping. Reporting native class-ID scores would compare different categories.

## Interpretation and limitations

- Dataset annotations include boxes/polygons; native detection validation evaluates enclosing boxes for polygon labels, not segmentation masks.
- COCO and dataset class IDs have different meanings. Ambulance detection mAP and per-class detection scores are unavailable for this checkpoint. Native model.val was intentionally not run.
- No negative images: image-level accuracy equals recall, precision is trivial when alerts occur, and specificity/false-positive rate cannot be estimated. These are NOT deployment accuracy claims.
- Image-level ground truth is positive if ANY annotated dataset object is present, including siren. This is an emergency-associated-image proxy, not proof of an active emergency or siren audio.
- Exact-byte duplicate checks do not detect resized/augmented copies or shared scenes. No external generalization or confidence-interval claim is made.

## Presentation wording

“Evaluated a pretrained YOLOv8n-based emergency-alert prototype on 121 annotated test images; measured image-level recall of 60.33%. The current evaluation lacks negative controls and a trained ambulance-specific detector, so it does not establish real-world ambulance detection accuracy.”

## Reproducibility

- Model SHA-256: `31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95`
- metrics.json records settings, environment, dataset audit and exact duplicates.
- test_manifest.json records evaluated paths and image/annotation hashes.
- per_image_results.csv records every decision, label and processing time.
- No training or test-threshold optimization was performed by this script.

Metric references: https://docs.ultralytics.com/modes/val/ and https://docs.ultralytics.com/guides/yolo-performance-metrics/
