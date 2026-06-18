# Unusual Activity Recognition: Approach and LOSO Findings

## Approach

The model is trained only on the supplied labeled 2D pose files in `keypointlabel/keypoints_with_labels_*.csv`. Label discrepancies were canonicalized (`Biting` to `Biting nails`, and `Throwing` to `Throwing things`), while unlabeled intervals were excluded from training and evaluation.

Each frame is translated to the hip center and divided by the larger of torso length and shoulder width. This removes most subject size, camera scale, and absolute-position differences. Continuous same-label intervals are divided into 60-frame (2-second) windows with a 30-frame stride. Features summarize normalized joint positions, upper-body joint distances, velocity, and acceleration.

Window features are then z-score normalized separately for each subject. This uses only the mean and standard deviation of that subject's unlabeled pose features, never their activity labels. It substantially reduces subject-specific feature offsets while remaining applicable to an unseen test participant.

Two Extra Trees classifiers are used:

1. An eight-class activity classifier with `min_samples_leaf=4`.
2. A binary normal/unusual classifier with `min_samples_leaf=2` that gates the activity result.

The binary gate ensures that abnormal detection is learned directly rather than being only a side effect of eight-class classification.

## LOSO protocol

Four folds were evaluated. In each fold, one subject was held out completely and the other three subjects were used for training. No windows from the held-out subject were included in model fitting.

| Held-out subject | Windows | Accuracy | Abnormal precision | Abnormal recall | Abnormal F1 |
|---|---:|---:|---:|---:|---:|
| 1 | 2,419 | 0.6660 | 0.9803 | 0.7393 | 0.8429 |
| 2 | 2,360 | 0.7157 | 0.9209 | 0.7098 | 0.8017 |
| 3 | 3,525 | 0.5294 | 0.6830 | 0.4918 | 0.5718 |
| 5 | 2,378 | 0.6552 | 0.9978 | 0.7722 | 0.8706 |
| **Overall** | **10,682** | **0.6295** | **0.8825** | **0.6634** | **0.7574** |

The overall eight-class macro-F1 was 0.6279. Compared with the previous model, overall accuracy increased from 0.5500 to 0.6295 and abnormal F1 increased from 0.6279 to 0.7574. Subject 3 remains the most difficult held-out domain, but its accuracy increased from 0.4366 to 0.5294 and its abnormal F1 from 0.4189 to 0.5718.

## Objective A output

After LOSO evaluation, the final model was fitted on all 10,682 labeled windows from the four subjects. It was then applied to the currently available pose-only files in `keypoint/video_*.csv`. The combined output is `outputs/A.csv`; per-participant files are also available as `outputs/A_1.csv`, `outputs/A_2.csv`, `outputs/A_3.csv`, and `outputs/A_5.csv`. The same model and prediction command can be applied directly when a separate unseen-participant test file is supplied.

## Findings and limitations

Generalization differs substantially by subject. Subjects 1, 2, and 5 have high abnormal precision, but subject 3 produces many false abnormal predictions. This suggests a camera/viewpoint or performance-style domain shift that torso normalization does not fully remove.

Likely next improvements are multi-scale temporal features, left/right reflection augmentation, confidence-aware temporal smoothing, and threshold calibration using only training-subject inner validation folds. The current LOSO scores are window-level; the prediction command expands window predictions to every frame row in the required submission format.
