# Segmentation Feedback Handoff

This dashboard now has a committed feedback file that should be treated as the canonical segmentation supervision source:

- Feedback file: [artifacts/calibration_feedback.json](/Users/leekezar/Desktop/repositories/leekezar.github.io/asl-dashboard/artifacts/calibration_feedback.json)
- Dashboard UI / queue logic: [index.html](/Users/leekezar/Desktop/repositories/leekezar.github.io/asl-dashboard/index.html)
- Offline trainer: [train_segmentation_models.py](/Users/leekezar/Desktop/repositories/leekezar.github.io/asl-dashboard/train_segmentation_models.py)
- App data / features source: [assets/app_data.js](/Users/leekezar/Desktop/repositories/leekezar.github.io/asl-dashboard/assets/app_data.js)
- Local persistence server: [../serve_dashboard.py](/Users/leekezar/Desktop/repositories/leekezar.github.io/serve_dashboard.py)

## Current Data Snapshot

- Total feedback entries: `206`
- Boundary-calibration annotations: `204`
- Saved fit records: `2`
- Videos:
  - `coffee`: `105`
  - `eclipse`: `101`
- `notASign` annotations: `19`

Combined-choice counts in the current file:

- `good`: `38`
- `early_start_early_end`: `25`
- `inv_early_start_late_end`: `22`
- `early_start`: `18`
- `late_start`: `18`
- `late_end`: `17`
- `inv_late_start_early_end`: `15`
- `early_end`: `12`
- `late_start_late_end`: `8`
- `early_start_late_end`: `6`
- `late_start_early_end`: `6`

## What The Labels Mean

The UI now uses image buttons, but the trainer should continue to learn from the underlying start/end boundary labels. Each annotation stores:

- `combinedChoice`
- `startChoice`
- `endChoice`
- `referenceStartFrame`
- `referenceEndFrame`
- `referenceCenterFrame`
- `prevSpanFrames`
- `nextSpanFrames`
- `notASign`
- `configSnapshot`

Important: `combinedChoice` is display-level metadata. The training target is still encoded by `startChoice` and `endChoice`.

Current active boundary labels used in practice:

- Start side:
  - `good`
  - `early_0_25`
  - `late_0_25`
  - `invalid_early_50_plus`
- End side:
  - `good`
  - `early_0_25`
  - `late_0_25`
  - `invalid_early_50_plus`

The trainer still contains legacy support for older labels like `perfect`, `early_25_50`, `late_25_50`, and `invalid_50_plus`. Those are not the main labels in the committed feedback file, but backward compatibility is intentional.

## Feature Set

The current segmentation training setup uses these features per window/frame:

- left wrist speed
- right wrist speed
- left wrist acceleration
- right wrist acceleration
- left fingertip internal motion
- right fingertip internal motion
- ISR uncertainty
- visibility
- segment length

The feature extraction lives in `build_video_dataset(...)` in [train_segmentation_models.py](/Users/leekezar/Desktop/repositories/leekezar.github.io/asl-dashboard/train_segmentation_models.py).

## Current Trainer Behavior

`train_segmentation_models.py` does three things:

1. Loads dashboard app data and calibration feedback.
2. Trains several boundary classifiers over the extracted features.
3. Searches segmentation hyperparameters against the categorical calibration loss.

Supported model families in the script:

- logistic regression
- random forest
- extra trees
- hist gradient boosting
- MLP

The dashboard itself currently uses the simpler learned boundary model plus local hyperparameter search. The offline trainer is the place to do deeper sweeps and compare model families.

## Recommended Training Workflow

1. Use the committed feedback file, not browser localStorage.
2. Treat `startChoice` and `endChoice` as the supervision signal.
3. Keep `combinedChoice` only for reporting and UI diagnostics.
4. Evaluate per-video and pooled performance; do not assume coffee and eclipse behave the same.
5. Preserve `notASign` as a weak negative signal rather than a hard exclusion.
6. When adding new UI labels later, keep the mapping in the dashboard from image choice -> `startChoice`/`endChoice` explicit and versioned.

## Minimal Commands

From the repo root:

```bash
cd /Users/leekezar/Desktop/repositories/leekezar.github.io/asl-dashboard
```

Quick local sweep without W&B:

```bash
python3 train_segmentation_models.py --feedback-json artifacts/calibration_feedback.json --wandb-mode disabled
```

If a separate agent wants to log online:

```bash
python3 train_segmentation_models.py --feedback-json artifacts/calibration_feedback.json --wandb-mode online
```

## Notes For A Separate Agent

- Do not depend on Chrome localStorage dumps unless you are explicitly recovering unsaved work.
- The committed `artifacts/calibration_feedback.json` should be treated as the stable training input.
- If the dashboard label set changes, first update the dashboard mapping, then confirm the trainer still interprets the resulting `startChoice` / `endChoice` values correctly.
- If queue selection is being tuned, keep it separate from the supervised model training path. The queue is an active-learning policy, not the model itself.
