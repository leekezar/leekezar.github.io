#!/usr/bin/env python3
"""Offline HPO for ASL dashboard segmentation models.

This script reads the dashboard app data plus calibration feedback, builds the
requested feature set, trains five boundary models, searches segmentation
hyperparameters for each, and logs the results to Weights & Biases.
"""

from __future__ import annotations

import argparse
import base64
import codecs
import csv
import json
import math
import random
import re
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import wandb
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CALIBRATION_STORAGE_KEY = "asl-dashboard-feedback-v2"
CALIBRATION_TAIL_PRUNE_AFTER = 1774297500000
DEFAULT_WANDB_PROJECT = "asl-dashboard-segmentation-hpo"
TIP_OFFSETS = (4, 8, 12, 16, 20)

START_CALIBRATION_OPTIONS = [
    {"id": "invalid_early_50_plus", "label": "Invalid Start", "mode": "early", "min": 0.5, "max": math.inf, "ref": "prev"},
    {"id": "early_25_50", "label": "Very Early", "mode": "early", "min": 0.25, "max": 0.5, "ref": "prev"},
    {"id": "early_0_25", "label": "Early", "mode": "early", "min": 0.0, "max": 0.25, "ref": "prev"},
    {"id": "good", "label": "Good", "mode": "neutral"},
    {"id": "perfect", "label": "Perfect", "mode": "neutral"},
    {"id": "late_0_25", "label": "Late", "mode": "late", "min": 0.0, "max": 0.25, "ref": "current"},
    {"id": "late_25_50", "label": "Very Late", "mode": "late", "min": 0.25, "max": 0.5, "ref": "current"},
    {"id": "invalid_late_50_plus", "label": "Invalid Start", "mode": "late", "min": 0.5, "max": math.inf, "ref": "current"},
]

END_CALIBRATION_OPTIONS = [
    {"id": "invalid_early_50_plus", "label": "Invalid End", "mode": "early", "min": 0.5, "max": math.inf, "ref": "current"},
    {"id": "early_25_50", "label": "Very Early", "mode": "early", "min": 0.25, "max": 0.5, "ref": "current"},
    {"id": "early_0_25", "label": "Early", "mode": "early", "min": 0.0, "max": 0.25, "ref": "current"},
    {"id": "good", "label": "Good", "mode": "neutral"},
    {"id": "perfect", "label": "Perfect", "mode": "neutral"},
    {"id": "late_0_25", "label": "Late", "mode": "late", "min": 0.0, "max": 0.25, "ref": "next"},
    {"id": "late_25_50", "label": "Very Late", "mode": "late", "min": 0.25, "max": 0.5, "ref": "next"},
    {"id": "invalid_late_50_plus", "label": "Invalid End", "mode": "late", "min": 0.5, "max": math.inf, "ref": "next"},
]


@dataclass
class VideoDataset:
    video_id: str
    fps: float
    frames: int
    window_centers: np.ndarray
    top_uncertainty: np.ndarray
    features: np.ndarray
    combined_motion: np.ndarray
    combined_scale: float
    visible: np.ndarray
    base_segments: List[dict]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def smooth_array(source: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0:
        return source.copy()
    out = np.zeros_like(source, dtype=np.float32)
    for idx in range(source.shape[0]):
        start = max(0, idx - radius)
        end = min(source.shape[0], idx + radius + 1)
        out[idx] = float(np.mean(source[start:end]))
    return out


def finite_quantile(values: Iterable[float], q: float, fallback: float = 0.0) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return fallback
    return float(np.quantile(np.asarray(finite, dtype=np.float32), q))


def unpack_array(spec: dict) -> np.ndarray:
    raw = base64.b64decode(spec["base64"])
    dtype = spec["dtype"]
    if dtype == "int16":
        arr = np.frombuffer(raw, dtype=np.int16)
    elif dtype == "int32":
        arr = np.frombuffer(raw, dtype=np.int32)
    elif dtype == "float32":
        arr = np.frombuffer(raw, dtype=np.float32)
    elif dtype == "float16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return arr.copy()


def load_app_data(app_path: Path) -> dict:
    text = app_path.read_text()
    prefix = "window.__ASL_DASHBOARD_DATA__="
    if not text.startswith(prefix):
        raise ValueError(f"Unexpected app data format in {app_path}")
    return json.loads(text[len(prefix):-1])


def load_feedback_from_leveldb(leveldb_dir: Path, storage_key: str = CALIBRATION_STORAGE_KEY) -> dict:
    best_blob = None
    best_origin_score = -1
    preferred_origins = ("http://127.0.0.1:8768", "http://localhost:8768", "https://leekezar.github.io")
    for ldb_path in sorted(leveldb_dir.glob("*.ldb")):
        proc = subprocess.run(
            ["leveldbutil", "dump", str(ldb_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in proc.stdout.splitlines():
            if storage_key not in line or ": val => " not in line:
                continue
            match = re.search(r"val => '(.*)'$", line)
            if not match:
                continue
            encoded = match.group(1)
            decoded = codecs.decode(encoded, "unicode_escape")
            if decoded.startswith("\x01"):
                decoded = decoded[1:]
            origin_score = 0
            for idx, origin in enumerate(preferred_origins):
                if origin in line:
                    origin_score = len(preferred_origins) - idx
                    break
            if origin_score >= best_origin_score:
                best_origin_score = origin_score
                best_blob = decoded
    if not best_blob:
        raise FileNotFoundError(f"Could not find {storage_key} in {leveldb_dir}")
    return json.loads(best_blob)


def load_feedback_json(path: Path) -> dict:
    return json.loads(path.read_text())


def normalize_choice_id(choice_id: str, side: str) -> str:
    if choice_id == "invalid_50_plus":
        return "invalid_late_50_plus" if side == "start" else "invalid_early_50_plus"
    return choice_id


def prune_feedback(feedback: dict) -> dict:
    out = {}
    for key, entry in feedback.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") == "boundary-calibration" and (entry.get("createdAt") or 0) >= CALIBRATION_TAIL_PRUNE_AFTER:
            continue
        if entry.get("kind") == "segmentation-fit" and (entry.get("updatedAt") or 0) >= CALIBRATION_TAIL_PRUNE_AFTER:
            continue
        copied = dict(entry)
        if copied.get("kind") == "boundary-calibration":
            copied["startChoice"] = normalize_choice_id(copied.get("startChoice", ""), "start")
            copied["endChoice"] = normalize_choice_id(copied.get("endChoice", ""), "end")
        out[key] = copied
    return out


def calibration_entries(feedback: dict) -> List[dict]:
    return sorted(
        [entry for entry in feedback.values() if entry.get("kind") == "boundary-calibration"],
        key=lambda entry: entry.get("createdAt", 0),
    )


def split_entries(entries: List[dict], seed: int, val_fraction: float = 0.2) -> Tuple[List[dict], List[dict]]:
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for entry in entries:
        grouped["not_a_sign" if entry.get("notASign") else "boundary"].append(entry)
    train, val = [], []
    for bucket in grouped.values():
        bucket = list(bucket)
        rng.shuffle(bucket)
        val_count = max(1, int(round(len(bucket) * val_fraction))) if len(bucket) >= 5 else max(1, len(bucket) // 4)
        val.extend(bucket[:val_count])
        train.extend(bucket[val_count:])
    if not train:
        train, val = val[:-1], val[-1:]
    train.sort(key=lambda entry: entry.get("createdAt", 0))
    val.sort(key=lambda entry: entry.get("createdAt", 0))
    return train, val


def pose_point_at(decoded_pose: np.ndarray, payload: dict, frame_idx: int, point_idx: int, image_width: float, image_height: float):
    if frame_idx < 0 or frame_idx >= payload["frameCount"]:
        return None
    base = frame_idx * payload["pointCount"] * 2 + point_idx * 2
    raw_x = float(decoded_pose[base])
    raw_y = float(decoded_pose[base + 1])
    if not math.isfinite(raw_x) or not math.isfinite(raw_y):
        return None
    if raw_x == 0 and raw_y == 0:
        return None
    return (
        raw_x / payload["scale"] * image_width,
        raw_y / payload["scale"] * image_height,
    )


def fingertip_internal_motion(
    decoded_pose: np.ndarray,
    payload: dict,
    frame_idx: int,
    hand_start: int,
    image_width: float,
    image_height: float,
    hand_keypoints: int,
    tip_offsets: Tuple[int, ...] = TIP_OFFSETS,
) -> float:
    if frame_idx <= 0:
        return 0.0
    wrist_now = pose_point_at(decoded_pose, payload, frame_idx, hand_start, image_width, image_height)
    wrist_prev = pose_point_at(decoded_pose, payload, frame_idx - 1, hand_start, image_width, image_height)
    if wrist_now is None or wrist_prev is None:
        return 0.0
    values = []
    for offset in tip_offsets:
        if offset >= hand_keypoints:
            continue
        tip_now = pose_point_at(decoded_pose, payload, frame_idx, hand_start + offset, image_width, image_height)
        tip_prev = pose_point_at(decoded_pose, payload, frame_idx - 1, hand_start + offset, image_width, image_height)
        if tip_now is None or tip_prev is None:
            continue
        prev_rel_x = tip_prev[0] - wrist_prev[0]
        prev_rel_y = tip_prev[1] - wrist_prev[1]
        now_rel_x = tip_now[0] - wrist_now[0]
        now_rel_y = tip_now[1] - wrist_now[1]
        path_delta = math.hypot(now_rel_x - prev_rel_x, now_rel_y - prev_rel_y)
        scale = max(10.0, 0.5 * (math.hypot(prev_rel_x, prev_rel_y) + math.hypot(now_rel_x, now_rel_y)))
        values.append(path_delta / scale)
    return float(np.mean(values)) if values else 0.0


def frame_uncertainty(window_centers: np.ndarray, top_uncertainty: np.ndarray, frames: int, fps: float) -> np.ndarray:
    frame_prob = np.zeros(frames, dtype=np.float32)
    for center, value in zip(window_centers, top_uncertainty):
        frame = int(clamp(round(float(center)), 0, frames - 1))
        frame_prob[frame] = max(frame_prob[frame], float(value))
    radius = max(1, round(fps * 0.06))
    return smooth_array(frame_prob, radius)


def segment_length_per_frame(segments: List[dict], frames: int) -> np.ndarray:
    out = np.zeros(frames, dtype=np.float32)
    for seg in segments:
        length = max(1, int(seg["endFrame"]) - int(seg["startFrame"]))
        out[int(seg["startFrame"]):int(seg["endFrame"]) + 1] = length
    if not np.any(out):
        out[:] = 1
    return out


def build_video_dataset(app_data: dict, video_id: str) -> VideoDataset:
    payload = app_data["videos"][video_id]
    motion_spec = payload["motion"]
    pose_payload = payload["pose"]
    pose_spec = app_data["shared"]["poseSpec"]
    decoded_pose = unpack_array(payload["pose"])
    left_speed = unpack_array(motion_spec["leftSpeed"]).astype(np.float32)
    right_speed = unpack_array(motion_spec["rightSpeed"]).astype(np.float32)
    left_wrist = unpack_array(motion_spec["leftWrist"]).astype(np.float32)
    right_wrist = unpack_array(motion_spec["rightWrist"]).astype(np.float32)
    frames = int(pose_payload["frameCount"])
    fps = float(payload["videoInfo"]["fps"])
    speed_scale = float(motion_spec["speedScale"])
    image_width = float(motion_spec["imageWidth"])
    image_height = float(motion_spec["imageHeight"])
    hand_keypoints = int(pose_spec["handKeypoints"])
    left_hand_start = int(pose_spec["leftHandStart"])
    right_hand_start = int(pose_spec["rightHandStart"])

    visible = np.zeros(frames, dtype=np.uint8)
    left_internal = np.zeros(frames, dtype=np.float32)
    right_internal = np.zeros(frames, dtype=np.float32)
    for frame in range(frames):
        lx = float(left_wrist[frame * 2])
        ly = float(left_wrist[frame * 2 + 1])
        rx = float(right_wrist[frame * 2])
        ry = float(right_wrist[frame * 2 + 1])
        visible[frame] = int(math.isfinite(lx) and math.isfinite(ly)) + int(math.isfinite(rx) and math.isfinite(ry))
        left_internal[frame] = fingertip_internal_motion(
            decoded_pose, pose_payload, frame, left_hand_start, image_width, image_height, hand_keypoints
        )
        right_internal[frame] = fingertip_internal_motion(
            decoded_pose, pose_payload, frame, right_hand_start, image_width, image_height, hand_keypoints
        )

    internal_scale = max(0.04, finite_quantile(np.concatenate([left_internal, right_internal]), 0.9, 0.12))
    right_norm = np.clip(right_speed / max(1e-6, speed_scale), 0.0, 2.0)
    left_norm = np.clip(left_speed / max(1e-6, speed_scale), 0.0, 2.0)
    right_internal_norm = np.clip(right_internal / internal_scale, 0.0, 2.0)
    left_internal_norm = np.clip(left_internal / internal_scale, 0.0, 2.0)
    combined_motion = speed_scale * (
        0.56 * right_norm +
        0.18 * left_norm +
        0.18 * right_internal_norm +
        0.08 * left_internal_norm
    )
    combined_scale = max(1.0, finite_quantile(combined_motion, 0.9, speed_scale))

    left_accel = np.abs(np.gradient(left_speed.astype(np.float32))) / max(1e-6, speed_scale)
    right_accel = np.abs(np.gradient(right_speed.astype(np.float32))) / max(1e-6, speed_scale)

    window_starts = np.asarray(payload["windowData"]["starts"], dtype=np.float32)
    window_ends = np.asarray(payload["windowData"]["ends"], dtype=np.float32)
    window_centers = 0.5 * (window_starts + window_ends)
    sign_top_prob_spec = payload["windowData"]["signTopProb"]
    sign_top_prob = unpack_array(sign_top_prob_spec).astype(np.float32)
    top_k = int(sign_top_prob_spec["shape"][1])
    top_uncertainty = 1.0 - sign_top_prob[::top_k]
    frame_unc = frame_uncertainty(window_centers, top_uncertainty, frames, fps)

    base_segments = [
        {
            "id": int(seg["id"]),
            "startFrame": int(seg["startFrame"]),
            "endFrame": int(seg["endFrame"]),
        }
        for seg in payload["segments"]
    ]
    length_by_frame = segment_length_per_frame(base_segments, frames)

    rows = []
    for center, unc in zip(window_centers, top_uncertainty):
        frame = int(clamp(round(float(center)), 0, frames - 1))
        rows.append([
            left_speed[frame] / max(1e-6, speed_scale),
            right_speed[frame] / max(1e-6, speed_scale),
            left_accel[frame],
            right_accel[frame],
            left_internal[frame] / internal_scale,
            right_internal[frame] / internal_scale,
            frame_unc[frame] if frame_unc.shape[0] else float(unc),
            visible[frame] / 2.0,
            length_by_frame[frame] / max(1.0, fps),
        ])

    return VideoDataset(
        video_id=video_id,
        fps=fps,
        frames=frames,
        window_centers=window_centers.astype(np.float32),
        top_uncertainty=top_uncertainty.astype(np.float32),
        features=np.asarray(rows, dtype=np.float32),
        combined_motion=combined_motion.astype(np.float32),
        combined_scale=float(combined_scale),
        visible=visible,
        base_segments=base_segments,
    )


def boundary_training_examples(video_data: VideoDataset, entries: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    positives = []
    for idx, seg in enumerate(video_data.base_segments):
        if idx > 0:
            positives.append((int(seg["startFrame"]), 0.7))
        if idx < len(video_data.base_segments) - 1:
            positives.append((int(seg["endFrame"]), 0.7))
    not_sign_ranges = []
    for entry in entries:
        if entry.get("notASign"):
            not_sign_ranges.append((int(entry["referenceStartFrame"]), int(entry["referenceEndFrame"])))
            continue
        positives.append((int(entry["referenceStartFrame"]), 2.4))
        positives.append((int(entry["referenceEndFrame"]), 2.4))
    pos_radius = max(4, round(video_data.fps * 0.12))
    neg_radius = max(9, round(video_data.fps * 0.30))

    x_rows, y_rows, w_rows = [], [], []
    for idx, center in enumerate(video_data.window_centers):
        frame = int(round(float(center)))
        inside_not_sign = any(start <= frame <= end for start, end in not_sign_ranges)
        dist_to_boundary = min((abs(frame - boundary) for boundary, _ in positives), default=10 ** 9)
        strongest = None
        for boundary, weight in positives:
            dist = abs(frame - boundary)
            if dist > pos_radius:
                continue
            if strongest is None or weight > strongest[0]:
                strongest = (weight, dist)
        label = None
        weight = 0.0
        if inside_not_sign:
            label = 0
            weight = 0.22
        elif strongest is not None:
            label = 1
            weight = strongest[0] * (1 - strongest[1] / (pos_radius + 1))
        elif dist_to_boundary >= neg_radius:
            label = 0
            weight = 0.85
        if label is None or weight <= 0:
            continue
        x_rows.append(video_data.features[idx])
        y_rows.append(label)
        w_rows.append(weight)
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int32), np.asarray(w_rows, dtype=np.float32)


def model_search_space(model_name: str, rng: random.Random) -> dict:
    if model_name == "logistic_regression":
        return {
            "C": 10 ** rng.uniform(-2.0, 1.0),
            "class_weight": rng.choice([None, "balanced"]),
            "max_iter": 2000,
        }
    if model_name == "random_forest":
        return {
            "n_estimators": rng.choice([96, 128, 192, 256]),
            "max_depth": rng.choice([4, 6, 8, 12, None]),
            "min_samples_leaf": rng.choice([1, 2, 4, 6, 8]),
            "max_features": rng.choice(["sqrt", "log2", 0.55, 0.8, None]),
            "class_weight": rng.choice([None, "balanced"]),
            "n_jobs": -1,
        }
    if model_name == "extra_trees":
        return {
            "n_estimators": rng.choice([96, 128, 192, 256]),
            "max_depth": rng.choice([4, 6, 8, 12, None]),
            "min_samples_leaf": rng.choice([1, 2, 4, 6, 8]),
            "max_features": rng.choice(["sqrt", "log2", 0.55, 0.8, None]),
            "class_weight": rng.choice([None, "balanced"]),
            "n_jobs": -1,
        }
    if model_name == "hist_gradient_boosting":
        return {
            "learning_rate": rng.choice([0.03, 0.05, 0.08, 0.12, 0.18]),
            "max_depth": rng.choice([3, 4, 5, 6, None]),
            "max_leaf_nodes": rng.choice([15, 31, 63]),
            "min_samples_leaf": rng.choice([5, 10, 15, 20]),
            "l2_regularization": rng.choice([0.0, 1e-3, 1e-2, 1e-1]),
            "max_iter": rng.choice([120, 180, 240]),
        }
    if model_name == "mlp":
        return {
            "hidden_layer_sizes": rng.choice([(16,), (24,), (32,), (32, 16), (48, 24)]),
            "activation": rng.choice(["tanh", "relu"]),
            "alpha": 10 ** rng.uniform(-5.0, -2.0),
            "learning_rate_init": rng.choice([1e-3, 2e-3, 3e-3, 1e-2]),
            "max_iter": 800,
            "early_stopping": True,
            "n_iter_no_change": 25,
        }
    raise ValueError(f"Unknown model family: {model_name}")


def build_estimator(model_name: str, params: dict):
    if model_name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params)),
        ])
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=0, **params)
    if model_name == "extra_trees":
        return ExtraTreesClassifier(random_state=0, **params)
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=0, **params)
    if model_name == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(random_state=0, **params)),
        ])
    raise ValueError(f"Unknown model family: {model_name}")


def segmentation_space(rng: random.Random) -> dict:
    return {
        "threshold_mult": round(rng.uniform(0.38, 0.85), 2),
        "ema_before_sec": round(rng.uniform(0.1, 1.0), 2),
        "ema_after_sec": round(rng.uniform(0.0, 0.25), 2),
        "smoothing": rng.choice(["mean", "median", "ema"]),
        "min_hold_frames": rng.choice([2, 3, 4, 5]),
        "min_seg_frames": rng.choice([8, 10, 12, 14, 18, 24, 30]),
        "buffer_frames": rng.choice([1, 2, 3, 4]),
        "split_threshold": round(rng.uniform(0.38, 0.82), 2),
    }


def motion_threshold(video_data: VideoDataset, cfg: dict) -> np.ndarray:
    out = np.zeros(video_data.frames, dtype=np.float32)
    base_speed = video_data.combined_motion
    past_radius = max(1, round(video_data.fps * cfg["ema_before_sec"]))
    future_radius = max(0, round(video_data.fps * cfg["ema_after_sec"]))
    speed_vals = sorted(float(base_speed[idx]) for idx in range(video_data.frames) if video_data.visible[idx] > 0)
    q12_idx = max(0, int(math.floor((len(speed_vals) - 1) * 0.12))) if speed_vals else 0
    local_floor = max(4.5, speed_vals[q12_idx] if speed_vals else 4.5)
    for idx in range(video_data.frames):
        start = max(0, idx - past_radius)
        end = min(video_data.frames - 1, idx + future_radius)
        window = base_speed[start:end + 1]
        if cfg["smoothing"] == "median":
            smoothed = float(np.median(window))
        elif cfg["smoothing"] == "ema":
            tau = max(1.0, (past_radius + future_radius + 1) * 0.35)
            positions = np.arange(start, end + 1)
            weights = np.exp(-np.abs(positions - idx) / tau)
            smoothed = float(np.sum(window * weights) / np.sum(weights))
        else:
            smoothed = float(np.mean(window))
        out[idx] = max(local_floor, min(video_data.combined_scale * 0.62, smoothed * cfg["threshold_mult"]))
    return out


def frame_boundary_prob(video_data: VideoDataset, window_prob: np.ndarray) -> np.ndarray:
    frame_prob = np.zeros(video_data.frames, dtype=np.float32)
    for center, prob in zip(video_data.window_centers, window_prob):
        frame = int(clamp(round(float(center)), 0, video_data.frames - 1))
        frame_prob[frame] = max(frame_prob[frame], float(prob))
    return smooth_array(frame_prob, max(1, round(video_data.fps * 0.06)))


def generate_segment_ranges(video_data: VideoDataset, window_prob: np.ndarray, cfg: dict) -> List[dict]:
    threshold = motion_threshold(video_data, cfg)
    frame_prob = frame_boundary_prob(video_data, window_prob)
    holds = []
    run_start = -1
    for frame in range(video_data.frames):
        any_visible = video_data.visible[frame] > 0
        still = any_visible and video_data.combined_motion[frame] <= threshold[frame]
        if still and run_start < 0:
            run_start = frame
        if run_start >= 0 and ((not still) or frame == video_data.frames - 1):
            run_end = frame if still and frame == video_data.frames - 1 else frame - 1
            if run_end - run_start + 1 >= cfg["min_hold_frames"]:
                best_frame = run_start
                best_score = float("inf")
                for idx in range(run_start, run_end + 1):
                    score = video_data.combined_motion[idx] - frame_prob[idx] * video_data.combined_scale * 0.12
                    if score < best_score:
                        best_score = score
                        best_frame = idx
                holds.append({"startFrame": run_start, "endFrame": run_end, "frame": best_frame})
            run_start = -1

    boundaries = [0] + [hold["frame"] for hold in holds] + [video_data.frames - 1]
    boundaries = [frame for idx, frame in enumerate(boundaries) if idx == 0 or frame > boundaries[idx - 1]]
    intervals = []
    for idx in range(len(boundaries) - 1):
        start_frame = boundaries[idx]
        end_frame = boundaries[idx + 1]
        if np.any(video_data.visible[start_frame:end_frame + 1] > 0):
            intervals.append({"startFrame": start_frame, "endFrame": end_frame})

    generated = []
    next_id = max(seg["id"] for seg in video_data.base_segments) + 1
    min_gap = max(8, round(cfg["min_seg_frames"] * 0.75))
    min_seg = max(10, round(cfg["min_seg_frames"]))

    for interval in intervals:
        span = interval["endFrame"] - interval["startFrame"]
        if span < min_seg:
            continue
        split_frame = None
        candidate_idx = []
        for idx, center in enumerate(video_data.window_centers):
            if center >= interval["startFrame"] + min_gap and center <= interval["endFrame"] - min_gap:
                candidate_idx.append(idx)
        if span >= round(video_data.fps * 1.2) and len(candidate_idx) >= 5:
            best_candidate = None
            for pos in range(1, len(candidate_idx) - 1):
                prev_idx = candidate_idx[pos - 1]
                cur_idx = candidate_idx[pos]
                next_idx = candidate_idx[pos + 1]
                prev_unc = video_data.top_uncertainty[prev_idx]
                cur_unc = video_data.top_uncertainty[cur_idx]
                next_unc = video_data.top_uncertainty[next_idx]
                prev_prob = window_prob[prev_idx]
                cur_prob = window_prob[cur_idx]
                next_prob = window_prob[next_idx]
                uncertainty_peak = cur_unc >= prev_unc and cur_unc >= next_unc
                model_peak = cur_prob >= prev_prob and cur_prob >= next_prob
                if not uncertainty_peak and not model_peak:
                    continue
                score = 0.72 * float(cur_prob) + 0.28 * float(cur_unc)
                if score < cfg["split_threshold"]:
                    continue
                if best_candidate is None or score > best_candidate[0]:
                    best_candidate = (score, int(round(float(video_data.window_centers[cur_idx]))))
            if best_candidate is not None:
                split_frame = best_candidate[1]
        ranges = (
            [(interval["startFrame"], split_frame), (split_frame, interval["endFrame"])]
            if split_frame is not None
            else [(interval["startFrame"], interval["endFrame"])]
        )
        for start_frame, end_frame in ranges:
            if end_frame - start_frame >= min_seg:
                generated.append({"id": next_id, "startFrame": start_frame, "endFrame": end_frame})
                next_id += 1

    generated.sort(key=lambda seg: seg["startFrame"])
    final_ranges = []
    prev_end = 0
    for idx, seg in enumerate(generated):
        start_frame = max(seg["startFrame"], prev_end)
        end_frame = seg["endFrame"]
        if idx + 1 < len(generated):
            end_frame = min(end_frame, generated[idx + 1]["startFrame"] - cfg["buffer_frames"])
        if end_frame - start_frame < min_seg:
            continue
        final_ranges.append({"id": seg["id"], "startFrame": start_frame, "endFrame": end_frame})
        prev_end = end_frame + cfg["buffer_frames"]
    return final_ranges


def matching_segment_for_range(segments: List[dict], start_frame: int, end_frame: int):
    best = None
    best_overlap = -1
    best_center_dist = float("inf")
    target_center = 0.5 * (start_frame + end_frame)
    for seg in segments:
        overlap = max(0, min(end_frame, seg["endFrame"]) - max(start_frame, seg["startFrame"]))
        center = 0.5 * (seg["startFrame"] + seg["endFrame"])
        center_dist = abs(center - target_center)
        if overlap > best_overlap or (overlap == best_overlap and center_dist < best_center_dist):
            best = seg
            best_overlap = overlap
            best_center_dist = center_dist
    return best


def calibration_scale(entry: dict, choice: dict) -> float:
    current_span = max(10, int(entry["referenceEndFrame"]) - int(entry["referenceStartFrame"]))
    if choice.get("ref") == "prev":
        return max(10, int(entry.get("prevSpanFrames") or current_span))
    if choice.get("ref") == "next":
        return max(10, int(entry.get("nextSpanFrames") or current_span))
    return float(current_span)


def interval_loss(value: float, minimum: float, maximum: float, scale: float) -> float:
    unit = max(6.0, scale * 0.18)
    if value < minimum:
        return (minimum - value) / unit
    if value > maximum:
        return (value - maximum) / unit
    return 0.0


def boundary_loss_for_choice(choice_id: str, delta: float, entry: dict, options: List[dict], side: str) -> float:
    normalized = normalize_choice_id(choice_id, side)
    choice = next((item for item in options if item["id"] == normalized), None)
    if choice is None:
        return 1.2
    current_span = max(10, int(entry["referenceEndFrame"]) - int(entry["referenceStartFrame"]))
    if choice["mode"] == "neutral":
        tol = max(3, current_span * 0.04) if normalized == "perfect" else max(6, current_span * 0.12)
        return max(0.0, abs(delta) - tol) / tol
    scale = calibration_scale(entry, choice)
    if choice["mode"] == "late":
        lower = choice["min"] * scale
        upper = choice["max"] * scale if math.isfinite(choice["max"]) else math.inf
    else:
        lower = -choice["max"] * scale if math.isfinite(choice["max"]) else -math.inf
        upper = -choice["min"] * scale
    return interval_loss(delta, lower, upper, scale)


def calibration_loss_for_entry(entry: dict, segments: List[dict]) -> float:
    if entry.get("notASign"):
        match = matching_segment_for_range(segments, int(entry["referenceStartFrame"]), int(entry["referenceEndFrame"]))
        if match is None:
            return 0.0
        overlap = max(
            0,
            min(int(entry["referenceEndFrame"]), match["endFrame"]) - max(int(entry["referenceStartFrame"]), match["startFrame"]),
        )
        ref_span = max(10, int(entry["referenceEndFrame"]) - int(entry["referenceStartFrame"]))
        return 0.22 * min(1.5, overlap / ref_span)
    match = matching_segment_for_range(segments, int(entry["referenceStartFrame"]), int(entry["referenceEndFrame"]))
    if match is None:
        return 3.5
    delta_start = match["startFrame"] - int(entry["referenceStartFrame"])
    delta_end = match["endFrame"] - int(entry["referenceEndFrame"])
    start_loss = boundary_loss_for_choice(entry.get("startChoice", ""), delta_start, entry, START_CALIBRATION_OPTIONS, "start")
    end_loss = boundary_loss_for_choice(entry.get("endChoice", ""), delta_end, entry, END_CALIBRATION_OPTIONS, "end")
    overlap = max(
        0,
        min(int(entry["referenceEndFrame"]), match["endFrame"]) - max(int(entry["referenceStartFrame"]), match["startFrame"]),
    )
    ref_span = max(10, int(entry["referenceEndFrame"]) - int(entry["referenceStartFrame"]))
    overlap_penalty = max(0.0, 0.6 - overlap / ref_span) * 1.2
    return start_loss + end_loss + overlap_penalty


def fit_percent_for_entries(entries: List[dict], segments_by_video: Dict[str, List[dict]]) -> Tuple[float, float]:
    if not entries:
        return 0.0, float("inf")
    losses = []
    for entry in entries:
        losses.append(calibration_loss_for_entry(entry, segments_by_video.get(entry["videoId"], [])))
    mean_loss = float(np.mean(losses))
    fit_percent = float(round(100 * math.exp(-0.7 * mean_loss)))
    return fit_percent, mean_loss


def train_model(
    model_name: str,
    model_params: dict,
    video_datasets: Dict[str, VideoDataset],
    train_entries_by_video: Dict[str, List[dict]],
) -> Tuple[object, Dict[str, np.ndarray], dict]:
    x_rows, y_rows, w_rows = [], [], []
    for video_id, video_data in video_datasets.items():
        x, y, weights = boundary_training_examples(video_data, train_entries_by_video.get(video_id, []))
        if x.size == 0:
            continue
        x_rows.append(x)
        y_rows.append(y)
        w_rows.append(weights)
    x_train = np.concatenate(x_rows, axis=0)
    y_train = np.concatenate(y_rows, axis=0)
    w_train = np.concatenate(w_rows, axis=0)

    estimator = build_estimator(model_name, model_params)
    train_start = time.perf_counter()
    fit_kwargs = {}
    if model_name == "logistic_regression":
        fit_kwargs["clf__sample_weight"] = w_train
    elif model_name in {"random_forest", "extra_trees", "hist_gradient_boosting"}:
        fit_kwargs["sample_weight"] = w_train
    elif model_name == "mlp":
        fit_kwargs["clf__sample_weight"] = w_train
    estimator.fit(x_train, y_train, **fit_kwargs)
    train_ms = (time.perf_counter() - train_start) * 1000.0

    inference_start = time.perf_counter()
    window_prob = {}
    for video_id, video_data in video_datasets.items():
        probs = estimator.predict_proba(video_data.features)[:, 1]
        window_prob[video_id] = probs.astype(np.float32)
    inference_ms = (time.perf_counter() - inference_start) * 1000.0

    try:
        train_pr_auc = float(average_precision_score(y_train, estimator.predict_proba(x_train)[:, 1], sample_weight=w_train))
    except ValueError:
        train_pr_auc = 0.0
    try:
        train_roc_auc = float(roc_auc_score(y_train, estimator.predict_proba(x_train)[:, 1], sample_weight=w_train))
    except ValueError:
        train_roc_auc = 0.0

    metrics = {
        "train_ms": train_ms,
        "infer_ms": inference_ms,
        "train_examples": int(x_train.shape[0]),
        "train_positive_rate": float(np.mean(y_train)),
        "train_pr_auc": train_pr_auc,
        "train_roc_auc": train_roc_auc,
    }
    return estimator, window_prob, metrics


def sample_unique(fn, budget: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    seen = set()
    values = []
    while len(values) < budget:
        candidate = fn(rng)
        key = json.dumps(candidate, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        values.append(candidate)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="HPO for ASL dashboard segmentation boundary models.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent, help="Path to asl-dashboard directory")
    parser.add_argument("--feedback-json", type=Path, default=None, help="Path to committed calibration feedback JSON")
    parser.add_argument("--chrome-leveldb", type=Path, default=Path("~/Library/Application Support/Google/Chrome/Default/Local Storage/leveldb").expanduser())
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--model-trials", type=int, default=14)
    parser.add_argument("--seg-trials", type=int, default=18)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    app_data = load_app_data(args.repo_root / "assets" / "app_data.js")
    feedback_json_path = args.feedback_json or (args.repo_root / "artifacts" / "calibration_feedback.json")
    if feedback_json_path.exists():
        raw_feedback = load_feedback_json(feedback_json_path)
        feedback_source = str(feedback_json_path)
    else:
        raw_feedback = load_feedback_from_leveldb(args.chrome_leveldb)
        feedback_source = str(args.chrome_leveldb)
    feedback = prune_feedback(raw_feedback)
    entries = calibration_entries(feedback)
    if not entries:
        raise SystemExit("No boundary calibration entries found.")

    annotation_count = len(entries)
    print(f"Loaded {annotation_count} calibration annotations from {feedback_source}.")
    if annotation_count < 185:
        print(f"Warning: only {annotation_count} annotations found. The sweep will still run, but model ranking will be less stable.")
    else:
        print(f"{annotation_count} annotations is enough for a meaningful lightweight model sweep. I would want more for a larger neural model.")

    videos_with_annotations = sorted({entry["videoId"] for entry in entries})
    video_datasets = {video_id: build_video_dataset(app_data, video_id) for video_id in videos_with_annotations}
    train_entries, val_entries = split_entries(entries, args.seed)
    train_entries_by_video = defaultdict(list)
    val_entries_by_video = defaultdict(list)
    for entry in train_entries:
        train_entries_by_video[entry["videoId"]].append(entry)
    for entry in val_entries:
        val_entries_by_video[entry["videoId"]].append(entry)

    run = None
    if args.wandb_mode != "disabled":
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            config={
                "annotation_count": annotation_count,
                "videos": videos_with_annotations,
                "model_trials": args.model_trials,
                "seg_trials": args.seg_trials,
                "seed": args.seed,
                "feedback_source": feedback_source,
                "features": [
                    "left_wrist_speed",
                    "right_wrist_speed",
                    "left_wrist_acceleration",
                    "right_wrist_acceleration",
                    "left_finger_movement",
                    "right_finger_movement",
                    "isr_uncertainty",
                    "visibility",
                    "segment_length_sec",
                ],
            },
        )

    model_families = [
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "mlp",
    ]

    trial_rows = []
    best_row = None
    for family_idx, model_name in enumerate(model_families):
        model_param_candidates = sample_unique(lambda rng: model_search_space(model_name, rng), args.model_trials, args.seed + family_idx * 101)
        seg_param_candidates = sample_unique(segmentation_space, args.seg_trials, args.seed + family_idx * 313)
        family_best = None
        for model_trial_idx, model_params in enumerate(model_param_candidates):
            estimator, window_prob, train_metrics = train_model(model_name, model_params, video_datasets, train_entries_by_video)
            best_seg = None
            for seg_params in seg_param_candidates:
                segment_start = time.perf_counter()
                val_segments = {
                    video_id: generate_segment_ranges(video_datasets[video_id], window_prob[video_id], seg_params)
                    for video_id in videos_with_annotations
                }
                segment_ms = (time.perf_counter() - segment_start) * 1000.0
                val_fit, val_mean_loss = fit_percent_for_entries(val_entries, val_segments)
                row = {
                    "model_family": model_name,
                    "model_trial": model_trial_idx,
                    "model_params": model_params,
                    "segmentation_params": seg_params,
                    "val_fit_percent": float(val_fit),
                    "val_mean_loss": float(val_mean_loss),
                    "train_ms": float(train_metrics["train_ms"]),
                    "infer_ms": float(train_metrics["infer_ms"]),
                    "segment_ms": float(segment_ms),
                    "total_ms": float(train_metrics["train_ms"] + train_metrics["infer_ms"] + segment_ms),
                    "train_examples": int(train_metrics["train_examples"]),
                    "train_positive_rate": float(train_metrics["train_positive_rate"]),
                    "train_pr_auc": float(train_metrics["train_pr_auc"]),
                    "train_roc_auc": float(train_metrics["train_roc_auc"]),
                }
                if best_seg is None or row["val_fit_percent"] > best_seg["val_fit_percent"] or (
                    row["val_fit_percent"] == best_seg["val_fit_percent"] and row["total_ms"] < best_seg["total_ms"]
                ):
                    best_seg = row
                trial_rows.append(row)
                if run is not None:
                    wandb.log({
                        "trial/model_family": model_name,
                        "trial/val_fit_percent": row["val_fit_percent"],
                        "trial/val_mean_loss": row["val_mean_loss"],
                        "trial/total_ms": row["total_ms"],
                        "trial/train_pr_auc": row["train_pr_auc"],
                        "trial/train_roc_auc": row["train_roc_auc"],
                    })
            if family_best is None or best_seg["val_fit_percent"] > family_best["val_fit_percent"] or (
                best_seg["val_fit_percent"] == family_best["val_fit_percent"] and best_seg["total_ms"] < family_best["total_ms"]
            ):
                family_best = best_seg
        if best_row is None or family_best["val_fit_percent"] > best_row["val_fit_percent"] or (
            family_best["val_fit_percent"] == best_row["val_fit_percent"] and family_best["total_ms"] < best_row["total_ms"]
        ):
            best_row = family_best

    assert best_row is not None
    best_model_name = best_row["model_family"]
    final_estimator, final_window_prob, final_train_metrics = train_model(
        best_model_name,
        best_row["model_params"],
        video_datasets,
        defaultdict(list, {video_id: [entry for entry in entries if entry["videoId"] == video_id] for video_id in videos_with_annotations}),
    )
    final_segments = {
        video_id: generate_segment_ranges(video_datasets[video_id], final_window_prob[video_id], best_row["segmentation_params"])
        for video_id in videos_with_annotations
    }
    full_fit, full_mean_loss = fit_percent_for_entries(entries, final_segments)
    best_row["full_fit_percent"] = float(full_fit)
    best_row["full_mean_loss"] = float(full_mean_loss)
    best_row["final_train_ms"] = float(final_train_metrics["train_ms"])
    best_row["final_infer_ms"] = float(final_train_metrics["infer_ms"])

    output_dir = args.repo_root / "artifacts"
    output_dir.mkdir(exist_ok=True)
    results_json = output_dir / "segmentation_model_search_results.json"
    results_csv = output_dir / "segmentation_model_search_results.csv"
    best_json = output_dir / "best_segmentation_model.json"

    with results_json.open("w") as handle:
        json.dump(
            {
                "annotation_count": annotation_count,
                "train_count": len(train_entries),
                "val_count": len(val_entries),
                "videos": videos_with_annotations,
                "trials": trial_rows,
                "best": best_row,
            },
            handle,
            indent=2,
        )
    with results_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_family",
                "model_trial",
                "val_fit_percent",
                "val_mean_loss",
                "train_ms",
                "infer_ms",
                "segment_ms",
                "total_ms",
                "train_examples",
                "train_positive_rate",
                "train_pr_auc",
                "train_roc_auc",
                "model_params",
                "segmentation_params",
            ],
        )
        writer.writeheader()
        for row in trial_rows:
            writer.writerow({
                "model_family": row["model_family"],
                "model_trial": row["model_trial"],
                "val_fit_percent": row["val_fit_percent"],
                "val_mean_loss": row["val_mean_loss"],
                "train_ms": row["train_ms"],
                "infer_ms": row["infer_ms"],
                "segment_ms": row["segment_ms"],
                "total_ms": row["total_ms"],
                "train_examples": row["train_examples"],
                "train_positive_rate": row["train_positive_rate"],
                "train_pr_auc": row["train_pr_auc"],
                "train_roc_auc": row["train_roc_auc"],
                "model_params": json.dumps(row["model_params"]),
                "segmentation_params": json.dumps(row["segmentation_params"]),
            })
    with best_json.open("w") as handle:
        json.dump(
            {
                "annotation_count": annotation_count,
                "best_model_family": best_model_name,
                "best_model_params": best_row["model_params"],
                "best_segmentation_params": best_row["segmentation_params"],
                "validation_fit_percent": best_row["val_fit_percent"],
                "validation_mean_loss": best_row["val_mean_loss"],
                "full_fit_percent": full_fit,
                "full_mean_loss": full_mean_loss,
                "final_train_ms": final_train_metrics["train_ms"],
                "final_infer_ms": final_train_metrics["infer_ms"],
            },
            handle,
            indent=2,
        )

    summary_lines = [
        f"annotations: {annotation_count}",
        f"train/val: {len(train_entries)}/{len(val_entries)}",
        f"best model: {best_model_name}",
        f"validation fit: {best_row['val_fit_percent']:.1f}%",
        f"full fit: {full_fit:.1f}%",
        f"results: {results_json}",
    ]
    print("\n".join(summary_lines))

    if run is not None:
        wandb.summary["annotation_count"] = annotation_count
        wandb.summary["best_model_family"] = best_model_name
        wandb.summary["best_validation_fit_percent"] = best_row["val_fit_percent"]
        wandb.summary["best_full_fit_percent"] = full_fit
        wandb.summary["best_model_params"] = json.dumps(best_row["model_params"], sort_keys=True)
        wandb.summary["best_segmentation_params"] = json.dumps(best_row["segmentation_params"], sort_keys=True)
        table = wandb.Table(
            columns=[
                "model_family",
                "model_trial",
                "val_fit_percent",
                "val_mean_loss",
                "total_ms",
                "train_pr_auc",
                "train_roc_auc",
                "model_params",
                "segmentation_params",
            ]
        )
        sorted_rows = sorted(trial_rows, key=lambda row: (-row["val_fit_percent"], row["total_ms"]))[:50]
        for row in sorted_rows:
            table.add_data(
                row["model_family"],
                row["model_trial"],
                row["val_fit_percent"],
                row["val_mean_loss"],
                row["total_ms"],
                row["train_pr_auc"],
                row["train_roc_auc"],
                json.dumps(row["model_params"], sort_keys=True),
                json.dumps(row["segmentation_params"], sort_keys=True),
            )
        wandb.log({"top_trials": table})
        run.finish()


if __name__ == "__main__":
    main()
