"""Final deployable pipeline: reject ICA artifacts with the trained feature classifier.

This is the method the ablation picked as "cleaner + automatic": a random forest
on 12 per-component features, trained once on the expert labels in labels/*.json,
then applied to any recording with no per-recording review.

Subcommands
-----------
train   fit the classifier on all labelled recordings, pick a global threshold by
        max F1, and save the model to results/final_rf.joblib.
apply   (default) run the full chain on each recording and write the cleaned
        .fif plus a per-recording rejection list under output/.  Trains first if
        no model exists.

Run:
    python -I apply_final.py              # apply to all 10 recordings
    python -I apply_final.py --subjects P01 P03
    python -I apply_final.py --retrain    # force re-train before applying
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import mne
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import ablation_lib as al
import run_stage2 as rs

FEATURES = [
    "var_explained", "kurtosis", "spectral_slope", "line_ratio",
    "eog_corr", "emg_corr", "cardiac_ac", "topo_frontal",
    "topo_peripheral", "topo_max", "topo_focality", "topo_edge",
]

MODEL_PATH = al.RESULT_DIR / "final_rf.joblib"
OUTPUT = al.OUT_ROOT / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def load_labeled() -> pd.DataFrame:
    frames = []
    for path in sorted(al.LABEL_DIR.glob("*_ses-*.json")):
        key = path.stem
        block = json.loads(path.read_text(encoding="utf-8"))
        feats = pd.read_csv(al.cache_dir(key) / "ic_features.csv")
        feats["key"] = key
        feats["expert_artifact"] = [
            int(block["labels"][str(i)] != "brain") for i in feats["ic"]
        ]
        frames.append(feats)
    return pd.concat(frames, ignore_index=True)


def _best_threshold(prob: np.ndarray, y: np.ndarray) -> float:
    best, best_f1 = 0.5, -1.0
    for t in np.unique(np.round(prob, 3)):
        tp = int(((prob >= t) & (y == 1)).sum())
        fp = int(((prob >= t) & (y == 0)).sum())
        fn = int(((prob < t) & (y == 1)).sum())
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom else 0.0
        if f1 > best_f1:
            best, best_f1 = float(t), f1
    return best


def train() -> dict:
    df = load_labeled()
    rf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, class_weight="balanced",
        random_state=0, n_jobs=1,
    ).fit(df[FEATURES].values, df["expert_artifact"].values)
    th = _best_threshold(
        rf.predict_proba(df[FEATURES].values)[:, 1],
        df["expert_artifact"].values,
    )
    out = {"model": rf, "threshold": th, "features": FEATURES}
    joblib.dump(out, MODEL_PATH)
    tp = int(((out["model"].predict_proba(df[FEATURES].values)[:, 1] >= th)
              & (df["expert_artifact"] == 1)).sum())
    fp = int(((out["model"].predict_proba(df[FEATURES].values)[:, 1] >= th)
              & (df["expert_artifact"] == 0)).sum())
    fn = int(((out["model"].predict_proba(df[FEATURES].values)[:, 1] < th)
              & (df["expert_artifact"] == 1)).sum())
    print(
        f"trained on {len(df)} ICs; threshold={th:.3f}; "
        f"tp={tp} fp={fp} fn={fn} "
        f"(precision {tp/(tp+fp):.3f}, recall {tp/(tp+fn):.3f})"
    )
    return out


# --------------------------------------------------------------------------
# application
# --------------------------------------------------------------------------
def _ensure_chain(key: str) -> tuple[mne.io.BaseRaw, mne.preprocessing.ICA, pd.DataFrame, dict]:
    """Return (preprocessed raw, ica, features, iclabel) for a recording,
    reusing the stage-1 cache when present and recomputing otherwise."""
    d = al.cache_dir(key)
    raw_fif, ica_fif, feat_csv, icl_json = (
        d / "raw_preproc.fif", d / "ica.fif", d / "ic_features.csv", d / "iclabel.json",
    )
    if raw_fif.exists() and ica_fif.exists() and feat_csv.exists():
        raw = mne.io.read_raw_fif(raw_fif, preload=True, verbose=False)
        ica = mne.preprocessing.read_ica(ica_fif, verbose=False)
        feats = pd.read_csv(feat_csv)
        icl = json.loads(icl_json.read_text(encoding="utf-8")) if icl_json.exists() else None
        return raw, ica, feats, icl

    subject, ses = key.split("_")[0], key.split("_ses-")[1]
    raw0 = al.load_raw(subject, ses)
    bads, _ = al.detect_bad_channels(raw0)
    raw = al.preprocess(raw0, bads)
    ica = al.fit_ica(raw)
    feats = al.component_features(raw, ica)
    icl = al.iclabel(raw, ica)
    return raw, ica, feats, icl


def apply_one(key: str, model: dict) -> dict:
    raw, ica, feats, icl = _ensure_chain(key)
    rf, th = model["model"], model["threshold"]
    prob = rf.predict_proba(feats[FEATURES].values)[:, 1]
    exclude = [int(i) for i, p in zip(feats["ic"], prob) if p >= th]

    cleaned = rs.apply_exclusion(raw, ica, exclude)
    out_fif = OUTPUT / f"{key}_desc-clean_eeg.fif"
    cleaned.save(out_fif, overwrite=True, verbose=False)

    rows = []
    for i, p in zip(feats["ic"], prob):
        rows.append({
            "ic": int(i),
            "artifact_prob": round(float(p), 4),
            "rejected": bool(i in set(exclude)),
            "iclabel_label": (icl["labels"][i] if icl else ""),
            "iclabel_prob": (round(float(icl["probs"][i]), 4) if icl else ""),
        })
    pd.DataFrame(rows).to_csv(OUTPUT / f"{key}_rejection.csv", index=False)

    # quick before/after quality readout (manual-exclude set is only needed for
    # the residual-leakage metric; here it is the classifier's own excludes)
    comps = ica.get_components()
    m = rs.signal_metrics(cleaned, comps, exclude)
    m.update(rs.retained_metrics(raw, cleaned))
    summary = {
        "key": key,
        "n_excluded": len(exclude),
        "excluded": exclude,
        "threshold": th,
        "emg_index": round(m["emg_index"], 3),
        "mean_kurtosis": round(m["mean_kurtosis"], 3),
        "var_retained": round(m["var_retained"], 3),
        "residual_artifact_var": round(m["residual_artifact_var"], 3),
        "cleaned_fif": str(out_fif),
    }
    print(
        f"{key}: reject {len(exclude):2d} ICs -> EMG {m['emg_index']:.3f}, "
        f"kurt {m['mean_kurtosis']:.1f}, var retained {m['var_retained']:.2f}"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="apply", choices=["train", "apply"])
    ap.add_argument("--subjects", nargs="*", default=al.SUBJECTS)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    if args.command == "train" or args.retrain or not MODEL_PATH.exists():
        train()

    if args.command == "train":
        print(f"saved model to {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)
    print(f"loaded model, global threshold = {model['threshold']:.3f}\n")
    summaries = []
    for subject in args.subjects:
        key = al.recording_key(subject)
        summaries.append(apply_one(key, model))

    (OUTPUT / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote cleaned files and rejection lists to {OUTPUT}")


if __name__ == "__main__":
    main()
