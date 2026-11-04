"""Stage 2: apply manual / automatic IC rejection and evaluate both arms.

Prerequisites:
  * stage 1 cache (raw_preproc.fif, ica.fif, iclabel.json, events.csv)
  * labels/manual_labels.json with the expert labels for every recording

Run:  python -I run_stage2.py [--subjects P01 ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, wilcoxon

import ablation_lib as al

BANDS = {"theta": (4, 8), "mu": (8, 13), "beta": (13, 30), "low_gamma": (30, 40)}
DECODE_WINDOW = (1.0, 7.0)  # seconds after cue onset
N_FOLDS = 5


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------
def apply_exclusion(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA,
                    exclude: list[int]) -> mne.io.BaseRaw:
    r = raw.copy().load_data()
    ic = ica.copy()
    ic.exclude = list(exclude)
    ic.apply(r, verbose=False)
    return r


def manual_exclude(labels: dict[str, str]) -> list[int]:
    return [int(k) for k, v in labels.items() if v != "brain"]


# --------------------------------------------------------------------------
# signal-quality metrics
# --------------------------------------------------------------------------
def _pow(data: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    from mne.filter import filter_data

    return filter_data(data, sfreq=fs, l_freq=lo, h_freq=hi, verbose=False).var(axis=1)


def _idx(names: list[str], wanted: list[str]) -> list[int]:
    return [names.index(c) for c in wanted if c in names]


def signal_metrics(raw: mne.io.BaseRaw, comps: np.ndarray,
                   artifact_ics: list[int]) -> dict:
    fs = float(raw.info["sfreq"])
    names = raw.ch_names
    X = raw.get_data(picks="eeg")

    fi = _idx(names, al.FRONTAL)
    pi = _idx(names, al.PERIPHERAL)
    si = _idx(names, al.SENSORIMOTOR)

    front_low = _pow(X[fi], fs, 1, 7).sum()
    front_all = _pow(X[fi], fs, 1, 40).sum()
    periph_emg = _pow(X[pi], fs, 20, 45).sum()
    periph_all = _pow(X[pi], fs, 4, 45).sum()
    line = _pow(X, fs, 49, 51).sum() / (_pow(X, fs, 45, 55).sum() + 1e-30)
    alpha = np.log(_pow(X[si], fs, 8, 13) + 1e-30).mean()
    beta = np.log(_pow(X[si], fs, 13, 30) + 1e-30).mean()
    kurt = float(np.mean(kurtosis(X, axis=1, fisher=True)))

    # residual variance in the subspace of artifact IC topographies
    resid = np.nan
    if artifact_ics:
        A = comps[:, artifact_ics]
        coef, *_ = np.linalg.lstsq(A, X - X.mean(1, keepdims=True), rcond=None)
        recon = A @ coef
        resid = float(recon.var(axis=1).sum() / (X - X.mean(1, keepdims=True)).var(axis=1).sum())

    return {
        "blink_index": float(front_low / (front_all + 1e-30)),
        "emg_index": float(periph_emg / (periph_all + 1e-30)),
        "line_index": float(line),
        "log_alpha_sensorimotor": float(alpha),
        "log_beta_sensorimotor": float(beta),
        "mean_kurtosis": kurt,
        "residual_artifact_var": resid,
        "total_var": float(X.var(axis=1).sum()),
    }


def retained_metrics(raw: mne.io.BaseRaw, cleaned: mne.io.BaseRaw) -> dict:
    from mne.filter import filter_data

    fs = float(raw.info["sfreq"])
    X = filter_data(raw.get_data(picks="eeg"), fs, 4, 40, verbose=False)
    Y = filter_data(cleaned.get_data(picks="eeg"), fs, 4, 40, verbose=False)
    return {
        "var_retained": float(Y.var(axis=1).sum() / (X.var(axis=1).sum() + 1e-30)),
        "corr_with_raw": float(np.corrcoef(X.ravel(), Y.ravel())[0, 1]),
    }


# --------------------------------------------------------------------------
# decoding (3-class cue decoding, within-recording CV)
# --------------------------------------------------------------------------
def bandpower_features(raw: mne.io.BaseRaw, events: pd.DataFrame):
    from mne.filter import filter_data

    fs = float(raw.info["sfreq"])
    X = raw.get_data(picks="eeg")
    filtered = [filter_data(X, fs, lo, hi, verbose=False) for lo, hi in BANDS.values()]

    y, rows = [], []
    for _, ev in events.iterrows():
        t0 = ev["onset_s"] + DECODE_WINDOW[0]
        t1 = ev["onset_s"] + DECODE_WINDOW[1]
        s0, s1 = int(t0 * fs), int(t1 * fs)
        if s0 < 0 or s1 > X.shape[1]:
            continue
        feat = np.concatenate(
            [np.log(f[:, s0:s1].var(axis=1) + 1e-30) for f in filtered]
        )
        rows.append(feat)
        y.append(int(ev["label"]))
    return np.asarray(rows, dtype=float), np.asarray(y)


def decode_accuracy(raw: mne.io.BaseRaw, events: pd.DataFrame) -> float:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = bandpower_features(raw, events)
    if len(np.unique(y)) < 2 or len(y) < 10:
        return float("nan")
    clf = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    return float(np.mean(cross_val_score(clf, X, y, cv=cv, scoring="accuracy")))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def load_manual_labels() -> dict:
    """Load one reviewed label file per recording (labels/P01_ses-1.json, ...)."""
    out = {}
    for path in sorted(al.LABEL_DIR.glob("*_ses-*.json")):
        out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    if not out:
        raise FileNotFoundError(
            f"no reviewed label files in {al.LABEL_DIR}"
        )
    return out


def run_one(key: str, manual: dict) -> list[dict]:
    d = al.cache_dir(key)
    raw = mne.io.read_raw_fif(d / "raw_preproc.fif", preload=True, verbose=False)
    ica = mne.preprocessing.read_ica(d / "ica.fif", verbose=False)
    icl = json.loads((d / "iclabel.json").read_text(encoding="utf-8"))
    events = pd.read_csv(d / "events.csv")
    comps = ica.get_components()

    labels = {str(k): v for k, v in manual["labels"].items()}
    man_ex = manual_exclude(labels)
    auto090 = al.auto_exclude(icl, 0.90)
    auto080 = al.auto_exclude(icl, 0.80)
    auto070 = al.auto_exclude(icl, 0.70)
    auto090_other = al.auto_exclude(icl, 0.90, set(al.ARTIFACT_CLASSES) | {"other"})

    # matched-aggressiveness control: reject the same number of ICs as the
    # expert, ranked by ICLabel's strongest artifact-class probability
    scores = [
        (icl["probs"][i] if icl["labels"][i] in al.ARTIFACT_CLASSES else 0.0)
        for i in range(ica.n_components_)
    ]
    rank = np.argsort(scores)[::-1]
    auto_topk = [int(i) for i in rank[: len(man_ex)]]

    arms = {
        "none": [],
        "manual": man_ex,
        "auto090": auto090,
        "auto080": auto080,
        "auto070": auto070,
        "auto090_other": auto090_other,
        "auto_topk": auto_topk,
    }
    cleaned = {name: apply_exclusion(raw, ica, ex) for name, ex in arms.items()}

    rows = []
    for name, ex in arms.items():
        m = signal_metrics(cleaned[name], comps, man_ex)
        m.update(retained_metrics(raw, cleaned[name]))
        m.update(
            {
                "key": key,
                "arm": name,
                "n_excluded": len(ex),
                "excluded": ex,
                "decoding_accuracy": decode_accuracy(cleaned[name], events),
            }
        )
        rows.append(m)

    # component-level comparison (manual = reference)
    n = ica.n_components_
    man_set = set(man_ex)

    def _counts(auto_ex: list[int]) -> dict:
        a = set(auto_ex)
        tp = len(man_set & a)
        fp = len(a - man_set)
        fn = len(man_set - a)
        return {
            "n": len(a),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
        }

    c090 = _counts(auto090)
    cother = _counts(auto090_other)

    # threshold sweep for the automatic arm (manual labels are the reference)
    sweep = []
    for th in [0.60, 0.70, 0.80, 0.90, 0.95]:
        c = _counts(al.auto_exclude(icl, th))
        sweep.append({"key": key, "threshold": th, "manual_n": len(man_set), **c})
    rows.append(
        {
            "key": key,
            "arm": "component_eval",
            "n_components": n,
            "manual_n": len(man_set),
            "auto_n": c090["n"],
            "tp": c090["tp"],
            "fp": c090["fp"],
            "fn": c090["fn"],
            "precision": c090["precision"],
            "recall": c090["recall"],
            "f1": c090["f1"],
            "auto_other_n": cother["n"],
            "tp_other": cother["tp"],
            "fp_other": cother["fp"],
            "fn_other": cother["fn"],
            "precision_other": cother["precision"],
            "recall_other": cother["recall"],
            "f1_other": cother["f1"],
            "manual_labels": labels,
            "auto_labels": icl["labels"],
            "auto_probs": icl["probs"],
        }
    )
    return rows, sweep


def paired_tests(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = []
    for m in metrics:
        p = df.pivot(index="key", columns="arm", values=m)
        for a, b in [
            ("manual", "auto090"),
            ("manual", "auto_topk"),
            ("manual", "none"),
            ("auto090", "none"),
        ]:
            if a not in p or b not in p:
                continue
            x, y = p[a].dropna(), p[b].dropna()
            common = x.index.intersection(y.index)
            if len(common) < 3:
                continue
            x, y = x.loc[common], y.loc[common]
            diff = x - y
            try:
                stat, pval = wilcoxon(x, y)
            except ValueError:
                stat, pval = np.nan, np.nan
            out.append(
                {
                    "metric": m,
                    "comparison": f"{a} - {b}",
                    "mean_diff": float(diff.mean()),
                    "median_diff": float(diff.median()),
                    "p_wilcoxon": float(pval),
                    "n": len(common),
                }
            )
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=al.SUBJECTS)
    args = ap.parse_args()

    manual_all = load_manual_labels()
    rows, sweep_rows = [], []
    for subject in args.subjects:
        key = al.recording_key(subject)
        if key not in manual_all:
            print(f"skip {key}: no manual labels")
            continue
        print(f"=== {key} ===")
        r, s = run_one(key, manual_all[key])
        rows.extend(r)
        sweep_rows.extend(s)

    df = pd.DataFrame(rows)
    df.to_csv(al.RESULT_DIR / "per_recording_metrics.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(al.RESULT_DIR / "threshold_sweep.csv", index=False)

    metrics = [
        "blink_index", "emg_index", "line_index", "mean_kurtosis",
        "log_alpha_sensorimotor", "log_beta_sensorimotor",
        "var_retained", "corr_with_raw", "residual_artifact_var",
        "decoding_accuracy",
    ]
    comp = df[df["arm"] == "component_eval"]
    comp.to_csv(al.RESULT_DIR / "component_level.csv", index=False)
    tests = paired_tests(df[df["arm"] != "component_eval"], metrics)
    tests.to_csv(al.RESULT_DIR / "paired_tests.csv", index=False)

    print("\n=== per-arm means ===")
    print(
        df[df["arm"] != "component_eval"]
        .groupby("arm")[metrics]
        .mean()
        .round(4)
        .to_string()
    )
    print("\n=== component level (manual reference) ===")
    print(
        comp[["key", "n_components", "manual_n", "auto_n", "tp", "fp", "fn",
              "precision", "recall", "f1", "auto_other_n", "precision_other",
              "recall_other", "f1_other"]]
        .round(3)
        .to_string(index=False)
    )
    print("\n=== paired tests ===")
    print(tests.round(5).to_string(index=False))

    print("\n=== threshold sweep (mean over recordings) ===")
    print(
        pd.DataFrame(sweep_rows)
        .groupby("threshold")[["n", "tp", "fp", "fn", "precision", "recall", "f1"]]
        .mean()
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
