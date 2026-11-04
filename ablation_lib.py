"""Shared pipeline for the manual-vs-automatic ICA ablation.

Both arms use exactly the same input, automatic bad-channel detection,
interpolation, filtering, re-referencing and ICA decomposition.  They differ
only in which ICA components are rejected:

* manual      : expert labels (labels/manual_labels.json)
* automatic   : mne-icalabel / ICLabel probabilities (pipeline default 0.90)

The data are read from the BIDS folder; nothing in the BIDS folder is
modified.  Run scripts from this directory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis

ROOT = Path(r"D:\Downloads\BCI preprocessing")
BIDS_ROOT = ROOT / "nm000290"
OUT_ROOT = ROOT / "experiments" / "ica_ablation"
CACHE = OUT_ROOT / "cache"
FIG = OUT_ROOT / "figures"
LABEL_DIR = OUT_ROOT / "labels"
RESULT_DIR = OUT_ROOT / "results"
for _d in (CACHE, FIG, LABEL_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SUBJECTS = [f"P{i:02d}" for i in range(1, 11)]
SESSIONS = {s: "1" for s in SUBJECTS}

CH_NAMES = [
    "C2", "CP2", "P2", "Pz", "CPz", "Cz", "C1", "CP1", "P1", "P3",
    "PO7", "P5", "TP7", "CP5", "CP3", "C3", "PO8", "P6", "P4", "TP8",
    "CP6", "CP4", "C6", "C4", "FC6", "FC4", "FC2", "F6", "F4", "F2",
    "AF4", "AF3", "F5", "F3", "F1", "Fz", "FC5", "FC3", "FC1", "FCz", "C5",
]

FRONTAL = ["AF3", "AF4", "F1", "F2", "F3", "F4", "F5", "F6"]
PERIPHERAL = ["TP7", "TP8", "FC5", "FC6", "C5", "C6", "F5", "F6"]
SENSORIMOTOR = ["C3", "C4", "Cz", "CP3", "CP4", "CPz", "FC3", "FC4", "FCz"]

ARTIFACT_CLASSES = {
    "eye blink", "muscle artifact", "heart beat", "line noise", "channel noise",
}

# ---- fixed analysis parameters -------------------------------------------
SFREQ = 500.0
TRIM_START_S = 4.0               # amplifier/subject settling transient at run start
TRIM_END_S = 2.0                 # BAD_ACQ_SKIP padding at the EDF tail
PRE_L_FREQ, PRE_H_FREQ = 1.0, 100.0
NOTCH = [50.0, 100.0]
ICA_DECIM = 3
ICA_SEED = 1
AUTO_THRESHOLD = 0.90            # preprocessing1 default
ALT_THRESHOLDS = [0.80, 0.70]


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def recording_key(subject: str, session: str | None = None) -> str:
    session = session or SESSIONS[subject]
    return f"{subject}_ses-{session}"


def recording_path(subject: str, session: str | None = None) -> Path:
    session = session or SESSIONS[subject]
    files = sorted(
        BIDS_ROOT.glob(
            f"sub-{subject}/ses-{session}/eeg/"
            f"sub-{subject}_ses-{session}_task-OnlineRawGraz_run-*_eeg.edf"
        )
    )
    if not files:
        raise FileNotFoundError(f"no OnlineRawGraz EDF for {subject} ses-{session}")

    def run_no(p: Path) -> int:
        m = re.search(r"run-(\d+)", p.name)
        return int(m.group(1)) if m else 10**6

    return min(files, key=run_no)


def events_path(subject: str, session: str | None = None) -> Path:
    session = session or SESSIONS[subject]
    return recording_path(subject, session).with_name(
        recording_path(subject, session).name.replace("_eeg.edf", "_events.tsv")
    )


def load_raw(subject: str, session: str | None = None) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(recording_path(subject, session), preload=True, verbose=False)
    raw.crop(tmin=TRIM_START_S, tmax=raw.times[-1] - TRIM_END_S)
    raw.set_montage("standard_1005", on_missing="ignore")
    raw.info["bads"] = []
    return raw


def load_events(subject: str, session: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(events_path(subject, session), sep="\t")
    df = df[df["trial_type"].isin(["mi", "sub", "word"])].copy()
    df["label"] = df["trial_type"].map({"mi": 0, "sub": 1, "word": 2})
    # events are stored relative to the EDF start; the cached raw starts at
    # TRIM_START_S, so shift them into the cropped timeline
    df["onset_s"] = df["sample"].astype(float) / SFREQ - TRIM_START_S
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# automatic bad-channel detection
# --------------------------------------------------------------------------
def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 0.6745 * (x - med) / (mad + 1e-12)


def bad_channel_features(raw: mne.io.BaseRaw) -> pd.DataFrame:
    """PREP/EEGLAB-style features computed on a filtered copy."""
    r = (
        raw.copy()
        .filter(PRE_L_FREQ, PRE_H_FREQ, verbose=False)
        .notch_filter(NOTCH, verbose=False)
        .set_eeg_reference("average", verbose=False)
    )
    picks = mne.pick_types(r.info, eeg=True)
    X = r.get_data(picks=picks) * 1e6  # uV
    names = [r.ch_names[i] for i in picks]
    fs = float(r.info["sfreq"])

    pos = np.array([r.info["chs"][i]["loc"][:3] for i in picks])
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    corr = np.corrcoef(X)
    np.fill_diagonal(corr, 0.0)
    k = min(6, len(names) - 1)
    nb_corr = np.array(
        [np.sort(corr[i, np.argsort(dist[i])[1:k + 1]]).mean() for i in range(len(names))]
    )

    freqs, psd = welch(X, fs=fs, nperseg=int(2 * fs))
    hf = psd[:, (freqs >= 30) & (freqs <= 45)].sum(1)
    lf = psd[:, (freqs >= 1) & (freqs <= 4)].sum(1)
    var = X.var(1)
    ptp = np.ptp(X, axis=1)

    df = pd.DataFrame(
        {
            "channel": names,
            "log_hf": np.log(hf + 1e-20),
            "log_lf": np.log(lf + 1e-20),
            "log_var": np.log(var + 1e-20),
            "ptp_uv": ptp,
            "neighbor_corr": nb_corr,
            "hf_z": _robust_z(np.log(hf + 1e-20)),
            "lf_z": _robust_z(np.log(lf + 1e-20)),
            "var_z": _robust_z(np.log(var + 1e-20)),
            "nbc_z": _robust_z(nb_corr),
        }
    )
    return df


def detect_bad_channels(
    raw: mne.io.BaseRaw,
    hf_z: float = 5.0,
    lf_z: float = 5.0,
    var_z: float = 5.0,
    nbc_z: float = -5.0,
    nbc_abs: float = 0.35,
    ptp_uv: float = 2.0,
    max_frac: float = 0.15,
) -> tuple[list[str], pd.DataFrame]:
    df = bad_channel_features(raw)
    score = np.zeros(len(df))
    reasons: dict[str, list[str]] = {c: [] for c in df["channel"]}

    for i, row in df.iterrows():
        c = row["channel"]
        if row["ptp_uv"] < ptp_uv:
            reasons[c].append("flat")
            score[i] += 100
        if row["hf_z"] > hf_z:
            reasons[c].append("high_frequency_noise")
            score[i] += row["hf_z"]
        if row["lf_z"] > lf_z:
            reasons[c].append("low_frequency_drift")
            score[i] += row["lf_z"]
        if row["var_z"] > var_z:
            reasons[c].append("high_amplitude")
            score[i] += row["var_z"]
        if row["nbc_z"] < nbc_z and row["neighbor_corr"] < nbc_abs:
            reasons[c].append("low_neighbor_corr")
            score[i] += abs(row["nbc_z"])

    order = np.argsort(score)[::-1]
    cap = max(1, int(round(max_frac * len(df))))
    bads = [df["channel"].iloc[i] for i in order if score[i] > 0][:cap]
    df["is_bad"] = df["channel"].isin(bads)
    df["reason"] = df["channel"].map(lambda c: ";".join(reasons[c]))
    return bads, df


# --------------------------------------------------------------------------
# preprocessing + ICA
# --------------------------------------------------------------------------
def preprocess(raw: mne.io.BaseRaw, bads: Iterable[str]) -> mne.io.BaseRaw:
    r = raw.copy()
    bads = list(bads)
    if bads:
        r.info["bads"] = bads
        r.interpolate_bads(
            reset_bads=True,
            method={"eeg": "spline"},
            mode="accurate",
            verbose=False,
        )
    r.filter(PRE_L_FREQ, PRE_H_FREQ, method="fir", phase="zero",
             fir_window="hamming", verbose=False)
    r.notch_filter(NOTCH, method="fir", phase="zero", verbose=False)
    r.set_eeg_reference("average", verbose=False)
    return r


def fit_ica(raw: mne.io.BaseRaw) -> mne.preprocessing.ICA:
    ica = mne.preprocessing.ICA(
        n_components=None,
        random_state=ICA_SEED,
        method="picard",
        fit_params={"extended": True, "ortho": False},
        max_iter="auto",
        verbose=False,
    )
    ica.fit(raw, decim=ICA_DECIM, verbose=False)
    return ica


def iclabel(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA) -> dict:
    from mne_icalabel import label_components

    out = label_components(raw, ica, method="iclabel")
    return {
        "labels": [str(x) for x in out["labels"]],
        "probs": [float(x) for x in out["y_pred_proba"]],
    }


def auto_exclude(iclabel_out: dict, threshold: float = AUTO_THRESHOLD,
                 classes: set[str] | None = None) -> list[int]:
    classes = set(ARTIFACT_CLASSES if classes is None else classes)
    return [
        i for i, (lab, p) in enumerate(zip(iclabel_out["labels"], iclabel_out["probs"]))
        if lab in classes and p >= threshold
    ]


# --------------------------------------------------------------------------
# per-component diagnostics used for the manual labelling
# --------------------------------------------------------------------------
def _bandpass(data: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    from mne.filter import filter_data

    return filter_data(data, sfreq=fs, l_freq=lo, h_freq=hi, verbose=False)


def component_features(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA) -> pd.DataFrame:
    fs = float(raw.info["sfreq"])
    sources = ica.get_sources(raw).get_data()
    comps = ica.get_components()  # channel-space spatial patterns (n_channels, n_comp)
    names = raw.ch_names
    # exact per-component variance ratio (MNE), estimated on a 60-s window
    ev_raw = raw.copy().crop(0.0, min(60.0, raw.times[-1]))
    ev_ratio = np.array(
        [
            ica.get_explained_variance_ratio(ev_raw, components=[i])["eeg"]
            for i in range(ica.n_components_)
        ]
    )

    # EOG proxy: frontal channels, 0.5-7 Hz
    fi = [names.index(c) for c in FRONTAL if c in names]
    eog = _bandpass(raw.get_data(picks=fi), fs, 0.5, 7.0)
    # EMG proxy: peripheral channels, 20-45 Hz envelope
    pi = [names.index(c) for c in PERIPHERAL if c in names]
    emg = _bandpass(raw.get_data(picks=pi), fs, 20.0, 45.0)
    emg_env = np.abs(emg)
    win = int(0.05 * fs)
    emg_env = np.apply_along_axis(
        lambda v: np.convolve(v, np.ones(win) / win, mode="same"), 1, emg_env
    )

    rows = []
    for i in range(ica.n_components_):
        s = sources[i]
        s_eog = _bandpass(s[None, :], fs, 0.5, 7.0)[0]
        s_emg = _bandpass(s[None, :], fs, 20.0, 45.0)[0]
        s_env = np.convolve(np.abs(s_emg), np.ones(win) / win, mode="same")

        eog_r = max(abs(np.corrcoef(s_eog, e)[0, 1]) for e in eog)
        emg_r = max(abs(np.corrcoef(s_env, e)[0, 1]) for e in emg_env)

        freqs, psd = welch(s, fs=fs, nperseg=int(2 * fs))
        line = psd[(freqs >= 49) & (freqs <= 51)].mean() / (
            psd[(freqs >= 47) & (freqs <= 53)].mean() + 1e-30
        )
        sel = (freqs >= 2) & (freqs <= 40)
        slope = np.polyfit(np.log(freqs[sel]), np.log(psd[sel] + 1e-30), 1)[0]

        # cardiac rhythm: autocorrelation of 0.7-3 Hz signal, lag 0.6-1.2 s
        sc = _bandpass(s[None, :], fs, 0.7, 3.0)[0]
        sc = sc - sc.mean()
        ac = np.correlate(sc, sc, mode="full")[len(sc) - 1:]
        ac = ac / (ac[0] + 1e-30)
        lo, hi = int(0.6 * fs), int(1.2 * fs)
        cardiac_ac = float(ac[lo:hi].max()) if hi < len(ac) else np.nan

        topo = comps[:, i]
        topo_abs = np.abs(topo)
        topo_max = float(topo_abs.max())
        pos = np.array([raw.info["chs"][j]["loc"][:3] for j in range(len(names))])
        radius = np.linalg.norm(pos - pos.mean(axis=0), axis=1)
        edge = radius >= 0.75 * radius.max()
        row = {
            "ic": i,
            "var_explained": float(ev_ratio[i]),
            "kurtosis": float(kurtosis(s, fisher=False)),
            "spectral_slope": float(slope),
            "line_ratio": float(line),
            "eog_corr": float(eog_r),
            "emg_corr": float(emg_r),
            "cardiac_ac": cardiac_ac,
            "topo_frontal": float(topo[fi].mean()),
            "topo_peripheral": float(topo[pi].mean()),
            "topo_max": topo_max,
            "topo_focality": float(topo_max / (np.sqrt(np.mean(topo ** 2)) + 1e-30)),
            "topo_edge": float(topo_abs[edge].max() / (topo_max + 1e-30)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_component_diagnostics(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    features: pd.DataFrame,
    out_stem: Path,
    per_page: int = 10,
    duration_s: float = 10.0,
) -> list[Path]:
    fs = float(raw.info["sfreq"])
    sources = ica.get_sources(raw).get_data()
    comps = ica.get_components()
    info = mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True))
    n = ica.n_components_
    freqs, psd = welch(sources, fs=fs, nperseg=int(2 * fs))
    t = np.arange(int(duration_s * fs)) / fs

    paths: list[Path] = []
    for page, start in enumerate(range(0, n, per_page), start=1):
        idxs = list(range(start, min(start + per_page, n)))
        fig, axes = plt.subplots(
            len(idxs), 3,
            figsize=(17, 2.0 * len(idxs) + 0.8),
            gridspec_kw={"width_ratios": [1.3, 2.6, 2.2]},
            squeeze=False,
        )
        for row, i in enumerate(idxs):
            f = features.iloc[i]
            ax_t, ax_w, ax_p = axes[row]
            mne.viz.plot_topomap(
                comps[:, i], info, axes=ax_t, show=False, contours=4, cmap="RdBu_r"
            )
            ax_t.set_title(
                f"IC{i:03d}  var={f.var_explained*100:.1f}%  kurt={f['kurtosis']:.1f}",
                fontsize=8,
            )
            ax_w.plot(t, sources[i, :len(t)], lw=0.7, color="tab:blue")
            ax_w.set_xlim(0, duration_s)
            ax_w.set_ylabel("a.u.", fontsize=7)
            ax_w.tick_params(labelsize=6)
            ax_w.grid(alpha=0.3)
            ax_w.set_title(
                f"EOGr={f.eog_corr:.2f}  EMGr={f.emg_corr:.2f}  "
                f"line={f.line_ratio:.2f}  slope={f.spectral_slope:.2f}  "
                f"card={f.cardiac_ac:.2f}",
                fontsize=7,
            )
            ax_p.semilogy(freqs, psd[i], lw=0.7, color="tab:red")
            ax_p.set_xlim(0, 60)
            ax_p.tick_params(labelsize=6)
            ax_p.grid(alpha=0.3)
            ax_p.set_title("PSD", fontsize=7)
        fig.suptitle(out_stem.name, fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        p = out_stem.with_name(f"{out_stem.name}_ICs_{page}.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------
# cache helpers
# --------------------------------------------------------------------------
def cache_dir(key: str) -> Path:
    d = CACHE / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_stage1(key: str, raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA,
                bads: list[str], bad_df: pd.DataFrame, icl: dict,
                feats: pd.DataFrame) -> None:
    d = cache_dir(key)
    raw.save(d / "raw_preproc.fif", overwrite=True, verbose=False)
    ica.save(d / "ica.fif", overwrite=True, verbose=False)
    (d / "bads.json").write_text(
        json.dumps({"bads": bads, "features": bad_df.to_dict(orient="records")},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (d / "iclabel.json").write_text(json.dumps(icl, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    feats.to_csv(d / "ic_features.csv", index=False)
    load_events(key.split("_")[0], key.split("_ses-")[1]).to_csv(
        d / "events.csv", index=False
    )
