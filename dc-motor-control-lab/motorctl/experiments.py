"""Layer 4: experiments (tuning, comparison with CIs, sweeps, ablations, figures, validation)."""
from __future__ import annotations

import json
import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
from scipy import stats  # noqa: E402

from motorctl.controllers import PIDGains  # noqa: E402
from motorctl.estimators import KalmanFilter, steady_state_kalman_gain  # noqa: E402
from motorctl.factory import KINDS, build_controller  # noqa: E402
from motorctl.fixedpoint import QFormat  # noqa: E402
from motorctl.metrics import (control_effort, disturbance_metrics, iae, rms_delta,  # noqa: E402
                              saturation_fraction, step_metrics, window_std)
from motorctl.plant import MotorParams, continuous_matrices, discretize, expm, zoh  # noqa: E402
from motorctl.riccati import solve_dare  # noqa: E402
from motorctl.simulate import SimConfig, simulate  # noqa: E402

LABELS = {"pi": "PI (float)", "pi_fixed": "PI (Q17.14 int)", "lqi_oracle": "LQI (true state)",
          "lqg": "LQG (Kalman+LQI)"}


# ------------------------------------------------------------------ helpers
def mean_ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) < 2:
        return float(x.mean()), float("nan")
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(h)


def evaluate(res, cfg: SimConfig, p: MotorParams, band: float = 0.05) -> dict:
    sm = step_metrics(res.t, res.y, cfg.ref, cfg.t_step, cfg.t_dist, band)
    dm = disturbance_metrics(res.t, res.y, cfg.ref, cfg.t_dist, band * cfg.ref)
    return {
        "iae": iae(res.t, res.r - res.y),
        "rise_ms": sm["rise"] * 1e3,
        "overshoot_pct": sm["overshoot_pct"],
        "settle_ms": sm["settle"] * 1e3,
        "sse": sm["sse"],
        "dist_peak": dm["peak"],
        "dist_recover_ms": dm["recover"] * 1e3,
        "effort": control_effort(res.u, cfg.Ts),
        "du_rms": rms_delta(res.u),
        "ripple": window_std(res.t, res.y, cfg.t_dist - 0.15, cfg.t_dist),
        "sat_frac": saturation_fraction(res.u, p.vmax),
    }


def run_kind(kind, p, cfg, gains, **kw) -> dict:
    ctrl = build_controller(kind, p, cfg.Ts, pi_gains=gains, meas_sigma=cfg.meas_sigma,
                            enc_step=cfg.enc_step, delay=cfg.delay, **kw)
    res = simulate(p, ctrl, cfg)
    return evaluate(res, cfg, p), res


def aggregate(kind, p, cfg, gains, seeds, **kw) -> dict:
    rows = [run_kind(kind, p, replace(cfg, seed=s), gains, **kw)[0] for s in range(seeds)]
    return {k: mean_ci([r[k] for r in rows]) for k in rows[0]}


# ------------------------------------------------------------------ tuning
def tune_pi(p, cfg, kps=(0.02, 0.05, 0.1, 0.2, 0.4), kis=(1, 3, 10, 30, 60)):
    """Grid search minimising IAE * (1 + 0.02 * overshoot%) on a noisy seed-0 run."""
    best, grid = None, []
    for kp in kps:
        for ki in kis:
            g = PIDGains(kp, ki, umax=p.vmax)
            m, _ = run_kind("pi", p, cfg, g)
            score = m["iae"] * (1 + 0.02 * m["overshoot_pct"])
            grid.append((kp, ki, score))
            if best is None or score < best[0]:
                best = (score, kp, ki)
    return PIDGains(best[1], best[2], umax=p.vmax), grid


# ------------------------------------------------------------------ experiments
def compare(p, cfg, gains, seeds):
    return {k: aggregate(k, p, cfg, gains, seeds) for k in KINDS}


def sweep_noise(p, cfg, gains, seeds, sigmas=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0)):
    return {k: {s: aggregate(k, p, replace(cfg, meas_sigma=s), gains, seeds) for s in sigmas}
            for k in ("pi", "lqg")}


def sweep_rate(p, cfg, gains, seeds, Ts_list=(0.5e-3, 1e-3, 2e-3, 5e-3, 10e-3)):
    return {k: {ts: aggregate(k, p, replace(cfg, Ts=ts), gains, seeds) for ts in Ts_list}
            for k in ("pi", "lqg")}


def sweep_fixed(p, cfg, gains, seeds, fracs=(2, 4, 6, 8, 10, 12, 14, 16, 20), guards=(0, 8)):
    base = aggregate("pi", p, cfg, gains, seeds)["iae"][0]
    out = {"float_iae": base}
    for g in guards:
        out[g] = {}
        for f in fracs:
            a = aggregate("pi_fixed", p, cfg, gains, seeds, frac_bits=f, guard_bits=g)
            ctrl = build_controller("pi_fixed", p, cfg.Ts, pi_gains=gains, frac_bits=f, guard_bits=g)
            out[g][f] = {"iae": a["iae"], "rel_iae": a["iae"][0] / base,
                         "eff_ki_ts": ctrl.effective_ki_ts,
                         "state_bits": ctrl.qa.total_bits}
    return out


def ablation_antiwindup(p, cfg, gains):
    p2 = replace(p, vmax=6.5)
    cfg2 = replace(cfg, ref=110.0, tau_load=0.0, meas_sigma=0.0)
    out, traces = {}, {}
    for mode in ("none", "clamp", "backcalc"):
        m, res = run_kind("pi", p2, cfg2, replace(gains, umax=p2.vmax), antiwindup=mode)
        out[mode] = {k: m[k] for k in ("iae", "overshoot_pct", "settle_ms", "sat_frac")}
        traces[mode] = res
    return out, traces, cfg2


def ablation_lqg(p, cfg, gains, seeds):
    variants = {
        "LQG (full)": ("lqg", {}),
        "LQG, no torque feed-forward": ("lqg", {"use_dff": False}),
        "LQI, true state (no estimator)": ("lqi_oracle", {}),
    }
    return {name: aggregate(kind, p, cfg, gains, seeds, **kw) for name, (kind, kw) in variants.items()}


# ------------------------------------------------------------------ validation vs references
def validation_table() -> dict:
    p = MotorParams()
    A, B, C = continuous_matrices(p)
    rng = np.random.default_rng(0)
    e_expm = max(float(np.max(np.abs(expm(M) - sla.expm(M)))) / float(np.max(np.abs(sla.expm(M))))
                 for M in [rng.normal(size=(4, 4)) * 3 for _ in range(5)] + [A * 1e-3])
    import scipy.signal as sig
    e_zoh = 0.0
    for Ts in (2e-4, 1e-3, 5e-3):
        Ad, Bd = zoh(A, B, Ts)
        Ad2, Bd2, *_ = sig.cont2discrete((A, B, C, np.zeros((1, 2))), Ts, method="zoh")
        e_zoh = max(e_zoh, float(np.max(np.abs(Ad - Ad2))), float(np.max(np.abs(Bd - Bd2))))
    e_dare = 0.0
    for _ in range(6):
        Am, Bm = rng.normal(size=(3, 3)) * 0.6, rng.normal(size=(3, 1))
        ours, ref = solve_dare(Am, Bm, np.eye(3), [[0.5]]), sla.solve_discrete_are(Am, Bm, np.eye(3), [[0.5]])
        e_dare = max(e_dare, float(np.max(np.abs(ours - ref)) / np.max(np.abs(ref))))
    golden = abs(solve_dare([[1.0]], [[1.0]], [[1.0]], [[1.0]])[0, 0] - (1 + 5**0.5) / 2)
    # Kalman: dual-DARE gain vs scipy dual, and recursive KF covariance -> steady state
    Ad, Bd, Cd = discretize(p, 1e-3)
    from motorctl.estimators import augment_disturbance
    Aa, Bv, Ca = augment_disturbance(Ad, Bd)
    Q, R = np.diag(np.square((0.05, 0.5, 2e-4))), np.array([[0.8]])
    L, Pss = steady_state_kalman_gain(Aa, Ca, Q, R)
    Pref = sla.solve_discrete_are(Aa.T, Ca.T, Q, R)
    kf = KalmanFilter(Aa, Bv, Ca, Q, R, P0=np.eye(3) * 10)
    for _ in range(4000):
        kf.update(0.0)
        kf.predict(0.0)
    return {"expm_vs_scipy": e_expm, "zoh_vs_scipy": e_zoh, "dare_vs_scipy": e_dare,
            "golden_ratio_dare": float(golden),
            "kalman_gain_vs_scipy": float(np.max(np.abs(Pss - Pref)) / np.max(np.abs(Pref))),
            "kf_recursion_vs_dare": float(np.max(np.abs(kf.P - Pref)) / np.max(np.abs(Pref)))}


# ------------------------------------------------------------------ figures
def _savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_step(p, cfg, gains, path):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for k in KINDS:
        _, res = run_kind(k, p, cfg, gains)
        ax[0].plot(res.t * 1e3, res.y, label=LABELS[k], lw=1.3)
        ax[1].plot(res.t * 1e3, res.u, lw=1.0)
    ax[0].plot(res.t * 1e3, res.r, "k--", lw=0.8, label="reference")
    ax[0].axvline(cfg.t_dist * 1e3, color="gray", ls=":")
    ax[0].set_ylabel("speed [rad/s]")
    ax[0].legend(fontsize=8)
    ax[1].set_ylabel("command [V]")
    ax[1].set_xlabel("time [ms]  (load torque applied at dotted line)")
    ax[0].set_title("Speed step + load-torque disturbance (seed 0)")
    _savefig(fig, path)


def fig_noise(res, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for k in res:
        xs = sorted(res[k])
        for a, met in zip(ax, ("iae", "du_rms")):
            a.errorbar(xs, [res[k][s][met][0] for s in xs], [res[k][s][met][1] for s in xs],
                       marker="o", capsize=2, label=LABELS[k])
    ax[0].set_ylabel("IAE [rad]")
    ax[1].set_ylabel("RMS command change [V/sample]")
    for a in ax:
        a.set_xlabel("speed-sensor noise sigma [rad/s]")
        a.legend(fontsize=8)
    _savefig(fig, path)


def fig_rate(res, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for k in res:
        xs = sorted(res[k])
        ax.errorbar([1e3 * x for x in xs], [res[k][t]["iae"][0] for t in xs],
                    [res[k][t]["iae"][1] for t in xs], marker="o", capsize=2, label=LABELS[k])
    ax.set_xscale("log")
    ax.set_xlabel("sample period [ms] (1-sample compute delay)")
    ax.set_ylabel("IAE [rad]")
    ax.legend(fontsize=8)
    _savefig(fig, path)


def fig_fixed(res, path):
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for g in (k for k in res if k != "float_iae"):
        xs = sorted(res[g])
        ax.plot(xs, [res[g][f]["rel_iae"] for f in xs], marker="o", label=f"guard bits = {g}")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("fractional bits")
    ax.set_ylabel("IAE relative to float PI")
    ax.legend()
    _savefig(fig, path)


def fig_aw(traces, cfg, path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for m, r in traces.items():
        ax.plot(r.t * 1e3, r.y, label=m)
    ax.plot(r.t * 1e3, r.r, "k--", lw=0.8)
    ax.set_xlim(0, 150)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("speed [rad/s]")
    ax.set_title("Anti-windup ablation (6.5 V supply limit)")
    ax.legend()
    _savefig(fig, path)


# ------------------------------------------------------------------ report
def _cell(v, nd=3):
    m, h = v
    if m != m:
        return "n/a"
    return f"{m:.{nd}f} ± {h:.{nd}f}" if h == h else f"{m:.{nd}f}"


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def run_all(out_dir="results", seeds=20, quick=False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    p, cfg = MotorParams(), SimConfig()
    sw_seeds = 3 if quick else 8
    gains, grid = tune_pi(p, cfg, **({"kps": (0.05, 0.1, 0.2), "kis": (3, 10, 30)} if quick else {}))
    summary = {"pi_gains": {"kp": gains.kp, "ki": gains.ki}, "seeds": seeds, "sweep_seeds": sw_seeds}
    summary["validation"] = validation_table()
    cmp_ = compare(p, cfg, gains, seeds)
    summary["compare"] = cmp_
    nz, rt = sweep_noise(p, cfg, gains, sw_seeds), sweep_rate(p, cfg, gains, sw_seeds)
    fx = sweep_fixed(p, cfg, gains, max(2, sw_seeds // 2))
    aw, traces, cfg_aw = ablation_antiwindup(p, cfg, gains)
    ab = ablation_lqg(p, cfg, gains, sw_seeds)
    summary.update(noise={k: {str(s): v for s, v in d.items()} for k, d in nz.items()},
                   rate={k: {str(s): v for s, v in d.items()} for k, d in rt.items()},
                   fixed={str(g): ({str(f): v for f, v in d.items()} if g != "float_iae" else d)
                          for g, d in fx.items()},
                   antiwindup=aw, ablation_lqg=ab)
    fig_step(p, cfg, gains, os.path.join(out_dir, "step_response.png"))
    fig_noise(nz, os.path.join(out_dir, "noise_sweep.png"))
    fig_rate(rt, os.path.join(out_dir, "rate_sweep.png"))
    fig_fixed(fx, os.path.join(out_dir, "fixed_point_sweep.png"))
    fig_aw(traces, cfg_aw, os.path.join(out_dir, "antiwindup.png"))
    mets = [("iae", "IAE [rad]"), ("overshoot_pct", "Overshoot [%]"), ("settle_ms", "Settle 5% [ms]"),
            ("dist_peak", "Load dip [rad/s]"), ("dist_recover_ms", "Recover [ms]"),
            ("effort", "∫u² dt"), ("du_rms", "RMS Δu [V]")]
    md = [f"PI gains from grid search: kp={gains.kp}, ki={gains.ki}; {seeds} seeds, mean ± 95% CI\n",
          _table(["Controller"] + [m[1] for m in mets],
                 [[LABELS[k]] + [_cell(cmp_[k][m[0]]) for m in mets] for k in KINDS]), ""]
    md += ["Noise sweep (IAE / RMS Δu)\n",
           _table(["sigma [rad/s]", "PI IAE", "LQG IAE", "PI RMS Δu", "LQG RMS Δu"],
                  [[s, _cell(nz["pi"][s]["iae"]), _cell(nz["lqg"][s]["iae"]),
                    _cell(nz["pi"][s]["du_rms"]), _cell(nz["lqg"][s]["du_rms"])] for s in sorted(nz["pi"])]), ""]
    md += ["Sample-rate sweep (IAE, peak load dip)\n",
           _table(["Ts [ms]", "PI IAE", "LQG IAE", "PI dip", "LQG dip"],
                  [[1e3 * t, _cell(rt["pi"][t]["iae"]), _cell(rt["lqg"][t]["iae"]),
                    _cell(rt["pi"][t]["dist_peak"], 2), _cell(rt["lqg"][t]["dist_peak"], 2)] for t in sorted(rt["pi"])]), ""]
    fr = sorted(fx[0])
    md += [f"Fixed-point PI (float IAE = {fx['float_iae']:.3f})\n",
           _table(["frac bits", "ki*Ts realised (g=0)", "rel IAE g=0", "rel IAE g=8"],
                  [[f, f"{fx[0][f]['eff_ki_ts']:.6f}", f"{fx[0][f]['rel_iae']:.3f}", f"{fx[8][f]['rel_iae']:.3f}"] for f in fr]), ""]
    md += ["Anti-windup ablation (6.5 V limit, 110 rad/s step)\n",
           _table(["mode", "IAE", "overshoot %", "settle ms"],
                  [[m, f"{v['iae']:.3f}", f"{v['overshoot_pct']:.1f}", f"{v['settle_ms']:.1f}"] for m, v in aw.items()]), ""]
    md += ["LQG ablation\n",
           _table(["variant", "IAE", "Load dip", "Recover ms"],
                  [[n, _cell(v["iae"]), _cell(v["dist_peak"], 2), _cell(v["dist_recover_ms"], 1)] for n, v in ab.items()]), ""]
    md += ["Validation\n", _table(["check", "max error"], [[k, f"{v:.2e}"] for k, v in summary["validation"].items()])]
    with open(os.path.join(out_dir, "results.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    return summary
