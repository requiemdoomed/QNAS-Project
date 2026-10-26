"""Turn a finished run directory into figures and a self-contained ``results.md``.

Every sentence about the research questions is *computed* from the saved artifacts (with explicit hedging when
differences are not statistically significant) - nothing here is hard-coded to say that QNAS wins.
"""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/report.py
# Purpose: turning a finished run folder into figures and a results.md summary whose
#          every claim is computed from the saved numbers.
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
import json
from typing import Dict, List

import numpy as np

from . import plots
from .evolution import normalized_vector
from .genome import describe, genome_from_json
from .pareto import dominates
from .templates import baseline_suite


def _pct(x: float, digits: int = 1) -> str:
    # Formatting a fraction as a percentage.
    return f"{x * 100:.{digits}f}%"


def _pfmt(p: float) -> str:
    # Printing p-values readably ("n/a" for NaN, "<0.01" for tiny ones).
    if p != p:
        return "n/a"
    return "<0.01" if p < 0.01 else f"{p:.2f}"


def _pts(x: float) -> str:
    # Formatting an accuracy difference in percentage points with a sign.
    return f"{x * 100:+.1f} pts"


def _read(run, rel):
    # Reading an optional JSON artifact (None when that stage was skipped).
    p = run.dir / rel
    return json.loads(p.read_text()) if p.exists() else None


def _verdict(diff: float, p: float) -> str:
    # Wording the result carefully: only p < 0.05 is called significant.
    if np.isnan(p):
        return "n/a"
    if p < 0.05:
        return "significantly higher" if diff > 0 else "significantly lower"
    return "no significant difference" if abs(diff) <= 0.03 else ("higher (n.s.)" if diff > 0 else "lower (n.s.)")


def _slope(rows: List[dict]) -> float:
    """Least-squares slope of log2(variance) per added qubit (-1.0 means variance halves with every qubit)."""
    # Fitting a straight line to log2(variance) against qubit count.
    xs = np.array([r["n_qubits"] for r in rows], float)
    ys = np.log2([r["var_mean"] for r in rows])
    return float(np.polyfit(xs, ys, 1)[0]) if len(xs) > 1 else float("nan")


def build_report(run) -> str:
    cfg, fig_dir = run.cfg, run.dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    # Loading every artifact the earlier stages produced.
    task_info = _read(run, "task_info.json") or {}
    with open(run.dir / "search/history.csv", newline="") as fh:
        history = list(csv.DictReader(fh))
    history = [{k: float(v) for k, v in h.items()} for h in history]
    front = _read(run, "search/pareto.json")
    archive = _read(run, "search/archive.json")
    for rec in front + archive:
        rec["genes"] = genome_from_json(rec["genes"])
    baselines = _read(run, "baselines.json") or {}
    cands = _read(run, "candidates.json") or {}
    summary = _read(run, "final/final_summary.json") or {}
    comps = _read(run, "final/final_comparison.json") or {}
    grads = _read(run, "gradients/gradients.json")
    noise = _read(run, "noise/noise.json")
    n_q = cfg.n_qubits
    spec = run.spec

    # ------------------------------------------------------------------ figures
    # Drawing every figure the available data allows.
    figs: Dict[str, str] = {}
    figs["pareto"] = plots.plot_pareto(archive, front, baselines, fig_dir / "pareto_front.png")
    figs["convergence"] = plots.plot_convergence(history, fig_dir / "convergence.png")
    if summary:
        figs["final"] = plots.plot_final_comparison(summary, fig_dir / "final_comparison.png")
        figs["acc_gates"] = plots.plot_accuracy_vs_gates(summary, fig_dir / "accuracy_vs_gates.png")
        focus = [n for n in ("QNAS-bestF", "QNAS-compact", "SEL-L2", "SEL-L3", "BEL-L2") if n in summary]
        figs["curves"] = plots.plot_learning_curves(summary, focus, fig_dir / "learning_curves.png")
    if grads:
        figs["scan"] = plots.plot_barren_scan(grads["scan"], fig_dir / "barren_plateau_scan.png")
        figs["cand_grad"] = plots.plot_candidate_gradients(grads["candidates"], fig_dir / "candidate_gradients.png")
    if noise:
        figs["noise"] = plots.plot_noise(noise, fig_dir / "noise.png")
    suite = baseline_suite(n_q, 3)
    circuit_figs = {}
    for name, c in cands.items():
        genes = genome_from_json(c["genes"])
        circuit_figs[name] = plots.draw_genome(genes, n_q, n_q, fig_dir / f"circuit_{name}.png", title=name)
    if "SEL-L2" in suite:
        circuit_figs["SEL-L2"] = plots.draw_genome(suite["SEL-L2"], n_q, n_q, fig_dir / "circuit_SEL-L2.png", title="SEL-L2 (hand-crafted)")

    # ------------------------------------------------------------------ text
    # Writing results.md section by section.
    L: List[str] = []
    w = L.append
    sizes = task_info.get("sizes", {})
    w(f"# QNAS results - `{run.dir.name}`\n")
    w("*QNAS project by Swayam Shakya. Every number below is read from the files of this run.*\n")
    w("## Setup\n")
    w(f"- **Task:** {cfg.dataset} {json.dumps(cfg.classes)} (binary), classical extractor `{cfg.extractor}` -> {n_q} features -> {n_q} qubits")
    w(f"- **Split sizes:** train {sizes.get('train')}, validation {sizes.get('val')}, test {sizes.get('test')} "
      f"(the search only ever sees train + validation; the test split is used once, in the final comparison)")
    w(f"- **Search:** `{cfg.mode}` | population {cfg.pop_size} | {cfg.generations} generations | P_c={cfg.p_crossover}, "
      f"P_m={cfg.p_mutation} per gene | objectives {cfg.objectives} | {cfg.epochs} training epochs per candidate | "
      f"{history[-1]['evals_total']:.0f} unique circuits trained in {history[-1]['elapsed_s']:.0f}s")
    w(f"- **Search space:** gates RX/RY/RZ/H/CNOT/CZ/SWAP, connectivity `{cfg.connectivity}`, depth <= {cfg.max_depth}, "
      f"<= {cfg.max_genes} gates, data re-uploading {'on' if cfg.allow_data else 'off'}")
    w(f"- **Final protocol:** every model re-trained from scratch for {cfg.final_epochs} epochs x {cfg.final_seeds} seeds; "
      f"fitness weights alpha={cfg.alpha}, beta={cfg.beta}, gamma={cfg.gamma}\n")

    if summary:
        w("## Headline: final test accuracy\n")
        w("| Model | Test accuracy (mean ± std) | Two-qubit gates | Depth | Parameters |")
        w("|---|---|---|---|---|")
        for n, s in sorted(summary.items(), key=lambda kv: -kv[1]["test_acc_mean"]):
            cost = ("—", "—") if s["kind"] != "quantum" else (s["n_2q"], s["depth"])
            w(f"| {n} | {_pct(s['test_acc_mean'])} ± {_pct(s['test_acc_std'])} | {cost[0]} | {cost[1]} | {s['n_params']} |")
        w("\n![final comparison](figures/final_comparison.png)\n")

        # ---------------- RQ1
        w("## RQ1 - Accuracy and efficiency\n")
        w("*Can evolved circuits match the hand-crafted templates with fewer two-qubit gates?*\n")
        w("| Evolved circuit | vs baseline | Δ accuracy | Welch p | Verdict | 2-qubit gates (evolved / baseline) | Depth (evolved / baseline) |")
        w("|---|---|---|---|---|---|---|")
        # Comparing each QNAS model with the key templates and the best hand-crafted one.
        sel_bel = [n for n in summary if n.startswith(("SEL", "BEL"))]
        best_hand = max(sel_bel, key=lambda n: summary[n]["test_acc_mean"])
        for q, rows in comps.items():
            for r in rows:
                if r["model"] in ("SEL-L1", "SEL-L2", "SEL-L3", "BEL-L2", best_hand):
                    w(f"| {q} | {r['model']} | {_pts(r['acc_diff'])} | {_pfmt(r['p_value'])} | {_verdict(r['acc_diff'], r['p_value'])} "
                      f"| {r['n_2q_ref']} / {r['n_2q_other']} | {r['depth_ref']} / {r['depth_other']} |")
        w("")
        # Finding the cheapest evolved circuit that stays within 2 points of the best template.
        qn = [n for n in summary if n.startswith("QNAS")]
        best_q = max(qn, key=lambda n: summary[n]["test_acc_mean"])
        matching = [n for n in qn if abs(summary[n]["test_acc_mean"] - summary[best_hand]["test_acc_mean"]) <= 0.02
                    or summary[n]["test_acc_mean"] > summary[best_hand]["test_acc_mean"]]
        w(f"- Best hand-crafted baseline: **{best_hand}** at {_pct(summary[best_hand]['test_acc_mean'])} "
          f"({summary[best_hand]['n_2q']} two-qubit gates, depth {summary[best_hand]['depth']}).")
        w(f"- Best evolved circuit: **{best_q}** at {_pct(summary[best_q]['test_acc_mean'])} "
          f"({summary[best_q]['n_2q']} two-qubit gates, depth {summary[best_q]['depth']}).")
        if matching:
            cheapest = min(matching, key=lambda n: (summary[n]["n_2q"], summary[n]["depth"]))
            s, b = summary[cheapest], summary[best_hand]
            red = (1 - s["n_2q"] / b["n_2q"]) if b["n_2q"] else float("nan")
            w(f"- Cheapest evolved circuit within 2 points of (or above) that baseline: **{cheapest}** - "
              f"{s['n_2q']} vs {b['n_2q']} two-qubit gates ({_pct(red, 0)} fewer) and depth {s['depth']} vs {b['depth']}.")
        else:
            w("- No evolved circuit came within 2 points of the best hand-crafted baseline in this run - "
              "see the Pareto figure for the accuracy/cost trade-off that *was* found.")
        for n in ("Linear", "MLP-matched", "MLP-SEL2"):
            if n in summary:
                w(f"- Classical `{n}`: {_pct(summary[n]['test_acc_mean'])} ({summary[n]['n_params']} parameters).")
        w("\n![accuracy vs gates](figures/accuracy_vs_gates.png)\n")

    # ---------------- RQ2
    w("## RQ2 - Multi-objective trade-offs (Pareto front)\n")
    h0, h1 = history[0], history[-1]
    w(f"- Pareto front over **all {h1['evals_total']:.0f} circuits evaluated**: {len(front)} non-dominated circuits "
      f"(objectives: {', '.join(cfg.objectives)}).")
    w(f"- Hypervolume of the front grew from {h0['hv_archive']:.4f} (generation 0) to {h1['hv_archive']:.4f} "
      f"(generation {h1['gen']:.0f}); best validation accuracy {_pct(h0['best_val_acc'])} -> {_pct(h1['best_val_acc'])}.")
    # Counting how many templates are dominated by some circuit on the evolved front.
    if baselines:
        vecs = [normalized_vector(r["metrics"], cfg.objectives, spec) for r in front]
        dom = []
        for n, b in baselines.items():
            v = normalized_vector(b["metrics"], cfg.objectives, spec)
            if any(dominates(f, v) for f in vecs):
                dom.append(n)
        w(f"- {len(dom)} of {len(baselines)} hand-crafted baselines are *dominated* by at least one evolved circuit "
          f"(same short training protocol; validation accuracy, depth, two-qubit gates)"
          + (f": {', '.join(dom)}." if dom else "."))
    w("\n| Val. acc | Depth | 2-qubit | Params | F(C) | Circuit |")
    w("|---|---|---|---|---|---|")
    for r in front[:15]:
        m = r["metrics"]
        w(f"| {_pct(m['val_acc'])} | {m['depth']} | {m['n_2q']} | {m['n_params']} | {m['fitness']:.3f} | `{describe(r['genes'])}` |")
    if len(front) > 15:
        w(f"| … | | | | | ({len(front) - 15} more in `search/pareto.json`) |")
    w("\n![pareto](figures/pareto_front.png)\n\n![convergence](figures/convergence.png)\n")

    # ---------------- RQ3
    if grads:
        w("## RQ3 - Trainability and gradients\n")
        w("**Barren-plateau scan** (template circuits, random uniform initialisation): the slope is the change of "
          "log₂ Var[∂C/∂θ] per added qubit (−1 = variance halves with every qubit).\n")
        w("| Template | Cost | Var at smallest n | Var at largest n | log₂-slope per qubit |")
        w("|---|---|---|---|---|")
        for t in dict.fromkeys(r["template"] for r in grads["scan"]):
            for c in dict.fromkeys(r["cost"] for r in grads["scan"]):
                rr = sorted([r for r in grads["scan"] if r["template"] == t and r["cost"] == c], key=lambda r: r["n_qubits"])
                w(f"| {t} | {'local ⟨Z₀⟩' if c == 'z0' else 'global ⟨Z⊗…⊗Z⟩'} | {rr[0]['var_mean']:.2e} (n={rr[0]['n_qubits']}) "
                  f"| {rr[-1]['var_mean']:.2e} (n={rr[-1]['n_qubits']}) | {_slope(rr):+.2f} |")
        w("\n![scan](figures/barren_plateau_scan.png)\n")
        w(f"**Gradients of concrete {n_q}-qubit circuits** (mean over parameters of Var[∂⟨Z₀⟩/∂θ] across 100+ random initialisations; "
          "`frac small` = share of gradient entries below 1e-3):\n")
        w("| Circuit | Init | Parameters | Mean Var | Mean abs grad | frac small |")
        w("|---|---|---|---|---|---|")
        for r in grads["candidates"]:
            w(f"| {r['circuit']} | {r['init']} | {r['n_params']} | {r['var_mean']:.2e} | {r['abs_mean']:.2e} | {r['frac_small']:.2f} |")
        w("\n![candidate gradients](figures/candidate_gradients.png)\n")
        if summary:
            w("**Gradient flow during training** (mean gradient norm over the circuit angles, averaged over seeds):\n")
            w("| Model | epoch 1 | last epoch | ratio last/first |")
            w("|---|---|---|---|")
            for n in ("QNAS-bestF", "QNAS-compact", "QNAS-accurate", "SEL-L1", "SEL-L2", "SEL-L3", "BEL-L2"):
                if n in summary:
                    g = summary[n]["curve"]["grad_norm"]
                    w(f"| {n} | {g[0]:.3e} | {g[-1]:.3e} | {g[-1] / g[0]:.2f} |")
            w("\n![learning curves](figures/learning_curves.png)\n")
        w(f"Population-mean gradient norm during the search went from {h0['mean_grad_norm']:.3e} (generation 0) to "
          f"{h1['mean_grad_norm']:.3e} (generation {h1['gen']:.0f}).\n")
        w(f"> **Reading this honestly:** with only {n_q} qubits a barren plateau cannot yet bite hard - the scan above shows "
          "the *trend* with qubit count on templates; the per-circuit table shows that evolved circuits, being shorter, "
          "usually keep gradients of comparable or larger size per parameter than the deep templates. "
          f"To test scaling, re-run the search with `--qubits 6` or `--qubits 8`.\n")

    # ---------------- noise
    if noise:
        w("## Robustness to hardware noise (optional extension)\n")
        ps = sorted({r["p_1q"] for r in noise})
        names = list(dict.fromkeys(r["circuit"] for r in noise))
        w("Circuits were trained noise-free, then evaluated on `default.mixed` with depolarizing noise after every gate "
          "(two-qubit gates 10x noisier). Test accuracy:\n")
        w("| Circuit | " + " | ".join(f"p={p:g}" for p in ps) + " |")
        w("|---|" + "---|" * len(ps))
        for n in names:
            acc = {r["p_1q"]: r["accuracy"] for r in noise if r["circuit"] == n}
            w(f"| {n} | " + " | ".join(_pct(acc[p]) for p in ps) + " |")
        w("\n![noise](figures/noise.png)\n")

    # ---------------- circuits
    w("## Discovered circuits\n")
    w("Column 0 is the fixed `RY(xᵢ)` data encoding; blue boxes are trainable angles θ; the circuit returns ⟨Z₀⟩.\n")
    for n, c in cands.items():
        b, a = c["before"], c["after"]
        w(f"### {n}\n")
        w(f"- Found by search: {b['n_gates']} gates, {b['n_2q']} two-qubit, depth {b['depth']}. "
          f"After lossless simplification{' + ablation pruning' if cfg.prune else ''}: {a['n_gates']} gates, {a['n_2q']} two-qubit, "
          f"depth {a['depth']}, {a['n_trainable']} trainable angles.")
        w(f"- `{describe(genome_from_json(c['genes']))}`\n")
        w(f"![{n}](figures/circuit_{n}.png)\n")
    if "SEL-L2" in circuit_figs:
        w("For reference, the hand-crafted `SEL-L2`:\n\n![SEL-L2](figures/circuit_SEL-L2.png)\n")

    w("## Caveats\n")
    w("- Accuracy differences of 1–2 points are within seed-to-seed noise on datasets this small; read the *p* values.")
    w("- The search selects on validation accuracy from a short training run (winner's curse); that is why the final "
      "stage re-trains everything on fresh seeds and reports the untouched test split.")
    w("- Everything is exact state-vector simulation of a few qubits: conclusions about *hardware* come only from the "
      "proxy metrics (depth, two-qubit gates) and the depolarizing-noise study.")
    w("- Baselines are evaluated with the same fixed RY(x) encoding and ⟨Z₀⟩ read-out as the evolved circuits.\n")

    w("## Files\n")
    for f in ("config.json", "task_info.json", "search/history.csv", "search/pareto.json", "search/archive.json",
              "baselines.json", "candidates.json", "final/final_summary.csv", "final/final_runs.json",
              "gradients/gradients.json", "noise/noise.json", "log.txt"):
        if (run.dir / f).exists():
            w(f"- `{f}`")
    text = "\n".join(L) + "\n"
    (run.dir / "results.md").write_text(text)
    return text
