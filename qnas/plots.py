"""All figures. Matplotlib only; one fixed palette where each colour always means the same model family.

    QNAS (evolved)  blue   #2a78d6        SEL (hand-crafted)  orange #eb6834
    BEL (hand-crafted) aqua #1baf7a       classical baselines neutral gray

Each chart has a legend whenever it has two or more series, thin marks with a surface-coloured edge,
recessive grid lines and direct labels where there is room.  Identity is never colour-only (markers,
line styles and text labels repeat it).
"""

# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : qnas/plots.py
# Purpose: drawing every figure of the report with matplotlib (Pareto fronts,
#          convergence, comparisons, gradients, noise and circuit diagrams).
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from .genome import _FIXED_LABELS, DATA, TRAIN, Gene

# Fixing the palette: background, text inks, grid, and one colour per model family.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C_QNAS, C_SEL, C_BEL, C_CLS = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"
QNAS_SHADES = {"QNAS-bestF": "#2a78d6", "QNAS-compact": "#104281", "QNAS-accurate": "#6da7ec"}


def family_color(name: str) -> str:
    # Colouring by family (QNAS / SEL / BEL / classical) so a model keeps its colour in every chart.
    if name.startswith("QNAS"):
        return QNAS_SHADES.get(name, C_QNAS)
    if name.startswith("SEL"):
        return C_SEL
    if name.startswith("BEL"):
        return C_BEL
    return C_CLS


def _style() -> None:
    # Applying one quiet house style: light background, no top/right spines, faint grid.
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#b9b8b2", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 9.5, "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2.0,
        "figure.dpi": 110, "savefig.dpi": 170,
    })


def _save(fig, path) -> str:
    # Saving tightly cropped and closing the figure to free memory.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ------------------------------------------------------------------------------
# 1. Plotting the Pareto front (RQ2)
# ------------------------------------------------------------------------------
def plot_pareto(archive: List[dict], front: List[dict], baselines: Dict[str, dict], path) -> str:
    """Validation accuracy vs two-qubit gates (left) and vs depth (right).

    Everything is evaluated with the short search protocol, so baselines sit on the same axes as the
    evolved circuits.  Grey = every circuit tried during the search; blue = the Pareto front.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    rng = np.random.default_rng(0)
    for ax, key, xlabel in ((axes[0], "n_2q", "two-qubit gates"), (axes[1], "depth", "circuit depth")):
        # Plotting every evaluated circuit in grey, jittered sideways so integer counts do not overlap.
        ax_x = np.array([r["metrics"][key] for r in archive], float)
        ax_y = np.array([r["metrics"]["val_acc"] for r in archive], float)
        jitter = rng.uniform(-0.18, 0.18, size=len(ax_x))
        ax.scatter(ax_x + jitter, ax_y, s=16, color="#b4b3ad", alpha=0.45, linewidths=0, zorder=1)
        fx = np.array([r["metrics"][key] for r in front], float)
        fy = np.array([r["metrics"]["val_acc"] for r in front], float)
        # Drawing the 2-D staircase of this projection (best accuracy reachable at <= x).
        order = np.argsort(fx)
        best, sx, sy = -1.0, [], []
        for i in order:
            if fy[i] > best:
                best = fy[i]
                sx.append(fx[i]); sy.append(fy[i])
        ax.step(sx, sy, where="post", color=C_QNAS, lw=1.6, alpha=0.9, zorder=2)
        ax.scatter(fx, fy, s=58, color=C_QNAS, edgecolors=SURFACE, linewidths=1.4, zorder=4)
        # Overlaying the hand-crafted templates with their own markers and labels.
        for name, b in baselines.items():
            m = b["metrics"]
            color, marker = (C_SEL, "s") if name.startswith("SEL") else (C_BEL, "^")
            ax.scatter([m[key]], [m["val_acc"]], s=70, color=color, marker=marker, edgecolors=SURFACE,
                       linewidths=1.4, zorder=5)
            ax.annotate(name, (m[key], m["val_acc"]), textcoords="offset points", xytext=(5, 4), fontsize=7.5, color=INK2)
        ax.set_xlabel(xlabel)
        ax.set_xlim(left=-0.6)
    axes[0].set_ylabel("validation accuracy")
    axes[0].set_title("Accuracy vs two-qubit gates", loc="left")
    axes[1].set_title("Accuracy vs circuit depth", loc="left")
    # Adding one shared legend under both panels.
    handles = [
        Line2D([], [], marker="o", ls="", color="#b4b3ad", markersize=5, label="every circuit evaluated"),
        Line2D([], [], marker="o", ls="-", color=C_QNAS, markersize=7, markeredgecolor=SURFACE, label="QNAS Pareto front"),
        Line2D([], [], marker="s", ls="", color=C_SEL, markersize=7, label="StronglyEntanglingLayers (SEL)"),
        Line2D([], [], marker="^", ls="", color=C_BEL, markersize=7, label="BasicEntanglerLayers (BEL)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    return _save(fig, path)


# ------------------------------------------------------------------------------
# 2. Plotting search convergence (RQ2) and gradient size per generation (RQ3)
# ------------------------------------------------------------------------------
def plot_convergence(history: List[dict], path) -> str:
    _style()
    # Drawing four panels: accuracy, circuit size, hypervolume and starting gradient norm per generation.
    g = np.array([h["gen"] for h in history])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    ax.plot(g, [h["best_val_acc"] for h in history], color=C_QNAS, label="best in population")
    ax.plot(g, [h["mean_val_acc"] for h in history], color=C_QNAS, ls="--", alpha=0.7, label="population mean")
    ax.set_title("Validation accuracy"); ax.set_ylabel("accuracy"); ax.legend(loc="lower right")
    ax = axes[0, 1]
    ax.plot(g, [h["mean_n2q"] for h in history], color=C_QNAS, label="two-qubit gates")
    ax.plot(g, [h["mean_depth"] for h in history], color="#104281", ls="--", label="depth")
    ax.set_title("Circuit size (population mean)"); ax.set_ylabel("count"); ax.legend(loc="upper left")
    ax = axes[1, 0]
    ax.plot(g, [h["hv_archive"] for h in history], color=C_QNAS)
    ax.set_title("Hypervolume of the Pareto front"); ax.set_ylabel("hypervolume (higher = better)")
    ax = axes[1, 1]
    ax.plot(g, [h["mean_grad_norm"] for h in history], color=C_QNAS)
    ax.set_title("Gradient norm at start of training"); ax.set_ylabel("mean ||grad||  (population)")
    for ax in axes[1]:
        ax.set_xlabel("generation")
    return _save(fig, path)


# ------------------------------------------------------------------------------
# 3. Plotting the final comparison (RQ1)
# ------------------------------------------------------------------------------
def plot_final_comparison(summary: Dict[str, dict], path) -> str:
    """Test accuracy (mean +- std, dots = individual seeds) of every model, with its circuit cost."""
    _style()
    # Sorting models by accuracy and drawing bars, error bars and the individual seeds.
    names = sorted(summary, key=lambda n: summary[n]["test_acc_mean"])
    fig, ax = plt.subplots(figsize=(9.5, 0.46 * len(names) + 1.8))
    for i, n in enumerate(names):
        s = summary[n]
        c = family_color(n)
        ax.barh(i, s["test_acc_mean"], color=c, height=0.62, edgecolor=SURFACE, linewidth=1.2, zorder=2)
        ax.errorbar(s["test_acc_mean"], i, xerr=s["test_acc_std"], color=INK, capsize=3, lw=1.2, zorder=4)
        ax.scatter(s["test_acc_runs"], [i] * len(s["test_acc_runs"]), s=14, color=INK, alpha=0.55, zorder=5, linewidths=0)
        # Writing the score and the circuit cost to the right of each bar.
        cost = "classical" if s["kind"] != "quantum" else f"{s['n_2q']} 2q gates · depth {s['depth']} · {s['n_params']} params"
        ax.text(1.0, i, f"  {s['test_acc_mean']*100:.1f}%  |  {cost}", va="center", fontsize=8, color=INK2,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    lo = min(summary[n]["test_acc_mean"] for n in names)
    ax.set_xlim(max(0.4, lo - 0.12), 1.0)
    ax.set_xlabel("test accuracy")
    ax.set_title("Final comparison on the held-out test set")
    handles = [Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in
               ((C_QNAS, "QNAS (evolved)"), (C_SEL, "SEL (hand-crafted)"), (C_BEL, "BEL (hand-crafted)"), (C_CLS, "classical"))]
    ax.legend(handles=handles, loc="lower right", ncol=1)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_accuracy_vs_gates(summary: Dict[str, dict], path) -> str:
    """The headline RQ1 chart: test accuracy against the number of two-qubit gates."""
    _style()
    # Plotting each quantum model at (two-qubit gates, test accuracy), alternating label offsets.
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    k = 0
    for n, s in summary.items():
        if s["kind"] != "quantum":
            continue
        k += 1
        marker = "o" if n.startswith("QNAS") else ("s" if n.startswith("SEL") else "^")
        ax.errorbar(s["n_2q"], s["test_acc_mean"], yerr=s["test_acc_std"], fmt=marker, color=family_color(n),
                    markersize=9, markeredgecolor=SURFACE, markeredgewidth=1.4, capsize=3, lw=1.4, zorder=3)
        ax.annotate(n, (s["n_2q"], s["test_acc_mean"]), textcoords="offset points", xytext=(8, 6 if k % 2 else -13), fontsize=8, color=INK2)
    # Marking the classical baselines as horizontal reference lines.
    for n, ls in (("Linear", ":"), ("MLP-matched", "--"), ("MLP-SEL2", "-.")):
        if n in summary:
            ax.axhline(summary[n]["test_acc_mean"], color=C_CLS, ls=ls, lw=1.4)
            ax.text(ax.get_xlim()[1], summary[n]["test_acc_mean"], f" {n}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("two-qubit gates (fewer = more hardware-friendly)")
    ax.set_ylabel("test accuracy (mean ± std over seeds)")
    ax.set_title("Accuracy vs hardware cost")
    handles = [
        Line2D([], [], marker="o", ls="", color=C_QNAS, markersize=8, label="QNAS (evolved)"),
        Line2D([], [], marker="s", ls="", color=C_SEL, markersize=8, label="SEL (hand-crafted)"),
        Line2D([], [], marker="^", ls="", color=C_BEL, markersize=8, label="BEL (hand-crafted)"),
        Line2D([], [], ls=":", color=C_CLS, label="classical baselines"),
    ]
    ax.legend(handles=handles, loc="upper left")
    return _save(fig, path)


def plot_learning_curves(summary: Dict[str, dict], names: Sequence[str], path) -> str:
    _style()
    # Drawing loss, test accuracy and gradient norm per epoch; line style separates models of one family.
    names = [n for n in names if n in summary]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    styles = ["-", "--", "-.", ":"]
    seen_family: Dict[str, int] = {}
    for n in names:
        fam = n.split("-")[0]
        k = seen_family.get(fam, 0)
        seen_family[fam] = k + 1
        c = family_color(n)
        cur = summary[n]["curve"]
        for ax, key in zip(axes, ("loss", "test_acc", "grad_norm")):
            ax.plot(cur["epoch"], cur[key], color=c, ls=styles[k % 4], label=n)
    axes[0].set_title("Training loss"); axes[0].set_ylabel("BCE loss")
    axes[1].set_title("Test accuracy"); axes[1].set_ylabel("accuracy")
    axes[2].set_title("Gradient flow (RQ3)"); axes[2].set_ylabel("mean ||grad|| of circuit angles")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[1].legend(loc="lower right", fontsize=8)
    return _save(fig, path)


# ------------------------------------------------------------------------------
# 4. Plotting gradients and barren plateaus (RQ3)
# ------------------------------------------------------------------------------
def plot_barren_scan(rows: List[dict], path) -> str:
    _style()
    # Drawing one panel per cost function, with a 2^-n guide line for comparison.
    costs = sorted({r["cost"] for r in rows}, key=lambda c: (c != "z0", c))
    titles = {"z0": "Local cost  <Z₀>", "global_z": "Global cost  <Z⊗…⊗Z>"}
    templates = list(dict.fromkeys(r["template"] for r in rows))
    fig, axes = plt.subplots(1, len(costs), figsize=(5.6 * len(costs), 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cost in zip(axes, costs):
        for t, ls, mk in zip(templates, ("-", "--"), ("o", "s")):
            rr = sorted([r for r in rows if r["cost"] == cost and r["template"] == t], key=lambda r: r["n_qubits"])
            xs = [r["n_qubits"] for r in rr]
            ax.plot(xs, [r["var_mean"] for r in rr], color=C_SEL, ls=ls, marker=mk, markersize=7,
                    markeredgecolor=SURFACE, label=t)
        xs = sorted({r["n_qubits"] for r in rows})
        # Anchoring the guide line at the largest variance measured at the smallest size.
        ref = [r["var_mean"] for r in rows if r["n_qubits"] == xs[0] and r["cost"] == cost]
        anchor = max(ref) if ref else 0.06
        ax.plot(xs, [anchor * 2.0 ** (-(x - xs[0])) for x in xs], color=C_CLS, ls=":", lw=1.4, label="∝ 2⁻ⁿ (exponential decay)")
        ax.set_yscale("log")
        ax.set_xlabel("number of qubits n")
        ax.set_xticks(xs)
        ax.set_title(titles.get(cost, cost))
        ax.legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel("Var[ ∂C/∂θ ]  (random uniform init)")
    return _save(fig, path)


def plot_candidate_gradients(rows: List[dict], path) -> str:
    _style()
    # Drawing one panel per initialisation, with log-scaled variance bars.
    inits = [i for i in ("uniform", "small") if any(r["init"] == i for r in rows)]
    titles = {"uniform": "Random init  θ ~ U[0, 2π)", "small": "Near-identity init  θ ~ N(0, 0.3²)"}
    fig, axes = plt.subplots(1, len(inits), figsize=(6.2 * len(inits), 0.4 * len({r["circuit"] for r in rows}) + 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    names = list(dict.fromkeys(r["circuit"] for r in rows))
    for ax, init in zip(axes, inits):
        for i, n in enumerate(names):
            r = next(r for r in rows if r["circuit"] == n and r["init"] == init)
            ax.barh(i, r["var_mean"], color=family_color(n), height=0.6, edgecolor=SURFACE, linewidth=1.2, zorder=2)
            ax.text(r["var_mean"], i, f"  {r['var_mean']:.2e}", va="center", fontsize=8, color=INK2)
        ax.set_xscale("log")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("mean Var[ ∂<Z₀>/∂θ ] over parameters")
        ax.set_title(titles[init])
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    return _save(fig, path)


# ------------------------------------------------------------------------------
# 5. Plotting noise robustness
# ------------------------------------------------------------------------------
def plot_noise(rows: List[dict], path) -> str:
    _style()
    # Plotting accuracy against the noise level for every trained circuit, with the chance line.
    names = list(dict.fromkeys(r["circuit"] for r in rows))
    ps = sorted({r["p_1q"] for r in rows})
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    styles = ["-", "--", "-.", ":"]
    counts: Dict[str, int] = {}
    for n in names:
        fam = n.split("-")[0]
        k = counts.get(fam, 0); counts[fam] = k + 1
        rr = sorted([r for r in rows if r["circuit"] == n], key=lambda r: r["p_1q"])
        ax.plot(range(len(ps)), [r["accuracy"] for r in rr], color=family_color(n), ls=styles[k % 4], marker="o",
                markersize=6, markeredgecolor=SURFACE, label=n)
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels([f"{p:g}" for p in ps])
    ax.set_xlabel("single-qubit depolarizing rate p   (two-qubit gates: 10 × p)")
    ax.set_ylabel("test accuracy of the noise-free-trained circuit")
    ax.set_title("Accuracy on noisy hardware (default.mixed)")
    ax.axhline(0.5, color=C_CLS, lw=1, ls=":")
    ax.text(len(ps) - 1, 0.5, "chance ", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.legend(loc="lower left", fontsize=8)
    return _save(fig, path)


# ------------------------------------------------------------------------------
# 6. Drawing a circuit diagram from a genome (symbolic parameters)
# ------------------------------------------------------------------------------
def draw_genome(genes: Sequence[Gene], n_qubits: int, n_features: int, path, title: Optional[str] = None) -> str:
    """Quantum-circuit diagram. Column 0 is the fixed RY(x_i) encoding; θ = trainable angle, x_i = input feature."""
    _style()
    # Placing each gate in the earliest free column (same ASAP rule as circuit_depth).
    level = {w: 0 for w in range(n_qubits)}
    cols = []  # collecting (column, gene) pairs
    for g in genes:
        c = max(level[w] for w in g.wires) + 1
        for w in g.wires:
            level[w] = c
        cols.append((c, g))
    n_cols = max([c for c, _ in cols], default=0) + 1
    fig, ax = plt.subplots(figsize=(max(5.0, 0.85 * n_cols + 1.6), 0.8 * n_qubits + 1.0))
    ax.set_xlim(-0.8, n_cols - 0.2)
    ax.set_ylim(n_qubits - 0.4, -0.7)
    ax.axis("off")
    # Drawing the qubit wires and their labels.
    for q in range(n_qubits):
        ax.plot([-0.55, n_cols - 0.35], [q, q], color="#9a998f", lw=1.0, zorder=1)
        ax.text(-0.75, q, f"q{q}", ha="right", va="center", fontsize=9, color=INK2)

    def box(x, y, label, fc):
        # Drawing a rounded gate box with its label.
        ax.add_patch(FancyBboxPatch((x - 0.3, y - 0.27), 0.6, 0.54, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec="#7a796f", lw=1.0, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=8, color=INK, zorder=4)

    # Drawing column 0 (the fixed encoding), then every evolved gate.
    for i in range(n_features):
        box(0, i % n_qubits, f"RY\nx{i}", "#e8e7e2")
    for c, g in cols:
        if g.is_two_qubit:
            a, b = g.wires
            ax.plot([c, c], [a, b], color=INK, lw=1.4, zorder=2)
            if g.gate == "CNOT":
                ax.add_patch(Circle((c, a), 0.075, color=INK, zorder=4))
                ax.add_patch(Circle((c, b), 0.17, fc=SURFACE, ec=INK, lw=1.4, zorder=4))
                ax.plot([c - 0.17, c + 0.17], [b, b], color=INK, lw=1.4, zorder=5)
                ax.plot([c, c], [b - 0.17, b + 0.17], color=INK, lw=1.4, zorder=5)
            elif g.gate == "CZ":
                for w in (a, b):
                    ax.add_patch(Circle((c, w), 0.075, color=INK, zorder=4))
            else:  # drawing SWAP as two crosses
                for w in (a, b):
                    ax.plot([c - 0.12, c + 0.12], [w - 0.12, w + 0.12], color=INK, lw=1.6, zorder=5)
                    ax.plot([c - 0.12, c + 0.12], [w + 0.12, w - 0.12], color=INK, lw=1.6, zorder=5)
        elif g.is_rotation:
            if g.mode == TRAIN:
                lab, fc = f"{g.gate}\nθ", "#cde2fb"
            elif g.mode == DATA:
                lab, fc = f"{g.gate}\nx{g.arg}", "#fbe3d6"
            else:
                lab, fc = f"{g.gate}\n{_FIXED_LABELS[g.arg]}", "#ecebe6"
            box(c, g.wires[0], lab, fc)
        else:
            box(c, g.wires[0], "H", "#ecebe6")
    ax.text(n_cols - 0.35, -0.55, "measure <Z₀>", ha="right", va="center", fontsize=8.5, color=INK2)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    return _save(fig, path)
