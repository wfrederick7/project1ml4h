"""Generate appendix figures for Section 3 (Representation Learning)."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "rep_learning" / "results"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load pretrain summary ──
with open(RESULTS_DIR / "q3_1_pretrain" / "summary.json") as f:
    pretrain = json.load(f)

train_hist = pretrain["train_history"]
eval_hist = pretrain["eval_history"]

epochs = [h["epoch"] for h in train_hist]
train_nce = [h["train_nce"] for h in train_hist]
alignment = [h["alignment"] for h in train_hist]
uniformity = [h["uniformity"] for h in train_hist]
collapse = [h["collapse"] for h in train_hist]

eval_epochs = [h["epoch"] for h in eval_hist]
val_nce = [h["val_nce"] for h in eval_hist]
val_auroc = [h["val_auroc"] for h in eval_hist]
val_auprc = [h["val_auprc"] for h in eval_hist]

# ── Figure 1: Pretraining curves (2x2) ──
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# Train + Val NCE loss
axes[0, 0].plot(epochs, train_nce, label="Train NCE", color="#3b82f6", linewidth=1)
axes[0, 0].plot(eval_epochs, val_nce, "o-", label="Val NCE", color="#ef4444", markersize=3, linewidth=1)
axes[0, 0].axvline(x=144, color="gray", linestyle="--", alpha=0.5, label="Best ckpt (ep 144)")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("InfoNCE Loss")
axes[0, 0].set_title("(a) InfoNCE Loss")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

# Alignment & Uniformity
ax1 = axes[0, 1]
ax1.plot(epochs, alignment, label="Alignment", color="#f59e0b", linewidth=1)
ax2 = ax1.twinx()
ax2.plot(epochs, uniformity, label="Uniformity", color="#8b5cf6", linewidth=1)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Alignment", color="#f59e0b")
ax2.set_ylabel("Uniformity", color="#8b5cf6")
ax1.set_title("(b) Alignment & Uniformity")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax1.grid(alpha=0.3)

# Collapse
axes[1, 0].plot(epochs, collapse, color="#10b981", linewidth=1)
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Avg Pairwise Cosine Sim")
axes[1, 0].set_title("(c) Collapse Monitoring")
axes[1, 0].grid(alpha=0.3)

# Linear probe during pretraining
axes[1, 1].plot(eval_epochs, val_auroc, "s-", label="Val AuROC", color="#3b82f6", markersize=3, linewidth=1)
axes[1, 1].plot(eval_epochs, val_auprc, "^-", label="Val AuPRC", color="#ef4444", markersize=3, linewidth=1)
axes[1, 1].axvline(x=144, color="gray", linestyle="--", alpha=0.5, label="Best ckpt")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Score")
axes[1, 1].set_title("(d) Linear Probe (Val)")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3)

fig.suptitle("Q3.1 — Contrastive Pretraining Curves", fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "pretrain_curves.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIG_DIR / 'pretrain_curves.png'}")

# ── Figure 2: Label scarcity comparison ──
with open(RESULTS_DIR / "q3_2_label_scarce" / "summary.json") as f:
    scarce = json.load(f)

n_labels = [100, 500, 1000, 4000]
x_labels = ["100", "500", "1000", "Full"]

models = {
    "UniLSTM": "uni_lstm_q2_2",
    "BiLSTM": "bi_lstm_q2_2",
    "Transformer": "transformer_q2_3a",
}
colors_scratch = {"UniLSTM": "#94a3b8", "BiLSTM": "#64748b", "Transformer": "#475569"}
probe_color = "#ef4444"

fig, (ax_roc, ax_prc) = plt.subplots(1, 2, figsize=(10, 4.5))

for name, key in models.items():
    data = scarce["experiment_a_from_scratch"][key]
    aurocs = [data[n]["test"]["auroc"] for n in ["100", "500", "1000", "full"]]
    auprcs = [data[n]["test"]["auprc"] for n in ["100", "500", "1000", "full"]]
    ax_roc.plot(x_labels, aurocs, "o--", label=f"{name} (scratch)", color=colors_scratch[name], markersize=4)
    ax_prc.plot(x_labels, auprcs, "o--", label=f"{name} (scratch)", color=colors_scratch[name], markersize=4)

probe_data = scarce["experiment_b_linear_probe"]
probe_aurocs = [probe_data[n]["test"]["auroc"] for n in ["100", "500", "1000", "full"]]
probe_auprcs = [probe_data[n]["test"]["auprc"] for n in ["100", "500", "1000", "full"]]
ax_roc.plot(x_labels, probe_aurocs, "s-", label="Linear probe (pretrained)", color=probe_color, markersize=5, linewidth=2)
ax_prc.plot(x_labels, probe_auprcs, "s-", label="Linear probe (pretrained)", color=probe_color, markersize=5, linewidth=2)

for ax, title in [(ax_roc, "Test AuROC"), (ax_prc, "Test AuPRC")]:
    ax.set_xlabel("Number of Labelled Patients")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

fig.suptitle("Q3.2 — Label Scarcity: Pretrained vs From-Scratch", fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "label_scarcity.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIG_DIR / 'label_scarcity.png'}")

print("Done.")
