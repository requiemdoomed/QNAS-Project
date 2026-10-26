"""Example 3 - train a hybrid quantum-classical classifier.  Run:  python examples/03_hybrid_classifier.py

Pipeline: 8x8 digit images -> PCA (classical, 4 features) -> angles in [0, pi] -> quantum circuit
(StronglyEntanglingLayers, 2 layers) -> <Z0> -> scale*<Z0>+bias -> logistic loss.  Takes about 10 seconds.
"""
# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : examples/03_hybrid_classifier.py
# Purpose: training one hybrid quantum-classical classifier on the digits task.
# -----------------------------------------------------------------------------

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # running without installing the package

import torch

from qnas.data import prepare_task
from qnas.model import HybridQuantumModel
from qnas.templates import strongly_entangling
from qnas.train import TrainConfig, evaluate_model, fit

# Preparing the even-vs-odd digits task with 4 PCA features.
torch.manual_seed(0)
task = prepare_task("digits", ((0, 2, 4, 6, 8), (1, 3, 5, 7, 9)), n_features=4, extractor="pca", seed=0)
print("split sizes:", task.sizes())

# Wrapping a 2-layer StronglyEntanglingLayers circuit as a PyTorch model and training it.
model = HybridQuantumModel(strongly_entangling(4, 2), n_qubits=4, n_features=4)
print("trainable parameters:", model.n_parameters)
res = fit(model, task.x_train, task.y_train, TrainConfig(epochs=10, batch_size=64, lr=0.1), seed=0,
          eval_sets={"val": (task.x_val, task.y_val)}, eval_every_epoch=True)
# Printing the learning curve and the final test accuracy.
for h in res["history"]:
    print(f"epoch {h['epoch']:2d}  loss {h['loss']:.3f}  |grad| {h['grad_norm']:.3f}  val acc {h['val']['acc']:.3f}")
print("test accuracy:", evaluate_model(model, task.x_test, task.y_test)["acc"])
