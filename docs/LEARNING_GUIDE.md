# Learning guide - quantum machine learning through this project

*Written by Swayam Shakya as a companion to the QNAS code.*

Read this next to the code. Each section names the idea, the file where it lives, and a small experiment you can run.

## 1. Qubits, gates, measurement  -> `examples/01_hello_qubit.py`
A qubit state is a unit vector; gates are rotations of it. `RY(x)` tilts the qubit by angle *x*, `RX(θ)` rotates about the X axis,
and measuring Pauli-Z returns `<Z> ∈ [-1, 1]`. For one qubit `<Z> = cos(x)·cos(θ)`. The example checks PennyLane's answer against this formula.

## 2. Encoding classical data  -> `qnas/circuit.py::apply_circuit`
Our classifier turns 4 numbers into 4 rotation angles (`RY(x_i)` on qubit *i*, angle encoding). The classical front end
(`qnas/data.py`: PCA or a CNN autoencoder) squeezes an image to 4 numbers, then `AngleScaler` maps them into [0, π]
so that different inputs give different rotations.

## 3. Variational circuits and hybrid training  -> `qnas/model.py`, `qnas/train.py`, `examples/03_hybrid_classifier.py`
Trainable rotation angles θ are tuned by gradient descent, exactly like neural-network weights. The model is
`logit = scale · <Z0> + bias`; PyTorch computes gradients through the simulated circuit (`diff_method="backprop"`).
*Try:* run example 3, then change `strongly_entangling(4, 2)` to `basic_entangler(4, 2)` and compare. Why does BEL do so much worse here?
(Hint: BEL has one RX angle per qubit per layer, SEL has three angles per qubit, so BEL is far less expressive; check `count_trainable` for both.)

## 4. Entanglement is a resource and a cost  -> `qnas/templates.py`, `qnas/genome.py::count_two_qubit`
CNOT/CZ gates create entanglement. They are also the noisiest operations on real hardware, which is why the search
minimises the number of two-qubit gates and the circuit depth, not only the error.

## 5. Circuit architecture as a genome  -> `qnas/genome.py`, `qnas/space.py`, `examples/02_build_a_genome.py`
A circuit is a list of `Gene(gate, wires, mode, arg)`. `mode` says whether a rotation is trainable, fixed, or data-dependent.
Crossover swaps sub-circuits between parents, mutation changes/inserts/deletes gates, and `repair` enforces limits so offspring stay legal.

## 6. Evolutionary search  -> `qnas/evolution.py`, `qnas/fitness.py`, `examples/04_tiny_search.py`
Each candidate circuit is trained for a few epochs and scored. Two modes:
* **weighted** - one number `F = α·acc − β·depth/D − γ·N2Q/N`. Simple, but you must pick α, β, γ in advance.
* **nsga2** (default) - keeps a whole *Pareto front*: circuits for which no other is better in every objective. You pick the trade-off afterwards.
Evaluation is deterministic (the random seed comes from a hash of the genome) so results can be cached and runs reproduced.
`qnas/pareto.py` has the exact hypervolume indicator, a single number that rises as the front improves.

## 7. Is "better architecture" real?  -> `qnas/final.py`
The search picks circuits using the validation set, which biases their score upwards. So the chosen circuits are
retrained from scratch on several seeds and compared on the untouched **test** set with Welch t-tests (`compare_to_reference`).
Always report the spread over seeds; differences smaller than it are noise.

## 8. Barren plateaus  -> `qnas/gradients.py`
For deep random circuits the gradient variance shrinks exponentially with the number of qubits (McClean et al., 2018), so training stalls.
`barren_plateau_scan` measures `Var[∂<Z0>/∂θ]` for 2…N qubits. With only 4 qubits the effect is mild - the scan shows the trend, not a disaster.
The batched trick: stack many random initialisations into a `(B, P)` weight matrix and call `out.sum().backward()`; row *b* of the gradient is the gradient for init *b*.

## 9. Noise  -> `qnas/noise.py`
`default.mixed` simulates density matrices, so a depolarizing channel can follow every gate. Circuits are trained noise-free
and evaluated noisy: fewer two-qubit gates means graceful degradation. *Try:* `python -m qnas noise --out runs/standard`.

## 10. Simplification  -> `qnas/simplify.py`
Evolved circuits contain redundancy. Lossless rules merge neighbouring same-axis rotations and cancel back-to-back CNOTs.
Ablation pruning then deletes any gate whose removal does not hurt validation accuracy.

## Suggested exercises
1. Change `--qubits 6` and see how the search space and costs change.
2. Run with `--data-reuploading` and look at why accuracy rises while the search objective (cost) favours trivial circuits (a good lesson in objective design).
3. Run `--cold-start` and compare with the warm-started default: how much does evolution need good seeds at this budget? (Compare with `runs/cold_start`.)
4. Add a new gate (e.g. `CRZ`) in `genome.py` / `circuit.py` and extend the tests.
5. Use a real dataset: put `pneumoniamnist.npz` in `data/` and run `--dataset pneumoniamnist`.

## Reading list
McClean et al. 2018 (barren plateaus); Grant et al. 2019 (identity-block initialisation); Schuld et al. 2020 (circuit-centric classifiers);
Du et al. 2022 (quantum architecture search); Deb et al. 2002 (NSGA-II); PennyLane demos on variational classifiers and `TorchLayer`.
