"""Example 1 - one qubit, by hand.  Run:  python examples/01_hello_qubit.py

Learning goals: state, rotation, measurement, and how a *gradient of a circuit* is computed.
A feature x is encoded as an RY(x) rotation, a trainable RX(theta) follows, and we measure <Z>.
Analytically  <Z> = cos(x) * cos(theta).
"""
# -----------------------------------------------------------------------------
# QNAS - Evolutionary Quantum Neural Architecture Search
# Author : Swayam Shakya
# File   : examples/01_hello_qubit.py
# Purpose: showing one qubit by hand: encoding, a trainable rotation, measurement and its gradient.
# -----------------------------------------------------------------------------

import math

import pennylane as qml
import torch

# Creating a one-qubit simulator.
dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(x, theta):
    qml.RY(x, wires=0)       # encoding the data value as a rotation angle
    qml.RX(theta, wires=0)   # applying the trainable gate
    return qml.expval(qml.PauliZ(0))


# Evaluating the circuit and back-propagating to get d<Z>/d theta.
x = torch.tensor(0.7)
theta = torch.tensor(0.4, requires_grad=True)
value = circuit(x, theta)
value.backward()
# Printing the simulated values next to the pen-and-paper formulas.
print(f"<Z>            = {float(value.detach()):.6f}   (cos(x)cos(theta) = {math.cos(0.7) * math.cos(0.4):.6f})")
print(f"d<Z>/d theta   = {float(theta.grad):.6f}   (-cos(x)sin(theta)  = {-math.cos(0.7) * math.sin(0.4):.6f})")
print(qml.draw(circuit)(x, theta))
