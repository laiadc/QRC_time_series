import itertools
import random
from random import sample
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
from numpy.lib.scimath import sqrt as csqrt
import pennylane as qml
from pennylane import numpy as pnp
from scipy import stats
from scipy.sparse import random as sparse_random

from scipy.sparse.linalg import eigs
from scipy.stats import unitary_group
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (mean_absolute_error, median_absolute_error,
                             r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class QuantumRC():
    def __init__(self, nqbits=6, num_gates=20, gates_set=['CNOT', 'T', 'H'], alpha=1e-1, t=10):
        self.nqbits = nqbits
        self.num_gates = num_gates
        self.gates_set = gates_set
        self.lm = None
        self.rho = None
        self.alpha = alpha
        self.t = t
        self.dev = qml.device("default.qubit", wires=self.nqbits + 1)
        if gates_set == 'Ising':
            self.unitary_circuit = self.get_unitary_ising()
        else:
            self.unitary_circuit = self.get_unitary()
        self.observables = self.get_observables()

    def get_unitary_ising(self):
        def circuit(theta=0.0):
            for i in range(self.nqbits):
                qml.RX(theta, wires=i)
            for i in range(self.nqbits):
                qml.CNOT(wires=[i, (i + 1) % (self.nqbits + 1)])
        return circuit

    def get_unitary(self):
        def circuit():
            for i in range(self.num_gates):
                gate = random.choice(self.gates_set)
                if gate == 'CNOT':
                    qbit1 = random.randint(0, self.nqbits)
                    qbit2 = random.randint(0, self.nqbits)
                    while qbit2 == qbit1:
                        qbit2 = random.randint(0, self.nqbits)
                    qml.CNOT(wires=[qbit1, qbit2])
                elif gate == 'X':
                    qbit = random.randint(0, self.nqbits)
                    qml.PauliX(wires=qbit)
                elif gate == 'S':
                    qbit = random.randint(0, self.nqbits)
                    qml.S(wires=qbit)
                elif gate == 'H':
                    qbit = random.randint(0, self.nqbits)
                    qml.Hadamard(wires=qbit)
                elif gate == 'T':
                    qbit = random.randint(0, self.nqbits)
                    qml.T(wires=qbit)
        return circuit

    def get_observables(self):
        observables = []
        for i in range(self.nqbits + 1):
            observables.append(qml.PauliX(i))
            observables.append(qml.PauliZ(i))
        return observables

    def train(self, X_train_scaled, y_train_scaled, scaler):
        psi_datas = np.zeros((y_train_scaled.shape[0] + 1, 2))
        psi_datas[0] = np.array([1, 0])
        for i in range(y_train_scaled.shape[0]):
            val = np.clip(y_train_scaled[i], 0, 1)
            psi_datas[i + 1] = np.sqrt(1 - val) * np.array([1, 0]) + np.sqrt(val) * np.array([0, 1])
        self.psi_datas_train = psi_datas

        # Initialisiere das Bell-State für die restlichen Qubits
        bell_state = np.ones(2 ** self.nqbits) * 1 / np.sqrt(2 ** self.nqbits)
        psi = np.kron(psi_datas[0], bell_state).reshape(-1, 1)
        self.rho = psi @ psi.conj().T

        X_trainQ = np.zeros((y_train_scaled.shape[0], len(self.observables)))

        for i in range(1, y_train_scaled.shape[0]):
            def feature_circuit():
                qml.StatePrep(psi_datas[i], wires=0, normalize=True)
                qml.StatePrep(bell_state, wires=list(range(1, self.nqbits + 1)), normalize=True)
                self.unitary_circuit()
                return [qml.expval(obs) for obs in self.observables]

            qnode = qml.QNode(feature_circuit, self.dev)
            results = qnode()
            X_trainQ[i - 1] = results

        X_train = np.hstack([X_trainQ, X_train_scaled])
        self.lm = Ridge(alpha=self.alpha)
        self.lm.fit(X_train, y_train_scaled.reshape(-1, 1))
        y_train_pred = self.lm.predict(X_train)
        y_train_pred = scaler.inverse_transform(y_train_pred).flatten()
        return y_train_pred

    def test(self, X_test_scaled, y_test_scaled, scaler):
        psi_datas_test = np.zeros((y_test_scaled.shape[0] + 1, 2))
        psi_datas_test[0] = self.psi_datas_train[-1]
        for i in range(y_test_scaled.shape[0]):
            val = np.clip(y_test_scaled[i], 0, 1)
            psi_datas_test[i + 1] = np.sqrt(1 - val) * np.array([1, 0]) + np.sqrt(val) * np.array([0, 1])
        self.psi_datas_train = psi_datas_test
        y_test_pred = []

        for i in range(1, y_test_scaled.shape[0] + 1):
            def feature_circuit():
                qml.StatePrep(psi_datas_test[i], wires=0, normalize=True)
                bell_state = np.ones(2 ** self.nqbits) * 1 / np.sqrt(2 ** self.nqbits)
                qml.StatePrep(bell_state, wires=list(range(1, self.nqbits + 1)), normalize=True)
                self.unitary_circuit()
                return [qml.expval(obs) for obs in self.observables]

            qnode = qml.QNode(feature_circuit, self.dev)
            results = qnode()
            X_testQ = np.array(results).reshape(1, -1)
            X_test = np.hstack([X_testQ, X_test_scaled[i - 1].reshape(1, -1)])
            y_hat = self.lm.predict(X_test.reshape(1, -1))
            if y_hat[0, 0] > 1:
                y_hat[0, 0] = 1 - 0.001
            elif y_hat[0, 0] < 0:
                y_hat[0, 0] = 0.001
            y_hat = scaler.inverse_transform(y_hat).flatten()
            y_test_pred.append(y_hat)
        y_test_pred = np.array(y_test_pred)
        return y_test_pred
