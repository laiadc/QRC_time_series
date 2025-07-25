import itertools
import random
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
from numpy.lib.scimath import sqrt as csqrt
from qiskit import Aer, QuantumCircuit
from qiskit.circuit import Parameter, QuantumRegister
from qiskit.circuit.library import Diagonal
from qiskit.compiler import transpile
from qiskit.extensions import UnitaryGate
from qiskit.opflow import *
from qiskit.opflow import I, StateFn, X, Y, Z
from qiskit.opflow.evolutions import PauliTrotterEvolution, Suzuki
from qiskit.quantum_info import DensityMatrix, Pauli, Statevector
from qiskit.quantum_info.states.utils import partial_trace
from scipy import stats
from scipy.sparse import random
from scipy.sparse.linalg import eigs
from scipy.stats import unitary_group
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (mean_absolute_error, median_absolute_error,
                             r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class QuantumRC():
    def __init__(self, nqbits=6, num_gates =20, gates_set = ['CNOT', 'T', 'H'], alpha=1e-1, t=10):
        self.nqbits = nqbits
        self.num_gates = num_gates
        self.gates_set = gates_set
        self.lm = None
        self.rho=None
        self.alpha = alpha
        self.t=t
        if gates_set=='Ising':
            self.U = self.get_unitary_ising()
        else:
            self.U = self.get_unitary()
        self.observables = self.get_observables()
        
    def get_unitary_ising(self):
        qubit_idx = list(range(self.nqbits+1))
        qubit_pairs = list(itertools.combinations(qubit_idx, 2))
        Js = 1 # np.random.normal(0.75, 0.1)
        h_over_Js = 0.1
        h = Js*h_over_Js
        pauli_op = 0
        name_gate=''
        for i in range(self.nqbits+1):
            name_gate+= 'I' 
        for pair in qubit_pairs:
            name = name_gate[:pair[0]] + 'Z' + name_gate[(pair[0]+1):pair[1]] + 'Z' + name_gate[(pair[1]+1):]
            #coef = np.random.uniform(0.5-self.Js, 0.5+self.Js)
            coef = np.random.uniform(-Js/2,Js/2)
            pauli_op += coef*PauliOp(Pauli(name))

        for qubit in qubit_idx:
            name = name_gate[:qubit] + 'X' + name_gate[(qubit+1):]
            coef = h #np.random.uniform(0.5-self.hs, 0.5+self.hs)
            pauli_op += coef*PauliOp(Pauli(name))

        evo_time = Parameter('θ')
        evolution_op = (evo_time*pauli_op).exp_i()
        trotterized_op = PauliTrotterEvolution(trotter_mode=Suzuki(order=2, reps=1)).convert(evolution_op)
        bound = trotterized_op.bind_parameters({evo_time: self.t})

        qc_ham = bound.to_circuit()
        # Get unitary
        backend = Aer.get_backend('unitary_simulator')
        job = backend.run(transpile(qc_ham, backend))
        U = job.result().get_unitary(qc_ham)
        return U

    def get_unitary(self):
        # Get Unitary
        qc = QuantumCircuit(self.nqbits + 1)
        gate_idx = list(range(len(self.gates_set)))
        qubit_idx = list(range(self.nqbits + 1))
        # Apply random gates to random qubits
        for i in range(self.num_gates):
            # Select random gate
            idx = sample(gate_idx,1)[0] 
            gate = self.gates_set[idx]
            if gate=='CNOT': # For 2-qubit gates
                # Select qubit 1 and 2 (different qubits)
                qbit1 = sample(qubit_idx,1)[0]
                qubit_idx2 = qubit_idx.copy()
                qubit_idx2.remove(qbit1)
                qbit2 = sample(qubit_idx2,1)[0]
                # Apply gate to qubits
                qc.cx(qbit1, qbit2) 
            else: # For 1-qubit gates
                # Select qubit
                qbit = sample(qubit_idx,1)[0]
                if gate=='X':# Apply gate
                    qc.x(qbit) 
                if gate=='S':
                    qc.s(qbit) 
                if gate=='H':
                    qc.h(qbit) 
                if gate=='T':
                    qc.t(qbit) 
        # Get unitary
        backend = Aer.get_backend('unitary_simulator')
        job = backend.run(transpile(qc, backend))
        U = job.result().get_unitary(qc)
        return U
    
    def get_observables(self):
        # Define measurements
        observables = []
        name_gate=''
        for i in range(self.nqbits+1):
            name_gate+= 'I' 
        for i in range(self.nqbits+1):
            # X
            op_nameX = name_gate[:i] + 'X' + name_gate[(i+1):]
            obs = PauliOp(Pauli(op_nameX))
            observables.append(obs)
            # Z
            op_nameZ = name_gate[:i] + 'Z' + name_gate[(i+1):]
            obs = PauliOp(Pauli(op_nameZ))
            observables.append(obs)
        return observables
    
    def train(self, X_train_scaled, y_train_scaled, scaler):   
        # Initialization: Get states |psi>
        psi_datas = np.zeros((y_train_scaled.shape[0]+1, 2))
        psi_datas[0] = np.array([1,0])
        for i in range(y_train_scaled.shape[0]):
            psi_datas[i+1] = np.sqrt(1-y_train_scaled[i])*np.array([1,0]) + np.sqrt(y_train_scaled[i])*np.array([0,1])
        self.psi_datas_train = psi_datas    
        # Initial density matrix: |psi0> \otimes |Bell>
        bell_state = np.ones(2**self.nqbits)*1/np.sqrt(2**self.nqbits)
        psi = np.kron(psi_datas[0], bell_state).reshape(-1,1)
        self.rho = psi @ psi.conj().T # Construct density matrix
        
        # Vector of quanutm features
        X_trainQ = np.zeros((y_train_scaled.shape[0], len(self.observables)))    
        ## TRAINING!!!!
        for i in range(1,y_train_scaled.shape[0]):
            # Apply unitary
            rho_total = self.U.conj().T @ self.rho @ self.U
            rho_partial = partial_trace(rho_total, [0]).data # Get partial trace
            psi = psi_datas[i].reshape(-1,1)
            psi_new = psi @ psi.conj().T # Get input state for the following step
            rho_new =  np.kron(psi_new, rho_partial)
            self.rho = rho_new.copy()
            # Compute measurements
            results=[]
            for obs in self.observables:
                obs_mat = obs.to_spmatrix()
                expect = DensityMatrix(obs_mat @ rho_total).trace().real
                results.append(expect)
            X_trainQ[i-1] = results   
        # Add regressor variables
        X_train = np.hstack([X_trainQ, X_train_scaled])
        # Fit Ridge regression
        self.lm = Ridge(alpha=self.alpha)
        self.lm.fit(X_train, y_train_scaled.reshape(-1,1))
        y_train_pred = self.lm.predict(X_train)
        y_train_pred = scaler.inverse_transform(y_train_pred).flatten()
        
        return y_train_pred
    
    def test(self,X_test_scaled,y_test_scaled,scaler):
        psi_datas_test = np.zeros((y_test_scaled.shape[0]+1, 2))
        psi_datas_test[0] = self.psi_datas_train[-1]
        for i in range(y_test_scaled.shape[0]):
            psi_datas_test[i+1] = np.sqrt(1-y_test_scaled[i])*np.array([1,0]) + np.sqrt(y_test_scaled[i])*np.array([0,1])
        self.psi_datas_train = psi_datas_test
        y_test_pred = []    
        ## Making the predictions!
        for i in range(1,y_test_scaled.shape[0]+1):
            # Apply unitary
            rho_total = self.U.conj().T @ self.rho @ self.U
            rho_partial = partial_trace(rho_total, [0]).data# Get partial trace
            psi = psi_datas_test[i].reshape(-1,1)
            psi_new = psi @ psi.conj().T # Get input state for the following step
            rho_new =  np.kron(psi_new, rho_partial)
            self.rho = rho_new.copy()
            # Compute measrements
            results=[]
            for obs in self.observables:
                obs_mat = obs.to_spmatrix()
                expect = DensityMatrix(obs_mat @ rho_total).trace().real
                results.append(expect)
            X_testQ = np.array(results).reshape(1,-1) 
            X_test = np.hstack([X_testQ, X_test_scaled[i-1].reshape(1,-1)])
            y_hat = self.lm.predict(X_test.reshape(1,-1)) #Predict with ridge regression
            if y_hat[0,0]>1:
                y_hat[0,0]=1-0.001
            elif y_hat[0,0]<0:
                y_hat[0,0] =0.001
            y_hat = scaler.inverse_transform(y_hat).flatten()
            y_test_pred.append(y_hat)
        y_test_pred = np.array(y_test_pred) 
        return y_test_pred