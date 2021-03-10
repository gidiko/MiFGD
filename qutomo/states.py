
# coding: utf-8

import random
import numpy as np


import qiskit
from qiskit.tools import outer

import methods
import measurements
import projectors


class State:
    def __init__(self, n):
        
        quantum_register   = qiskit.QuantumRegister(n, name='qr')
        classical_register = qiskit.ClassicalRegister(n, name='cr')

        self.quantum_register   = quantum_register
        self.classical_register = classical_register
    
        self.n = n
        
        self.circuit_name = None
        self.circuit      = None
        self.measurement_circuit_names = []
        self.measurement_circuits      = []

        
    def create_circuit(self):
        raise NotImplemented

    
    def execute_circuit(self):
        # XXX not needed?
        pass

    
    def get_state_vector(self):
        if self.circuit is None:
            self.create_circuit()
        
        # XXX add probe?
        backend_engine = qiskit.Aer.get_backend('statevector_simulator')
        job     = qiskit.execute(self.circuit, backend_engine)
        result  = job.result()
        state_vector = result.get_statevector(self.circuit_name)
        return state_vector        

    
    def get_state_matrix(self):
        state_vector = self.get_state_vector()
        state_matrix = outer(state_vector)
        return state_matrix
    
    
    def create_measurement_circuits(self, labels, label_format='big_endian'):
        
        if self.circuit is None:
            self.create_circuit()
        
        qubits = range(self.n)
        
        for label in labels:

            # for aligning to the natural little_endian way of iterating through bits below
            if label_format == 'big_endian':
                effective_label = label[::-1]
            else:
                effective_label = label[:]
            probe_circuit = qiskit.QuantumCircuit(self.quantum_register, 
                                           self.classical_register, 
                                           name=label)
            
            for qubit, letter in zip(*[qubits, effective_label]): 
                probe_circuit.barrier(self.quantum_register[qubit])
                if letter == 'X':
                    probe_circuit.u2(0.,       np.pi, self.quantum_register[qubit])  # H
                elif letter == 'Y':
                    probe_circuit.u2(0., 0.5 * np.pi, self.quantum_register[qubit])  # H.S^*
                probe_circuit.measure(self.quantum_register[qubit], 
                                      self.classical_register[qubit])
                
            measurement_circuit_name = self.make_measurement_circuit_name(self.circuit_name, label)    
            measurement_circuit      = self.circuit + probe_circuit
            
            self.measurement_circuit_names.append(measurement_circuit_name)
            self.measurement_circuits.append(measurement_circuit)
        

    @staticmethod    
    def make_measurement_circuit_name(circuit_name, label):
        name = '%s-%s' % (circuit_name, label)
        return name

    
    def execute_measurement_circuits(self, labels,
                                     backend   = 'qasm_simulator',
                                     num_shots = 100,
                                     label_format='big_endian'):
        
        if self.measurement_circuit_names == []:
            self.create_measurement_circuits(labels, label_format)
        
        circuit_names = self.measurement_circuit_names

        backend_engine = qiskit.Aer.get_backend(backend)
        job = qiskit.execute(self.measurement_circuits, backend_engine, shots=num_shots)
        result = job.result()
        
        data_dict_list = []
        for i, label in enumerate(labels):
            measurement_circuit_name = self.make_measurement_circuit_name(self.circuit_name, label)
            data_dict = {'measurement_circuit_name' : measurement_circuit_name,
                         'circuit_name'             : self.circuit_name,
                         'label'                    : label,
                         'count_dict'               : result.get_counts(i),
                         'backend'                  : backend,
                         'num_shots'                : num_shots}
            data_dict_list.append(data_dict)
        return data_dict_list


class GHZState(State):
    def __init__(self, n):
        State.__init__(self, n)
        self.circuit_name = 'GHZ'

        
    def create_circuit(self):        
        circuit = qiskit.QuantumCircuit(self.quantum_register, 
                                 self.classical_register, 
                                 name=self.circuit_name)

        circuit.h(self.quantum_register[0])
        for i in range(1, self.n):
            circuit.cx(self.quantum_register[0], self.quantum_register[i])   
        
        self.circuit = circuit

        
class HadamardState(State):
    def __init__(self, n):        
        State.__init__(self, n)
        self.circuit_name = 'Hadamard'
        
        
    def create_circuit(self):
        circuit = qiskit.QuantumCircuit(self.quantum_register, 
                                 self.classical_register, 
                                 name=self.circuit_name)
        
        for i in range(self.n):
            circuit.h(self.quantum_register[i])   
        
        self.circuit = circuit

        
class RandomState(State):
    def __init__(self, n, seed=0, depth=40):
        State.__init__(self, n)
        self.circuit_name = 'Random-%d' % (self.n, )

        self.seed  = seed
        self.depth = depth

        
    def create_circuit(self):
        random.seed(a=self.seed)
        circuit = qiskit.QuantumCircuit(self.quantum_register, 
                                 self.classical_register, 
                                 name=self.circuit_name)

        for j in range(self.depth):
            if self.n == 1:
                op_ind = 0
            else:
                op_ind = random.randint(0, 1)
            if op_ind == 0: # U3
                qind = random.randint(0, self.n - 1)
                circuit.u3(random.random(), random.random(), random.random(),
                           self.quantum_register[qind])
            elif op_ind == 1: # CX
                source, target = random.sample(range(self.n), 2)
                circuit.cx(self.quantum_register[source],
                           self.quantum_register[target])
        
        self.circuit = circuit


if __name__ == '__main__':

    ############################################################
    ### Example of creating and running an experiment
    ############################################################

    n = 4
    labels = projectors.generate_random_label_list(20, 4)

    state   = GHZState(4)
    state.create_circuit() 
    data_dict_list = state.execute_measurement_circuits(labels)

            
