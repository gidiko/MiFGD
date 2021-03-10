
# coding: utf-8

import multiprocessing
import ctypes
import json
import random
import pickle

import numpy as np
from mpi4py import MPI
from qiskit.quantum_info import state_fidelity

import projectors
import measurements



############################################################
## Parallel projFGD base class
## XXX WIP
############################################################
class Worker:
    def __init__(self,
                 process_idx,
                 num_processes,
                 params_dict):

        projector_store_path     = params_dict.get('projector_store_path', None)
        num_iterations           = params_dict['num_iterations']
        eta                      = params_dict['eta']

        beta                     = params_dict.get('beta', None)
        trace                    = params_dict.get('trace', 1.0)
        target_state             = params_dict.get('target_state', None) 
        convergence_check_period = params_dict.get('convergence_check_period', 10)
        relative_error_tolerance = params_dict.get('relative_error_tolerance', 0.0001)

        parity_flavor            = params_dict.get('parity_flavor', 'effective')

        pauli_correlation_measurements_fpath = params_dict.get('pauli_correlation_measurements_fpath',
                                                               None)
        pauli_basis_measurements_fpath       = params_dict.get('pauli_basis_measurements_fpath',
                                                               None)

        # alternatives: 'pauli_basis', 'pauli_correlation'
        measurements_type                    = params_dict.get('measurements_type', 'pauli_correlation') 

        measurement_store_path   = params_dict.get('measurement_store_path', None)

        tomography_labels            = params_dict.get('tomography_labels', None)
        tomography_bitvectors_list   = params_dict.get('tomography_bitvectors_list', None)
        
        density_matrix           = params_dict.get('density_matrix', None)

        label_format             = params_dict.get('label_format', 'big_endian')

        # XXX deactivated
        # store_load_batch_size    = params_dict.get('store_load_batch_size', 10)
        debug                    = params_dict.get('debug', True)  

        seed                     = params_dict.get('seed', 0)

        mu                       = params_dict.get('mu', 0.0)

        # tomography_labels (and tomography_bitvectors)
        if tomography_labels == None:
            tomography_labels  = measurements.MeasurementStore.load_labels(measurement_store_path)

        n = len(tomography_labels[0])
        d = 2 ** n
        if measurements_type == 'pauli_basis':
            _tomography_labels = tomography_labels.copy()
            tomography_labels     = []
            tomography_bitvectors = []
            if tomography_bitvectors_list is None:
                complete_bitvectors   = [format(idx, 'b').zfill(n) for idx in range(d)]
                tomography_bitvectors = complete_bitvectors * len(_tomography_labels)
                for i, label in enumerate(_tomography_labels):
                    tomography_labels.extend([label] * d)
                
            else:
                for i, label in enumerate(tomography_labels):
                    bitvectors = tomography_bitvectors_list[i]
                    tomography_bitvectors.extend(bitvectors)
                    tomography_labels.extend([label] * len(bitvectors))               
                
        num_tomography_labels = len(tomography_labels) 
        label_list            = split_list(tomography_labels, num_processes)[process_idx]
        num_labels            = len(label_list)
        if measurements_type == 'pauli_basis':
            num_tomography_bitvectors = len(tomography_bitvectors) 
            bitvector_list            = split_list(tomography_bitvectors, num_processes)[process_idx]
            num_bitvectors            = len(bitvector_list)
            

        ################################################################################
        # Loading projectors
        ################################################################################
        if projector_store_path is not None:
            # load projectors in batches
            projector_dict = {}
            start = 0
            # end   = min(num_labels,  store_load_batch_size)
            end = num_labels
            if debug:
                print('Loading projectors')                
            while True:
                if measurements_type == 'pauli_correlation':
                    data_dict = projectors.ProjectorStore.load(projector_store_path, label_list[start:end])
                elif measurements_type == 'pauli_basis':
                    data_dict = projectors.PauliStringProjectorStore.load(projector_store_path,
                                                                          label_list[start:end],
                                                                          bitvector_list[start:end])
                projector_dict.update(data_dict)
                start = end
                # end   = min(num_labels, start  + store_load_batch_size)
                end = num_labels
                if debug:
                    print('%d projectors loaded' % len(projector_dict))
                if start == num_labels:
                    break

            if measurements_type == 'pauli_correlation':
                projector_list = [projector_dict[label] for label in label_list]
            elif measurements_type == 'pauli_basis':
                projector_list = []
                for label, bitvector in zip(*[label_list, bitvector_list]):
                    projector_list.append(projector_dict[label][bitvector])
                #projector_list = [projector_dict[label][bitvector] for (label, bitvector) in
                #zip(*[label_list, bitvector_list])]
            del projector_dict
        else:
            if debug:
                print('Computing projectors')
            projector_list   = [projectors.build_projector_naive(label, label_format) for label in label_list]
            projector_list   = [projectors.build_projection_vector(label, bitvector) for label, bitvector
                                in zip(*[label_list, bitvector_list])]
        if debug:
            print('Projectors ready to compute with')

        ################################################################################
        # Loading measurements
        ################################################################################
        if pauli_correlation_measurements_fpath is not None:
            if debug:
                print('Loading Pauli correlation measurements')                
            if pauli_correlation_measurements_fpath.endswith('.json'):
                with open(pauli_correlation_measurements_fpath, 'r') as f:
                    data_dict = json.load(f)
            measurement_list = [data_dict[label] for label in label_list]
            del data_dict

        if pauli_basis_measurements_fpath is not None:
            if debug:
                print('Loading Pauli basis measurements')                
            if pauli_basis_measurements_fpath.endswith('.json'):
                with open(pauli_basis_measurements_fpath, 'r') as f:
                    data_dict = json.load(f)
            measurement_list = [data_dict[label][bitvector] for label, bitvector
                                in zip(*[label_list, bitvector_list])]
            del data_dict
        
        elif measurement_store_path is not None:
            # load measurements in batches
            measurement_dict = {}
            start = 0
            # end   = min(num_labels,  store_load_batch_size)
            end = num_labels
            if debug:
                print('Loading measurements')                
            while True:
                data_dict = measurements.MeasurementStore.load(measurement_store_path, label_list[start:end])
                measurement_dict.update(data_dict)
                start = end
                # end   = min(num_labels, start + store_load_batch_size)
                end = num_labels
                if debug:
                    print('%d measurements loaded' % len(measurement_dict))
                if start == num_labels:
                    break
            
            count_dict_list = [measurement_dict[label] for label in label_list]
            measurement_object_list = [measurements.Measurement(label, count_dict) for (label, count_dict) in 
                                       zip(*[label_list, count_dict_list])]
                
            if measurements_type == 'pauli_correlation':
                measurement_list = [measurement_object.get_pauli_correlation_measurement(beta, parity_flavor)[label] for
                                    (label, measurement_object) in zip(*[label_list,
                                                                         measurement_object_list])]
            elif measurements_type == 'pauli_basis':
                measurement_list = [measurement_object.get_pauli_basis_measurement(beta)[label][bitvector] for
                                    (label, bitvector, measurement_object) in zip(*[label_list,
                                                                                    bitvector_list,
                                                                                    measurement_object_list])]
            del count_dict_list, measurement_object_list
        
        elif density_matrix is not None:
            if debug:
                print('Computing measurements')
            if measurements_type == 'pauli_correlation':
                                measurement_list = get_measurements(density_matrix, projector_list)
        if debug:
            print('Measurements ready to compute with')

        self.num_iterations = num_iterations
        self.eta            = eta
        
        self.trace                     = trace
        self.target_state              = target_state
        self.convergence_check_period  = convergence_check_period
        self.relative_error_tolerance  = relative_error_tolerance
        self.seed                      = seed
        self.mu                        = mu
        
        self.label_list     = label_list
        self.bitvector_list = bitvector_list
        self.measurement_list = measurement_list
        self.projector_list   = projector_list

        self.num_tomography_labels = num_tomography_labels
        self.num_labels            = num_labels

        self.num_bitvectors        = num_bitvectors
        
        n = len(label_list[0])
        d = 2 ** n
        
        self.n            = n
        self.num_elements = d
        
        self.process_idx   = process_idx
        self.num_processes = num_processes

        self.error_list                 = []
        self.relative_error_list        = []

        self.target_error_list          = []
        self.target_relative_error_list = []
        
        self.iteration                  = 0
        self.converged                  = False
        self.convergence_iteration      = 0

        self.fidelity_list              = []

        # additional attributes useful for experimentation
        self.circuit_name = params_dict.get('circuit_name', '')
        self.backend      = params_dict.get('backend', 'local_qiskit_simulator')
        self.num_shots    = params_dict.get('num_shots', 8192)
        
        self.complete_measurements_percentage = params_dict.get('complete_measurements_percentage', 40)
            
        
    @staticmethod
    def single_projection_diff(projector, measurement, state):
        projection = projector.dot(state)
        trace      = np.dot(projection, state.conj())
        diff       = (trace - measurement) * projection
        return diff
    
    
    def projection_diff(self):
        state_diff = np.zeros(self.num_elements, dtype=np.complex)
        for projector, measurement in zip(*[self.projector_list, self.measurement_list]):
            state_diff += self.single_projection_diff(projector, measurement, self.momentum_state)
        return state_diff
    

    def initialize(self):

        real_state = np.random.RandomState(self.seed).randn(self.num_elements)
        imag_state = np.random.RandomState(self.seed).randn(self.num_elements)
        state      = real_state + 1.0j * imag_state
        
        state      = 1.0 / np.sqrt(self.num_tomography_labels) * state
        state_norm = np.linalg.norm(state)
        state      = state / state_norm

        self.state = state
        self.momentum_state = state[:]
                
        
    def compute(self):
        self.initialize()
        for self.iteration in range(self.num_iterations):
            if not self.converged:
                self.step()
        if self.convergence_iteration == 0:
            self.convergence_iteration = self.iteration

            
    def convergence_check(self):
                
        if self.process_idx == 0 and self.iteration % self.convergence_check_period == 0:
            # compute relative error
            numerator            = density_matrix_diff_norm(self.state, self.previous_state)
            denominator          = density_matrix_norm(self.state)

            error                = numerator  
            relative_error       = numerator / denominator

            self.error_list.append(error)
            self.relative_error_list.append(relative_error)

            if relative_error <= self.relative_error_tolerance:
                self.converged = True
                self.convergence_iteration = self.iteration
                
            if self.target_state is not None:
                # compute target relative error
                numerator             = density_matrix_diff_norm(self.state, self.target_state)
                denominator           = density_matrix_norm(self.target_state)

                target_error          = numerator
                target_relative_error = numerator / denominator

                self.target_error_list.append(target_error)
                self.target_relative_error_list.append(target_relative_error)

                ### check: if NOT validate=False then this fails
                fidelity = state_fidelity(self.target_state, self.state, validate=False)
                self.fidelity_list.append(fidelity)

    def dump(self, fpath):
        workspace = vars(self)
        with open(fpath, 'wb') as f:
            pickle.dump(workspace, f)

            
############################################################
## Basic sequential projFGD
## XXX WIP
############################################################
class BasicWorker(Worker):
    def __init__(self,
                 params_dict):
        
        process_idx   = 0
        num_processes = 1

        Worker.__init__(self,
                        process_idx,
                        num_processes,
                        params_dict)
                

    def step(self):
        self.previous_state          = self.state[:]
        self.previous_momentum_state = self.momentum_state[:]
        
        state_diff = self.projection_diff()

        state               = self.momentum_state - self.eta * state_diff
        self.state          = clip_normalize(state, self.trace)
        self.momentum_state = self.state + self.mu * (self.state - self.previous_state)

        if self.iteration % 100 == 0:
            print(self.iteration)

        self.convergence_check()

        
############################################################
## Parallel projFGD using MPI
## XXX WIP
############################################################
class MPIWorker(Worker):
    def __init__(self,
                 params_dict):
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        process_idx   = rank
        num_processes = size

        Worker.__init__(self,
                        process_idx,
                        num_processes,
                        params_dict)
        
        self.comm = comm
        self.size = size
        self.rank = rank
        

    def step(self):
        self.comm.barrier()
        self.previous_state = self.state[:]
        
        local_state_diff  = self.projection_diff()
        global_state_diff = self.comm.allreduce(local_state_diff, MPI.SUM)
        state      = self.state - self.eta * global_state_diff
        self.state = clip_normalize(state, self.trace)
        # print([self.process_idx, self.iteration, np.linalg.norm(global_state_diff)])
        # print([self.process_idx, self.iteration, np.linalg.norm(self.state)])
            
        self.convergence_check()
        # if self.process_idx == 0:
        # print(self.iteration)
        
        # remote_state = None
        # if self.process_idx == 0:
        #     self.comm.send(self.state, 1)
        # elif self.process_idx == 1:
        #     remote_state = self.comm.recv(remote_state, 0)
        # if self.process_idx == 1:
        #     # print([0, remote_state[0:4]])
        #     # print([1, self.state[0:4]])
        #     print(['*', np.linalg.norm(self.state - remote_state)])



        
############################################################
## Parallel projFGD using Python's builtin
## multiprocessing module
## XXX WIP
############################################################
class MultiprocessingWorker(Worker):
    def __init__(self, 
                 process_idx, num_processes, 
                 shared_state_diff_real,
                 shared_state_diff_imag,
                 shared_state_diff_lock,
                 iteration_barrier,
                 params_dict):

        Worker.__init__(self,
                        process_idx,
                        num_processes,
                        params_dict)
        
        self.shared_state_diff_real = shared_state_diff_real
        self.shared_state_diff_imag = shared_state_diff_imag
        self.shared_state_diff_lock = shared_state_diff_lock
        
        self.iteration_barrier = iteration_barrier
    
    ## XXX To revise
    def step(self):
        self.previous_state = self.state[:]
        
        local_state_diff  = self.projection_diff()

        # accumulate state diffs
        self.shared_state_diff_lock.acquire()
        self.shared_state_diff_real[:] = self.shared_state_diff_real[:] + local_state_diff.real
        self.shared_state_diff_imag[:] = self.shared_state_diff_imag[:] + local_state_diff.imag
        self.shared_state_diff_lock.release()
        self.iteration_barrier.wait()

        # read accumulated state diffs
        shared_state_diff      = np.zeros(self.num_elements, dtype=np.complex)
        
        self.shared_state_diff_lock.acquire()
        shared_state_diff.real = self.shared_state_diff_real[:]
        shared_state_diff.imag = self.shared_state_diff_imag[:]
        self.shared_state_diff_lock.release()
        self.iteration_barrier.wait()

        # update state
        state      = self.state - self.eta * shared_state_diff
        self.state = clip_normalize(state, self.trace)

        self.convergence_check()


############################################################
## Utility functions
## XXX To modularize/package
############################################################

def density_matrix_norm(state):
    conj_state = state.conj()
    norm = np.sqrt(sum([v**2 for v in [np.linalg.norm(state * item)
                                       for item in conj_state]]))
    return norm
    

def density_matrix_diff_norm(xstate, ystate):
    conj_xstate = xstate.conj()
    conj_ystate = ystate.conj()
    
    norm = np.sqrt(sum([v**2 for v in [np.linalg.norm(xstate * xitem - ystate * yitem)
                                       for xitem, yitem in zip(*[conj_xstate, conj_ystate])]]))
    return norm
    

def clip_normalize(vector, threshold):
    norm = np.linalg.norm(vector)
    if norm  > threshold:
        vector = (threshold / norm) * vector
    return vector
    

def generate_random_experiment(n=8,
                               num_shots = 1024,
                               num_labels = 100,
                               measurement_store_path = '/develop/measurements',
                               projector_store_path = '/develop/projectors'):
    
    # labels
    labels = projectors.generate_random_label_list(num_labels, n)
    
    # measurements
    measurement_dict = {}
    for label in labels:
        count_dict = measurements.generate_random_measurement_counts(n, num_shots)
        measurement_dict[label] = count_dict
    measurement_store = measurements.MeasurementStore(measurement_dict)
    measurement_store.save(measurement_store_path)
    
    # projectors
    projector_store = projectors.ProjectorStore(labels)
    projector_store.populate(projector_store_path)
    

def get_measurements(matrix, projector_list):
    measurement_list = []
    for projector in projector_list:
        trace = np.trace(projector.dot(matrix))
        measurement_list.append(trace)
    return measurement_list

        
def split_list(x, num_parts):
    n = len(x)
    size = n // num_parts
    parts = [x[i * size: (i+1) * size] for i in range(num_parts - 1 )]
    parts.append(x[(num_parts - 1) * size:])
    return parts


def single_projection_diff(projector, measurement, state):
    projection = projector.dot(state)
    trace      = np.dot(state, projection)
    diff       = (trace - measurement) * projection
    return diff


def projection_diff(projector_list, measurement_list, state):
    d = state.shape[0]
    state_diff = np.zeros(d, dtype=np.complex)
    for projector, measurement in zip(*[projector_list, measurement_list]):
        state_diff += single_projection_diff(projector, measurement, state)
    return state_diff


def compute(projector_list,
            measurement_list,
            shared_state, 
            eta,
            iteration_barrier,
            num_iterations):
    
    for iteration in range(num_iterations):
        with shared_state.get_lock():
            state = np.asarray(shared_state.get_obj(), dtype=np.complex)
        diff = projection_diff(projector_list, measurement_list, state)
        with shared_state.get_lock():
            state = np.asarray(shared_state.get_obj(), dtype=np.complex)
            state[:] = state + eta * diff
        iteration_barrier.wait()

def update(shared_vector,
           value, 
           iteration_barrier,
           num_iterations, 
           num_elements):
    for iteration in range(num_iterations):
        v = np.random.random(num_elements)
        with shared_vector.get_lock():
            shared_data = np.asarray(shared_vector.get_obj())
            shared_data[:] = shared_data + v
        iteration_barrier.wait()

        
if __name__ == '__main__':

    ############################################################
    ## Sample parallel run using MPI
    ## XXX WIP
    ## The snippet below in a worker.py script and launch as:
    ## mpirun -np <num-processes> python worker.py
    ############################################################

    params_dict = {'measurement_store_path' : '/develop/measurements',
               'projector_store_path'   : '/develop/projectors',
               'num_iterations'         : 4,
               'eta'                    : 0.01}

    worker = MPIWorker(params_dict)
    worker.compute()

    ############################################################
    ## Sample parallel run using MultiprocessingWorker
    ## XXX WIP
    ############################################################

    # setup arguments to pass to the workers
    params_dict = {'measurement_store_path' : '/develop/measurements',
                   'projector_store_path'   : '/develop/projectors',
                   'num_iterations'         : 4,
                   'eta'                    : 0.01}

    n             = 8
    num_processes = 10

    d = 2 ** n
    shared_state  = multiprocessing.Array(ctypes.c_double, np.ones(d))
    iteration_barrier = multiprocessing.Barrier(num_processes)


    # launch parallel computation
    process_list = []
    for process_idx in range(num_processes):
        worker = MultiprocessingWorker(process_idx, num_processes, 
                                       shared_state, iteration_barrier, 
                                       params_dict)
        process = multiprocessing.Process(target=worker.compute, args=())
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()

    
    
    ############################################################
    ## Sample parallel run using multiprocessing facilities
    ## XXX To move to separate class
    ############################################################

    n = 16
    d = 2 ** n

    num_iterations = 10
    num_processes  = 10
    num_elements   = d
    eta            = 0.01

    num_parts       = num_processes
    labels          = list(projector_dict.keys()) 
    splitted_labels = split_list(labels, num_parts)
    
    splitted_projectors  = [[projector_dict[x] for x in labels] for labels in splitted_labels]
    splitted_mesurements = [[measurements_dict[x] for x in labels] for labels in splitted_labels]

    state = np.ones(d, dtype=np.complex)
    projection_diff(splitted_projectors[0], splitted_mesurements[0], state)

    shared_state  = multiprocessing.Array(ctypes.c_double, np.zeros(d))
    iteration_barrier = multiprocessing.Barrier(num_processes)

    shared_state  = multiprocessing.Array(ctypes.c_double, np.zeros(d))
    iteration_barrier = multiprocessing.Barrier(num_processes)

    process_list = []
    for p in range(num_processes):
        projector_list   = splitted_projectors[p]
        measurement_list = splitted_mesurements[p]
        process = multiprocessing.Process(target=compute, args=(projector_list,
                                                                measurement_list,
                                                                shared_state,
                                                                eta,
                                                                iteration_barrier,
                                                                num_iterations))
        process.start()
        process_list.append(process)
        for process in process_list:
            process.join()


    process_list = []
    for p in range(num_processes):
        value = p # just for testing purposes
        process = multiprocessing.Process(target=update, args=(shared_vector,
                                                               value,
                                                               iteration_barrier,
                                                               num_iterations,
                                                               num_elements))
        process.start()
        process_list.append(process)
        for process in process_list:
            process.join()

    np.asarray(shared_vector.get_obj())

