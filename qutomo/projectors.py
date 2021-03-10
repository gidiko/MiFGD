
# coding: utf-8

from itertools import product
from functools import reduce
import pickle
import json
import os
import multiprocessing
import glob
import math
import shutil
import subprocess

import scipy.sparse as sparse
import numpy as np
import h5py


# Coordinates of non-zero entries in each of the X, Y, Z Pauli matrices...
ij_dict     = {'I' : [(0, 0), (1, 1)],
               'X' : [(0, 1), (1, 0)], 
               'Y' : [(0, 1), (1, 0)], 
               'Z' : [(0, 0), (1, 1)]}


# ... and the coresponding non-zero entries
values_dict = {'I' : [1.0, 1.0],
               'X' : [1.0, 1.0], 
               'Y' : [-1.j, 1.j], 
               'Z' : [1.0, -1.0]}


# X, Y, Z Pauli matrices
matrix_dict = {'I' : np.array([[1.0, 0.0],
                               [0.0, 1.0]]),
               'X' : np.array([[0.0, 1.0],
                               [1.0, 0.0]]),
               'Y' : np.array([[0.0, -1.j],
                               [1.j, 0.0]]),
               'Z' : np.array([[1.0, 0.0],
                               [0.0, -1.0]])}
# XXX Actually, from matrix_dict we can generate the above 

eig_dict = {'X' : {'0' : np.array([1.0/np.sqrt(2),  1.0 / np.sqrt(2)]),
                   '1' : np.array([1.0/np.sqrt(2), -1.0 / np.sqrt(2)])},

            'Y' : {'0' : np.array([1.0/np.sqrt(2), 1.0j / np.sqrt(2)]),
                   '1' : np.array([1.0/np.sqrt(2), -1.0j / np.sqrt(2)])},

            'Z' : {'0' : np.array([1.0, 0.0]),
                   '1' : np.array([0.0, 1.0])},
            ## XXX I to be removed
            'I' : {'0' : np.array([1.0, 0.0]),
                   '0' : np.array([1.0, 0.0])},
}

# Get a binary string from a binary list
def binarize(x):
    return int(''.join([str(y) for y in x]), 2)


# Generate a random label by sampling and concatenating n times from the symbols list
def generate_random_label(n, symbols=['I', 'X', 'Y', 'Z']):
    num_symbols = len(symbols)
    label = ''.join([symbols[i] for i in np.random.randint(0, num_symbols, size=n)])
    return label


def generate_random_label_list(size,
                               n,
                               symbols=['I', 'X', 'Y', 'Z'], factor=1.0, factor_step=0.1):

        factor      = 1.0
        factor_step = 0.1
        
        factor          = factor + factor_step
        effective_size  = int(size * factor)
        labels = list(set([generate_random_label(n, symbols) for i in range(effective_size)]))
        
        while(len(labels) < size):
            factor          = factor + factor_step
            effective_size  = int(size * factor)
            labels = list(set([generate_random_label(n, symbols) for i in range(effective_size)]))
        labels = labels[:size]
        return labels


# Generate a projector by accumulating the Kronecker products of Pauli matrices
# XXX Used basically to make sure that our fast implementation is correct
def build_projector_naive(label, label_format='big_endian'):
    if label_format == 'little_endian':
        label = label[::-1]
    if len(label) > 6:
        raise Exception('Too big matrix to generate!')
    projector = reduce(lambda acc, item: np.kron(acc, item), [matrix_dict[letter] for letter in label], [1])
    return projector

    
# Generate a projector by computing non-zero coordinates and their values in the matrix, aka the "fast" implementation
def build_projector_fast(label, label_format='big_endian'):
    if label_format == 'little_endian':
        label = label[::-1]

    n = len(label)
    d = 2 ** n
    
    # map's result NOT subscriptable in py3, just tried map() -> list(map()) for py2 to py3
    ij     = [list(map(binarize, y)) for y in 
              [zip(*x) for x in product(*[ij_dict[letter] for letter in label])]] 
    values = [reduce(lambda z, w: z * w, y) for y in 
              [x for x in product(*[values_dict[letter] for letter in label])]]
    ijv    = list(map(lambda x: (x[0][0], x[0][1], x[1]), zip(ij, values))) 
    
    i_coords, j_coords, entries = zip(*ijv)
    projector  = sparse.coo_matrix((entries, (i_coords, j_coords)), 
                                                     shape=(d, d), dtype=np.complex)
    return projector
# XXX Here for redundancy and convenience: has also been added to a separate Projector class


# Choose implementation for the projector
def build_projector(label, label_format='big_endian', fast=True):
    if fast:
        return build_projector_fast(label, label_format)
    return build_projector_naive(label, label_format)


# Multiply a projector Pi of given label (concatenation of Pauli matrix identifiers) with a vector: Pi * x
def matvec(label, x, label_format='big_endian', fast=True):
    projector = sparse.csr_matrix(build_projector(label, label_format, fast))
    y = projector.dot(x)
    return y


# utilities for saving in Projector class
def _hdf5_saver(label, path, lock):
    lock.acquire()
    Projector(label).save(path)
    lock.release()

def _pickle_saver(label, path):
    Projector(label).save(path)

# utilities for saving in PauliStringProjector class    
def _pauli_string_pickle_saver(label, bitvector, path):
    PauliStringProjector(label, bitvector).save(path)

    
def build_projection_vector(label, bitvector):
    n = len(label)
    projection_vector = reduce(lambda acc, item: np.kron(acc, item),
                               [eig_dict[letter][bit]
                                for (letter, bit) in zip(*[label[::-1], bitvector])])
    return projection_vector

def build_projection_vector_dict(label):
    n = len(label)
    d = 2 ** n
    projection_vector_dict = {}
    for idx in range(d):
        bitvector         = format(idx, 'b').zfill(n)
        projection_vector = build_projection_vector(label, bitvector)
        projection_vector_dict[bitvector] = projection_vector
    return projection_vector_dict

    
# A projector as a class, hopefully with convenient methods :)
class Projector:
    # Generate from a label or build from a dictionary represenation
    def __init__(self, arg, label_format='big_endian'):
        if isinstance(arg, str):
            self.label        = arg
            self.label_format = label_format
            self._generate()
        elif isinstance(arg, dict):
            data = arg
            self._build(data)    
    
    def _build(self, data):
        self.label        = data['label']
        self.label_format = data.get('label_format', 'big_endian')
        
        assert   data['num_columns'] == data['num_columns']
        entries  = data['values']
        i_coords = data['row_indices']
        j_coords = data['column_indices']
        d        = data['num_rows']
        dtype    = data['value_type']
        
        matrix  = sparse.coo_matrix((entries, (i_coords, j_coords)), 
                                    shape=(d, d), dtype=np.complex)
        self.matrix     = matrix
        self.csr_matrix = None
    
    # Here, injecting the "fast" implementation logic for our projector
    def _generate(self):
        if self.label_format == 'little_endian':
            label = self.label[::-1]
        else:
            label = self.label[:]
            
        n = len(label)
        d = 2 ** n
        
        # map's result NOT subscriptable in py3, just tried map() -> list(map()) for py2 to py3
        ij     = [list(map(binarize, y)) for y in 
                  [zip(*x) for x in product(*[ij_dict[letter] for letter in label])]]
        values = [reduce(lambda z, w: z * w, y) for y in 
                  [x for x in product(*[values_dict[letter] for letter in label])]]
        ijv    = list(map(lambda x: (x[0][0], x[0][1], x[1]), zip(ij, values)))
    
        i_coords, j_coords, entries = zip(*ijv)
        matrix  = sparse.coo_matrix((entries, (i_coords, j_coords)), 
                                    shape=(d, d), dtype=np.complex)
        self.matrix     = matrix
        self.csr_matrix = None
    
    # matvec'ing with a vector
    def dot(self, x):
        return self.csr().dot(x)
            
    
    # Get a sparse matrix representation of the projector in CSR format, 
    # i.e. ideal for matvec'ing it 
    def csr(self):
        if self.csr_matrix is None:
            self.csr_matrix = sparse.csr_matrix(self.matrix)
        return self.csr_matrix
    
    # Get a sparse matrix representation of the projector in COO format, 
    # i.e. 3 lists with row, col indices and entries
    def coo(self):
        return self.matrix
    
    
    # Get a dict representation of the projector,
    # i.e. ideal for serializing it
    # XXX Currently with Python's pickle format in mind; moving to json format would add to portability
    def dict(self):
        
        data = {
            'values'         : self.matrix.data, 
            'row_indices'    : self.matrix.row, 
            'column_indices' : self.matrix.col, 
            'num_rows'       : self.matrix.shape[0],
            'num_columns'    : self.matrix.shape[1],
            'value_type'     : self.matrix.dtype,
            'label'          : self.label
        }
        return data


    def _pickle_save(self, fpath):
        data = self.dict()
        with open(fpath, 'wb') as f:
            pickle.dump(data, f)


    def _hdf5_save(self, fpath):
        f = h5py.File(fpath, 'a')
        group = f.create_group(self.label)

        data_dict = self.dict()
        for key in ['column_indices', 'row_indices', 'values']:
            dataset = group.create_dataset(key, data = data_dict[key])
        group.attrs['num_columns'] = data_dict['num_columns']
        group.attrs['num_rows']    = data_dict['num_rows']
        f.close()
        
            
    # Save the projector to disk
    def save(self, path):
        if os.path.isdir(path):
            fpath = os.path.join(path, '%s.pickle' % self.label)
            self._pickle_save(fpath)
        elif path.endswith('.hdf5'):
            fpath = path
            self._hdf5_save(fpath)


    @classmethod
    def _pickle_load(cls, fpath):
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        return data


    @classmethod
    def _hdf5_load(cls, fpath, label):

        f = h5py.File(fpath, 'r')
        group = f[label]
        
        data = {'label' : label }
        data['num_rows']    = group.attrs['num_rows']
        data['num_columns'] = group.attrs['num_columns']

        data['column_indices'] = group['column_indices'][:]
        data['row_indices']    = group['column_indices'][:]
        data['values']         = group['values'][:]
        data['value_type']     = data['values'].dtype
        return data

    
    # Load a projector from disk
    @classmethod
    def load(cls, path, label, num_leading_symbols=0):
        if os.path.isdir(path):
            if num_leading_symbols == 0:
                fpath = os.path.join(path, '%s.pickle' % label)
                data  = cls._pickle_load(fpath)
            else:
                fragment_name = label[:num_leading_symbols]
                fpath         = os.path.join(path, fragment_name, '%s.pickle' % label)
                data          = cls._pickle_load(fpath) 
        elif path.endswith('.hdf5'):
            fpath = path
            data  = cls._hdf5_load(fpath, label)
        projector = cls(data)
        return projector


class PauliStringProjector:
    def __init__(self, arg, bitvector=None):
        if isinstance(arg, str):
            self.label     = arg
            self.bitvector = bitvector
            assert(len(self.label) == len(self.bitvector))
            self._generate()
        elif isinstance(arg, dict):
            data = arg
            self._build(data)    
        
    def _build(self, data):
        self.label             = data['label']
        self.bitvector         = data['bitvector']
        self.projection_vector = data['projection_vector']
             
    def _generate(self):
        self.projection_vector = build_projection_vector(self.label,
                                                         self.bitvector)

    def dot(self, x):
        scalar = np.dot(self.projection_vector.conjugate(), x)
        return scalar * self.projection_vector

    def dict(self):        
        data = {
            'label'             : self.label, 
            'bitvector'         : self.bitvector, 
            'projection_vector' : self.projection_vector, 
        }
        return data
    
    def _pickle_save(self, fpath):
        data = self.dict()
        with open(fpath, 'wb') as f:
            pickle.dump(data, f)

    # Save the projector to disk
    def save(self, path):
        if os.path.isdir(path):
            fpath = os.path.join(path, '%s-%s.pickle' % (self.label, self.bitvector))
            self._pickle_save(fpath)
  
    @classmethod
    def _pickle_load(cls, fpath):
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        return data

    # Load a projector from disk
    @classmethod
    def load(cls, path, label, bitvector):
        if os.path.isdir(path):
            fpath = os.path.join(path, '%s-%s.pickle' % (label, bitvector))
            data  = cls._pickle_load(fpath)
        projector = cls(data)
        return projector
  
    @staticmethod    
    def build_projection_vector(label, bitvector):
        n = len(label)
        projection_vector = reduce(lambda acc, item: np.kron(acc, item),
                                   [eig_dict[letter][bit]
                                    for (letter, bit) in zip(*[label[::-1], bitvector])])
        return projection_vector

    

# Trying to represent a collection of projectors that will live in a disk folder
# XXX Moving to a dask-based store, e.g dask's dict allowing concurrent access is one potential next step
# XXX Check how labels end in this
class ProjectorStore:
    def __init__(self, 
                 labels):
        self.labels = labels
        self.size   = len(labels)

    
    # Generate and save the projectors and do it in parallel, 
    # i.e. using all available cores in your system
    def populate(self, path):
        format = 'hdf5'
        if not path.endswith('.hdf5'):
            format = 'pickle'
            if not os.path.exists(path):
                os.mkdir(path)
                
        num_cpus   = multiprocessing.cpu_count()
        num_rounds = 1
        if self.size > num_cpus:
            num_rounds = self.size // num_cpus + 1 

        # XXX lock defies parallelization in generattion for hdf5
        # XXX Consider using MPI-based scheme
        if format == 'hdf5':
            lock = multiprocessing.Lock()
            
        for r in range(num_rounds):
            process_list = []
            for t in range(num_cpus):
                idx = r * num_cpus + t
                if idx == self.size:
                    break
                label = self.labels[idx]
                if format == 'pickle':
                    process = multiprocessing.Process(target=_pickle_saver,
                                                      args=(label, path))
                elif format == 'hdf5':
                    process = multiprocessing.Process(target=_hdf5_saver,
                                                     args = (label, path, lock))
                                
                process.start()
                process_list.append(process)
            # moving join() inside the outer loop to avoid "too many files open" error    
            for p in process_list:
                p.join()
 
    @classmethod
    def load_labels(cls, path):
        if path.endswith('.hdf5'):
            with h5py.File(path, 'r') as f:
                labels = f.keys()
        else:
            try:
                labels = [fname.split('.')[0] for fname in os.listdir(path)]
            except:
                fragment_paths = [os.path.join(path, fragment) for fragment in os.listdir(path)]
                labels = []
                for fragment_path in fragment_paths:
                    fragment_labels = [fname.split('.')[0] for fname in os.listdir(fragment_path)]
                    labels.extend(fragment_labels)
        return labels

                
    # Load projectors previously saved under a disk folder
    @classmethod
    def load(cls, path, labels=None):

        if labels == None:
            labels = cls.load_labels(path)

        # checking if the store is fragmented and compute num_leading_symbols
        names = os.listdir(path)
        aname = names[0]
        apath = os.path.join(path, aname)
        if os.path.isdir(apath):
            num_leading_symbols = len(aname)
        else:
            num_leading_symbols = 0

        # load the store
        projectors = [Projector.load(path, label, num_leading_symbols) for label in labels]
        projector_dict = {}
        for label, projector in zip(*[labels, projectors]):
            projector_dict[label] = projector
        return projector_dict


class PauliStringProjectorStore:
    def __init__(self, 
                 labels,
                 bitvectors_list=None):
        self.labels           = labels
        self.bitvectors_list  = bitvectors_list

        n = len(labels[0])
        d = 2 ** n
        
        if bitvectors_list == None:
            sizes               = [d for label in labels]
            complete_bitvectors = [format(idx, 'b').zfill(n) for idx in range(d)]
        else:
            assert(len(labels) == len(bitvectors_list))
            sizes = [len(bitvectors) for bitvectors in bitvectors_list]
        size = sum(sizes)
                     
        idx2tuple = {}
        base      = 0
        for i, label in enumerate(labels):
            if bitvectors_list == None:
                bitvectors = complete_bitvectors
            else:
                bitvectors = bitvectors_list[i]
            for j in range(sizes[i]):
                idx2tuple[base + j] = label, bitvectors[j] 
            base = base + sizes[i]
        self.size      = size
        self.idx2tuple = idx2tuple
            
    
    # Generate and save the projectors and do it in parallel, 
    # i.e. using all available cores in your system
    def populate(self, path):
        format = 'pickle'
        if not os.path.exists(path):
            os.mkdir(path)
                
        num_cpus   = multiprocessing.cpu_count()
        num_rounds = 1
        if self.size > num_cpus:
            num_rounds = self.size // num_cpus + 1 
            
        for r in range(num_rounds):
            process_list = []
            for t in range(num_cpus):
                idx = r * num_cpus + t
                if idx == self.size:
                    break
                label, bitvector = self.idx2tuple[idx]
                if format == 'pickle':
                    process = multiprocessing.Process(target=_pauli_string_pickle_saver,
                                                      args=(label, bitvector, path))
                process.start()
                process_list.append(process)
            # moving join() inside the outer loop to avoid "too many files open" error    
            for p in process_list:
                p.join()
 
    @classmethod
    def load_labels_bitvectors(cls, path):
        labels     = [fname.split('.')[0].split('-')[0] for fname in os.listdir(path)]
        bitvectors = [fname.split('.')[0].split('-')[1] for fname in os.listdir(path)]
        return labels, bitvectors

                
    # Load projectors previously saved under a disk folder
    @classmethod
    def load(cls, path, labels=None, bitvectors=None):
        _labels, _bitvectors = cls.load_labels_bitvectors(path)

        if labels == None:
            labels = _labels
            if bitvectors == None:
                bitvectors = _bitvectors

        # load the store
        projectors = [PauliStringProjector.load(path, label, bitvector) for label, bitvector in zip(*[labels, bitvectors])]
        projector_dict = {label : {} for label in set(labels)}
        for label, bitvector, projector in zip(*[labels, bitvectors, projectors]):
            projector_dict[label][bitvector] = projector
        return projector_dict

        


def fragment_store(source_path, target_path, num_leading_symbols = 3):
    symbols = ['I', 'X', 'Y', 'Z']
    labels  = ProjectorStore.load_labels(source_path)
    n = len(labels[0])
    if n > num_leading_symbols:
        fragment_names = list(map(lambda x: ''.join(x), product(symbols, repeat=num_leading_symbols)))
    else:
        return
    
    for fragment_name in fragment_names:
        fragment_path = os.path.join(target_path, fragment_name)
        if not os.path.exists(fragment_path):
            os.makedirs(fragment_path)

    for fragment_name in fragment_names:
        source_glob = os.path.join(source_path, '%s*' % fragment_name) 
        target_dir  = os.path.join(target_path, fragment_name)
        command = ' '.join(['cp', source_glob, target_dir])
        subprocess.check_call(command, shell=True)
        print(fragment_name)
    

if __name__ == '__main__':

    ############################################################
    ## Using and testing the code
    ## XXX To move to tests
    ############################################################


    ############################################################
    ### Is our "fast" projector implementation correct? As a function and also as a method in Projector?
    ############################################################
    label = 'XYZY'
    n = len(label)
    d = 2 ** n
    x = np.random.random(d)

    naive_projector_matrix  = build_projector(label, fast=False)
    fast_projector_matrix   = build_projector(label, fast=True).todense()
    class_projector_matrix  = Projector(label).csr().todense()

    
    function_correct = np.allclose(naive_projector_matrix, 
                                   fast_projector_matrix)
    
    method_correct   = np.allclose(naive_projector_matrix,
                                   class_projector_matrix)
    
    correct = function_correct and method_correct
    print(correct)

    
    ############################################################
    ### Let's use a 16-qubit projector
    ############################################################
    label = 'XZYYZXYZZYXYXXYZ'
    n = len(label)
    d = 2 ** n

    projector = Projector(label)
    projector.coo()

    projector.csr()
    projector.dict()

    projector.save('/tmp/projector.pickle')

    loaded_projector = Projector.load('/tmp/projector.pickle')

    x  = np.random.random(d)
    ny = projector.dot(x)
    ny_from_loaded = loaded_projector.dot(x)
    print(np.allclose(y, y_from_loaded))

    
    ############################################################
    ### Testing and using the ProjectorStore
    ############################################################
    size      = 1000
    # how many projectors in the store
    n         = 16
    # how many qubits
    store_dir = '/tmp/projectors'
    # where to host the projector store
    projector_store = ProjectorStore(size, n, store_dir)

    # Note that population happens in parallel: all available cores are used!
    projector_store.populate()


    ############################################################
    ### Some statistics
    ############################################################
    fpaths         = glob.glob(os.path.join(store_dir, '*.pickle'))
    num_projectors = len(fpaths)
    disk_bytesize  = sum([os.path.getsize(fpath) for fpath in fpaths])
    disk_bytesize_per_projector = disk_bytesize / num_projectors

    print('num_projectors          = %d' % num_projectors)
    print('disk_size               = %.2f MB' % (disk_bytesize/ 1.e6,) )
    print('disk_size_per_projector = %.2f MB' % (disk_bytesize_per_projector / 1.e6, ) )
    

    ############################################################
    ### Load the projector store into memory and have a look into the projectors
    ############################################################
    projector_dict = projector_store.load(store_dir)

    labels = list(projector_dict.keys())
    num_markers = 80
    
    for label in labels[:2]:
        projector = projector_dict[label]
        print(projector.dict())
        print('-' * num_markers)
        
        print(projector.csr())
        print('=' * num_markers)

    values_set = set([])

    for label in labels[:10]:
        values_set.update(set(projector_dict[label].coo().data.tolist()))

    values_set


    measurements_dict = {}
    for label, _ in projector_dict.items():
        measurements_dict[label] = np.random.random()

