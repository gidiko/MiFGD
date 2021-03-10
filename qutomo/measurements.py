
# coding: utf-8

import os
import pickle
import subprocess
from itertools import product
import json

import numpy as np
import h5py


def generate_random_measurement_counts(n, num_shots=1024):
    d = 2 ** n
    counts = np.random.random_sample(d)
    counts = counts / sum(counts)
    counts = [int(count * num_shots) for count in counts]
    diff   =  num_shots - sum(counts)
    idx    = np.random.randint(d)
    counts[idx] += diff
    
    count_dict = {}
    for idx, count in enumerate(counts):
        key = bin(idx)[2:]
        key = key.zfill(n)
        count_dict[key] = count
    return count_dict


def expand_convert_measurements(data):
    expanded_data    = {}
    for label, value in data.items():
        indices = list(filter(lambda i: label[i] == 'Z', range(len(label))))
        num = len(indices)
        tuples = list(itertools.product(['I', 'Z'], repeat=num))
        value_labels = []
        for t in tuples:
            item = list(label)
            for i, idx in enumerate(indices):
                item[idx] = t[i]
            value_label = ''.join(item)
            value_labels.append(value_label)
        for value_label in value_labels:
            expanded_data[value_label] = value.copy()
    
    measurement_dict = {}
    for key, counts in expanded_data.items():
        count_dict = {}
        for binary_symbols, binary_counts in counts.items():
            count_dict[binary_symbols] = int(round(binary_counts))
        new_key = key[::-1]
        measurement_dict[new_key] = count_dict
    return measurement_dict


class Measurement:
    def __init__(self, label, count_dict):
        
        akey = list(count_dict.keys())[0]
        assert len(label) == len(akey)

        self.label            = label
        self.count_dict       = self.zfill(count_dict)
        self.num_shots        = sum(count_dict.values())

    @staticmethod
    def zfill(count_dict):
        n = len(list(count_dict.keys())[0])
        d = 2 ** n
        result = {}
        for idx in range(d):
            key = bin(idx)[2:]
            key = key.zfill(n)
            result[key] = count_dict.get(key, 0)
        return result

    @staticmethod
    def naive_parity(key):
        return key.count('1')

    
    def effective_parity(self, key):
        indices    = [i for i, symbol in enumerate(self.label) if symbol == 'I']
        digit_list = list(key)
        for i in indices:
            digit_list[i] = '0'
        effective_key = ''.join(digit_list)
        return effective_key.count('1')

    def parity(self, key, parity_flavor='effective'):
        if parity_flavor == 'effective':
            return self.effective_parity(key)
        else:
            return self.naive_parity(key)
            
    
    def get_pauli_correlation_measurement(self, beta=None, parity_flavor='effective'):
        if beta == None:
            beta = 0.50922
        num_shots          = 1.0 * self.num_shots
        num_items          = len(self.count_dict)
        frequencies        = {k : (v + beta) / (num_shots + num_items * beta) for k, v in self.count_dict.items()}
        parity_frequencies = {k : (-1) ** self.parity(k, parity_flavor) * v for k, v in frequencies.items()}
        correlation        = sum(parity_frequencies.values())
        data = {self.label : correlation}
        return data
        
    def get_pauli_basis_measurement(self, beta=None):
        if beta == None:
            beta = 0.50922
        num_shots   = 1.0 * self.num_shots
        num_items   = len(self.count_dict)
        frequencies = {k : (v + beta) / (num_shots + num_items * beta) for k, v in self.count_dict.items()}
        data = {self.label: frequencies}
        return data

    def dict(self):
        data = {'label' : self.label,
                'count_dict' : self.count_dict}
        return data
            

    def _pickle_save(self, fpath):
        with open(fpath, 'wb') as f:
            pickle.dump(self.count_dict, f)


    def _hdf5_save(self, fpath):
        f = h5py.File(fpath, 'a')
        group = f.create_group(self.label)
        
        items = [[key, value] for key, value in self.count_dict.items()]
        keys   = np.array([item[0] for item in items], dtype='S')
        values = np.array([item[1] for item in items], dtype='int32')
        
        dataset = group.create_dataset('keys',   data = keys)
        dataset = group.create_dataset('values', data = values) 
        f.close()
        
    
    # Save the measurement to disk
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
        keys   = group['keys'][:]
        values = group['values'][:]

        data = {k: v for k, v in  zip(*[keys, values])}
        return data

    
    # Load a measurement from disk
    @classmethod
    def load(cls, path, label, num_leading_symbols=0):
        if os.path.isdir(path):
            if num_leading_symbols == 0:
                fpath = os.path.join(path, '%s.pickle' % label)
                count_dict  = cls._pickle_load(fpath)
            else:
                fragment_name = label[:num_leading_symbols]
                fpath         = os.path.join(path, fragment_name, '%s.pickle' % label)
                count_dict    = cls._pickle_load(fpath) 
        elif path.endswith('.hdf5'):
            fpath = path
            count_dict = cls._hdf5_load(fpath, label)
        measurement = cls(label, count_dict)
        return measurement

    

class MeasurementStore:
    def __init__(self,
                 measurement_dict):
        self.measurement_dict = measurement_dict
        
        self.labels = list(measurement_dict.keys())
        self.size   = len(self.labels)

    
    def save(self, path):
        format = 'hdf5'
        if not path.endswith('.hdf5'):
            format = 'pickle'
            if not os.path.exists(path):
                os.mkdir(path)
        for label, count_dict in self.measurement_dict.items():
            Measurement(label, count_dict).save(path)

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

    
    # Load measurements previously saved under a disk folder
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
        measurements = [Measurement.load(path, label, num_leading_symbols) for label in labels]
        measurement_dict = {}
        for label, measurement in zip(*[labels, measurements]):
            measurement_dict[label] = measurement.count_dict
        return measurement_dict

    @classmethod
    def _pauli_correlation_dict(cls, path, export_fpath,
                                label_list, measurement_object_list,
                                beta, parity_flavor):
        pauli_correlation_dict = {label : measurement_object.get_pauli_correlation_measurement(beta, parity_flavor)[label]
                                  for (label, measurement_object) in zip(*[label_list, measurement_object_list])}
        return pauli_correlation_dict


    @classmethod
    def _pauli_basis_dict(cls, path, export_fpath,
                          label_list, measurement_object_list,
                          beta):
        pauli_basis_dict = {label : measurement_object.get_pauli_basis_measurement(beta)[label]
                                  for (label, measurement_object) in zip(*[label_list, measurement_object_list])}
        return pauli_basis_dict

    
    @classmethod
    def export(cls, path, export_fpath, labels=None, export_type='pauli_correlation',
               export_params={'beta'                  : None,
                              'parity_flavor'         : 'effective',
                              'store_load_batch_size' : 1000,
                              'debug'                 : True}):

        beta                  = export_params['beta']
        parity_flavor         = export_params['parity_flavor']
        store_load_batch_size = export_params['store_load_batch_size']
        debug                 = export_params['debug']

        if labels == None:
            label_list = cls.load_labels(path)

        num_labels = len(label_list)

        # load measurements in batches
        measurement_dict = {}
        start = 0
        end   = min(num_labels,  store_load_batch_size)
        if debug:
            print('Loading measurements')                
        while True:
            data_dict = MeasurementStore.load(path, label_list[start:end])
            measurement_dict.update(data_dict)
            start = end
            end   = min(num_labels, start + store_load_batch_size)
            if debug:
                print('%d measurements loaded' % len(measurement_dict))
            if start == num_labels:
                    break
            
        count_dict_list = [measurement_dict[label] for label in label_list]
        measurement_object_list = [Measurement(label, count_dict) for (label, count_dict) in 
                                   zip(*[label_list, count_dict_list])]
        if debug:
            print('Converting measurements')
        if export_type   == 'pauli_correlation':
            export_dict = cls._pauli_correlation_dict(path, export_fpath,
                                                      label_list, measurement_object_list,
                                                      beta, parity_flavor)
        elif export_type == 'pauli_basis':
            export_dict = cls._pauli_basis_dict(path, export_fpath,
                                                label_list, measurement_object_list,
                                                beta)
        if export_fpath.endswith('.json'):
            with open(export_fpath, 'w') as f:
                json.dump(export_dict, f, indent=4, sort_keys=True)

        if debug:
            print('Measurements exported to file')



def fragment_store(source_path, target_path, num_leading_symbols = 3):
    symbols = ['I', 'X', 'Y', 'Z']
    labels  = MeasurementStore.load_labels(source_path)
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
