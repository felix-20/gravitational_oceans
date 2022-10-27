# https://www.kaggle.com/code/rodrigotenorio/generating-continuous-gravitational-wave-signals

from operator import ge
import os
import sys
from time import time

import numpy as np
import matplotlib.pyplot as plt

import pyfstat

from scipy import stats

import h5py

PATH_TO_DATA_FOLDER = './data/'

writer_kwargs_cw = {
                'tstart': 1238166018,
                'duration': 4 * 30 * 86400,  
                'detectors': 'H1,L1',        
                'sqrtSX': 1e-23,          
                'Tsft': 1800,             
                'SFTWindowType': 'tukey', 
                'SFTWindowBeta': 0.01,
               }

writer_kwargs_no_cw = {
    "label": "single_detector_gaussian_noise",
    "outdir": "PyFstat_example_data",
    "tstart": 1238166018,  # Starting time of the observation [GPS time]
    "duration": 4 * 30 * 86400,  # Duration [seconds] 4 month
    "detectors": "H1,L1",  # Detector to simulate, in this case LIGO Hanford
    "F0": 100.0,  # Central frequency of the band to be generated [Hz]
    "Band": 1.0,  # Frequency band-width around F0 [Hz]
    "sqrtSX": 1e-23,  # Single-sided Amplitude Spectral Density of the noise
    "Tsft": 1800,  # Fourier transform time duration
    "SFTWindowType": "tukey",  # Window function to compute short Fourier transforms
    "SFTWindowBeta": 0.01,  # Parameter associated to the window function
}

generator_cw = pyfstat.AllSkyInjectionParametersGenerator(
    priors={
        'tref': writer_kwargs_cw['tstart'],
        'F0': {'uniform': {'low': 100.0, 'high': 100.1}},
        'F1': lambda: 10**stats.uniform(-12, 4).rvs(),
        'F2': 0,
        'h0': lambda: writer_kwargs_cw['sqrtSX'] / stats.uniform(1, 10).rvs(),
        **pyfstat.injection_parameters.isotropic_amplitude_priors,
    },
)
generator_no_cw = pyfstat.AllSkyInjectionParametersGenerator(
    priors={
        'tref': writer_kwargs_no_cw['tstart'],
        'F0': {'uniform': {'low': 100.0, 'high': 100.1}},
        'F1': lambda: 10**stats.uniform(-12, 4).rvs(),
        'F2': 0,
        'h0': lambda: writer_kwargs_no_cw['sqrtSX'] / stats.uniform(1, 10).rvs(),
        **pyfstat.injection_parameters.isotropic_amplitude_priors,
    },
)

def generate_signal(id: int, should_contain_cw: bool = False):
    writer_kwargs = writer_kwargs_cw if should_contain_cw else writer_kwargs_no_cw
    signal_parameters_generator = generator_cw if should_contain_cw else generator_no_cw
    
    # Draw signal parameters.
    # Noise can be drawn by setting `params['h0'] = 0
    params = signal_parameters_generator.draw()
    writer_kwargs_cw['outdir'] = f'{PATH_TO_DATA_FOLDER}generated/Signal_{id}'
    writer_kwargs_cw['label'] = f'Signal_{id}'
    
    writer = pyfstat.Writer(**writer_kwargs_cw, **params)
    writer.make_data()
    
    # Data can be read as a numpy array using PyFstat
    frequency, timestamps, amplitudes = pyfstat.utils.get_sft_as_arrays(
        writer.sftfilepath
    )

    path_to_hdf5_files = f'{PATH_TO_DATA_FOLDER}cw_hdf5/' if should_contain_cw else f'{PATH_TO_DATA_FOLDER}no_cw_hdf5/'
    if not os.path.isdir(path_to_hdf5_files):
        os.makedirs(path_to_hdf5_files)

    with h5py.File(f'{path_to_hdf5_files}signal{id}.hdf5', 'w') as hd5_file:
        hd5_file.create_dataset('frequency_Hz', data=frequency, dtype='f')
        l1_grp = hd5_file.create_group('L1')
        l1_grp.create_dataset('SFTs', data=amplitudes['L1'], dtype='complex64')
        l1_grp.create_dataset('timestamps_GPS', data=timestamps['L1'], dtype='i')    
        h1_grp = hd5_file.create_group('H1')
        h1_grp.create_dataset('SFTs', data=amplitudes['H1'], dtype='complex64')
        h1_grp.create_dataset('timestamps_GPS', data=timestamps['H1'], dtype='i')


# Generate signals with parameters drawn from a specific population
num_signals = 1


snrs = np.zeros(num_signals)

for i in range(num_signals):
    generate_signal(i)
    generate_signal(i, True)