# https://www.kaggle.com/code/rodrigotenorio/generating-continuous-gravitational-wave-signals

import os

import h5py
import pyfstat
from scipy import stats

PATH_TO_DATA_FOLDER = './data/'


class GODataGenerator:
    def __init__(self,
        tstart: int = 1238166018,
        duration: int = 4 * 30 * 86400,
        detectors: str = 'H1,L1',
        sqrtSX: float = 1e-23,
        Tsft: int = 1800,
        SFTWindowType: str = 'tukey',
        SFTWindowBeta: float = 0.01,
        Band: float = 1.0) -> None:
        """
        GODataGenerator can be used to generate gravitational waves data

        Args:
            tstart (int, optional): Starting time of the observation [GPS time]. Defaults to 1238166018.
            duration (int, optional): Duration [seconds] 4 month. Defaults to 4*30*86400.
            detectors (str, optional): Detector to simulate. Defaults to 'H1,L1 LIGO Hanford and LIGO Livingstone'.
            sqrtSX (float, optional): Single-sided Amplitude Spectral Density of the noise. Defaults to 1e-23.
            Tsft (int, optional): Fourier transform time duration. Defaults to 1800.
            SFTWindowType (str, optional): Window function to compute short Fourier transforms. Defaults to 'tukey'.
            SFTWindowBeta (float, optional): Parameter associated to the window function. Defaults to 0.01.
            Band (float, optional): Frequency band-width around F0 [Hz]. Defaults to 1.0.
        """
        self.writer_kwargs_cw = {
            'tstart': tstart,
            'duration': duration,
            'detectors': detectors,
            'sqrtSX': sqrtSX,
            'Tsft': Tsft,
            'SFTWindowType': SFTWindowType,
            'SFTWindowBeta': SFTWindowBeta,
        }
        self.writer_kwargs_no_cw = {
            'tstart': tstart,
            'duration': duration,
            'detectors': detectors,
            'Band': Band,
            'sqrtSX': sqrtSX,
            'Tsft': Tsft,
            'SFTWindowType': SFTWindowType,
            'SFTWindowBeta': SFTWindowBeta,
        }

        self.generator_cw = pyfstat.AllSkyInjectionParametersGenerator(
            priors={
                'tref': self.writer_kwargs_cw['tstart'],
                'F0': {'uniform': {'low': 100.0, 'high': 100.1}},
                'F1': lambda: 10**stats.uniform(-12, 4).rvs(),
                'F2': 0,
                'h0': lambda: self.writer_kwargs_cw['sqrtSX'] / stats.uniform(1, 10).rvs(),
                **pyfstat.injection_parameters.isotropic_amplitude_priors,
            },
        )
        self.generator_no_cw = pyfstat.AllSkyInjectionParametersGenerator(
            priors={
                'tref': self.writer_kwargs_no_cw['tstart'],
                'F0': {'uniform': {'low': 100.0, 'high': 100.1}},
                'F1': lambda: 10**stats.uniform(-12, 4).rvs(),
                'F2': 0,
                'h0': lambda: self.writer_kwargs_no_cw['sqrtSX'] / stats.uniform(1, 10).rvs(),
                **pyfstat.injection_parameters.isotropic_amplitude_priors,
            },
        )

    def generate_signals(self, num_signals: int = 5) -> None:
        """
        Generates `num_signal` signals. Both with and without gravitational waves.

        Args:
            num_signals (int, optional): number of signals that should be produced. Defaults to 5.
        """
        for i in range(num_signals):
            self._generate_one_signal(i)
            self._generate_one_signal(i, True)

    def _generate_one_signal(self, id: int, should_contain_cw: bool = False) -> None:
        writer_kwargs = self.writer_kwargs_cw if should_contain_cw else self.writer_kwargs_no_cw
        signal_parameters_generator = self.generator_cw if should_contain_cw else self.generator_no_cw

        # Draw signal parameters.
        # Noise can be drawn by setting `params['h0'] = 0
        params = signal_parameters_generator.draw()
        writer_kwargs['outdir'] = f'{PATH_TO_DATA_FOLDER}generated/Signal_{id}'
        writer_kwargs['label'] = f'Signal_{id}'

        writer = pyfstat.Writer(**writer_kwargs, **params)
        writer.make_data()

        # Data can be read as a numpy array using PyFstat
        frequency, timestamps, amplitudes = pyfstat.utils.get_sft_as_arrays(
            writer.sftfilepath
        )

        path_to_hdf5_files = f'{PATH_TO_DATA_FOLDER}cw_hdf5/' if should_contain_cw else f'{PATH_TO_DATA_FOLDER}no_cw_hdf5/'
        if not os.path.isdir(path_to_hdf5_files):
            os.makedirs(path_to_hdf5_files)

        file_name = f'signal{id}'
        with h5py.File(f'{path_to_hdf5_files}{file_name}.hdf5', 'w') as hd5_file:
            file_grp = hd5_file.create_group(file_name)
            file_grp.create_dataset('frequency_Hz', data=frequency, dtype='f')
            l1_grp = file_grp.create_group('L1')
            l1_grp.create_dataset('SFTs', data=amplitudes['L1'], dtype='complex64')
            l1_grp.create_dataset('timestamps_GPS', data=timestamps['L1'], dtype='i')
            h1_grp = file_grp.create_group('H1')
            h1_grp.create_dataset('SFTs', data=amplitudes['H1'], dtype='complex64')
            h1_grp.create_dataset('timestamps_GPS', data=timestamps['H1'], dtype='i')
