# https://www.kaggle.com/code/rodrigotenorio/generating-continuous-gravitational-wave-signals
# This is the first approach to generating training data inspired by the link above

import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import h5py
import pyfstat

from src.data_management.analyse_file import plot_real_imag_spectrograms
from src.helper.utils import PATH_TO_TRAIN_FOLDER, print_blue, print_green, print_red


class GODataGenerator:
    def __init__(self,
        visualize: bool = False,
        tstart: int = 1238166018,
        duration: int = 4 * 30 * 86400,
        detectors: str = 'H1,L1',
        sqrtSX: float = 1e-23,
        Tsft: int = 360 - 1,
        SFTWindowType: str = 'tukey',
        SFTWindowBeta: float = 0.01,
        Band: float = 1.0) -> None:
        """
        GODataGenerator can be used to generate gravitational waves data

        Args:
            visualize (bool, optional): Generate a visualization of all generated data-files.
            tstart (int, optional): Starting time of the observation [GPS time]. Defaults to 1238166018.
            duration (int, optional): Duration [seconds] 4 month. Defaults to 4*30*86400.
            detectors (str, optional): Detector to simulate. Defaults to 'H1,L1 LIGO Hanford and LIGO Livingstone'.
            sqrtSX (float, optional): Single-sided Amplitude Spectral Density of the noise. Defaults to 1e-23.
            Tsft (int, optional): Fourier transform time duration. Defaults to 1800.
            SFTWindowType (str, optional): Window function to compute short Fourier transforms. Defaults to 'tukey'.
            SFTWindowBeta (float, optional): Parameter associated to the window function. Defaults to 0.01.
            Band (float, optional): Frequency band-width around F0 [Hz]. Defaults to 1.0.
        """
        self.visualize = visualize

        self.writer_kwargs_cw = {
            'tstart': tstart,
            'duration': duration,
            'detectors': detectors,
            'Band': Band,
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
            'F0': 100.0,
            'sqrtSX': sqrtSX,
            'Tsft': Tsft,
            'SFTWindowType': SFTWindowType,
            'SFTWindowBeta': SFTWindowBeta,
        }

        self.generator_cw = pyfstat.AllSkyInjectionParametersGenerator(
            priors={
                'tref': self.writer_kwargs_cw['tstart'],
                'F0': 100.0,
                'F1': -1e-9,
                'h0': sqrtSX / 20.0,
                'cosi': 1,
                'psi': 0.0,
                'phi': 0.0,
                **pyfstat.injection_parameters.isotropic_amplitude_priors,
            },
        )


    def _generate_one_signal(self, id: int, should_contain_cw: bool = False) -> None:
        params = {}
        writer_kwargs = None
        if should_contain_cw:
            params = self.generator_cw.draw()
            writer_kwargs = self.writer_kwargs_cw
        else:
            writer_kwargs = self.writer_kwargs_no_cw

        # define the folder to write to
        writer_kwargs['outdir'] = f'{PATH_TO_TRAIN_FOLDER}/generated/{id}'
        writer_kwargs['label'] = f'tmp_signal'
        writer = pyfstat.Writer(**writer_kwargs, **params)
        writer.make_data()

        # extract data from written files
        frequency, timestamps, amplitudes = pyfstat.utils.get_sft_as_arrays(
            writer.sftfilepath
        )

        # delete temporary files
        shutil.rmtree(f'{PATH_TO_TRAIN_FOLDER}/generated/{id}')

        path_to_hdf5_files = f'{PATH_TO_TRAIN_FOLDER}/cw_hdf5/' if should_contain_cw else f'{PATH_TO_TRAIN_FOLDER}/no_cw_hdf5/'
        if not os.path.isdir(path_to_hdf5_files):
            os.makedirs(path_to_hdf5_files)

        print_red(len(frequency))
        frequency_band_count = 1

        for band_index in range(frequency_band_count):
            amplitudes_h1_band = amplitudes['H1'][band_index::frequency_band_count]
            amplitudes_l1_band = amplitudes['L1'][band_index::frequency_band_count]
            frequency_band = frequency[band_index::frequency_band_count]

            file_name = f'signal{id}_{band_index}'
            with h5py.File(f'{path_to_hdf5_files}{file_name}.hdf5', 'w') as hd5_file:
                file_grp = hd5_file.create_group(file_name)
                file_grp.create_dataset('frequency_Hz', data=frequency_band, dtype='f')
                h1_grp = file_grp.create_group('H1')
                h1_grp.create_dataset('SFTs', shape=amplitudes_h1_band.shape, data=amplitudes_h1_band, dtype='complex64')
                h1_grp.create_dataset('timestamps_GPS', data=timestamps['H1'], dtype='i')
                l1_grp = file_grp.create_group('L1')
                l1_grp.create_dataset('SFTs', shape=amplitudes_l1_band.shape, data=amplitudes_l1_band, dtype='complex64')
                l1_grp.create_dataset('timestamps_GPS', data=timestamps['L1'], dtype='i')

            if self.visualize:
                with_cw = 'cw' if should_contain_cw else ''
                plot_real_imag_spectrograms(timestamps['H1'], frequency_band, amplitudes_h1_band, f'{file_name}_{with_cw}_h1')
                plot_real_imag_spectrograms(timestamps['L1'], frequency_band, amplitudes_l1_band, f'{file_name}_{with_cw}_l1')


def generate_signals(id) -> None:
    GODataGenerator()._generate_one_signal(id)



if __name__ == '__main__':
    with ProcessPoolExecutor() as p:
        for _ in p.map(generate_signals, range(100)):
            pass
