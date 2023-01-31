import pyfstat
import numpy as np
import math
import sys

from contextlib import redirect_stderr
from io import StringIO
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from os.path import join, isdir
from os import makedirs, devnull
from cv2 import imwrite
from secrets import token_hex
from shutil import rmtree

from src.helper.utils import PATH_TO_CACHE_FOLDER, PATH_TO_SIGNAL_FOLDER, print_red
from src.data_management.generation.timestamps_generator import GOTimestepGenerator

class GOSignalGenerator:
    def __init__(self, timesteps_big: list) -> None:
        timesteps_tmp = [x-timesteps_big[0] for x in timesteps_big]
        t_min = 630720013
        timesteps = [x+t_min for x in timesteps_tmp]
        t_start = timesteps[0]
        self.writer_kwargs = {
            'sqrtSX': 1e-23, # Single-sided Amplitude Spectral Density of the noise
            'Tsft': 1800, # Fourier transform time duration
            "SFTWindowType": "tukey",  # Window function to compute short Fourier transforms
            "SFTWindowBeta": 0.01,  # Parameter associated to the window function
            "detectors": "H1,L1",
            'timestamps': {
                'H1': np.array(timesteps, dtype=int),
                'L1': np.array(timesteps, dtype=int),
            }
       }

        self.signal_params = {
            # polarization angle
            'psi': np.random.uniform(-math.pi / 4, math.pi / 4),
            # phase
            'phi': np.random.uniform(0, math.pi * 2),
            # Cosine of the angle between the source and us. Range: [-1, 1]
            'cosi': np.random.uniform(-1, 1),
            # Central frequency of the band to be generated [Hz]
            'F0': np.random.uniform(50, 500),
            'F1': -10**np.random.uniform(-12, -8),
            'F2': 0.0,
            'Band': 0.2, # Frequency band-width around F0 [Hz]
            'Alpha': np.random.uniform(0, math.pi * 2), # Right ascension of the source's position on the sky
            'Delta': np.random.uniform(-math.pi / 2, math.pi / 2), # Declination of the source's position on the sky,
            'tp': t_start, #+ 86400 * random.randint(0, 30), # signal offset
            # 'h0': writer_kwargs["sqrtSX"] * np.random.uniform(0.10, 0.02),
            'h0': self.writer_kwargs["sqrtSX"] * 100,
            #         'asini': random.randint(10, 500), # amplitude of signal
            #         'period': random.randint(90, 730) * 86400,
        }

    def __call__(self, idx: int):
        path_to_tmp_folder = join(PATH_TO_CACHE_FOLDER, 'signal_generation', str(idx))
        makedirs(path_to_tmp_folder, exist_ok=True)
        
        self.writer_kwargs['outdir'] = path_to_tmp_folder
        self.writer_kwargs['label'] = 'Signal'
        try:
            sys.stdout = open(devnull, 'w')
            writer = pyfstat.BinaryModulatedWriter(**self.writer_kwargs, **self.signal_params)
            writer.make_data()
            frequency, timestamps, amplitudes = pyfstat.utils.get_sft_as_arrays(
                writer.sftfilepath
            )
            sys.stdout = sys.__stdout__

            rmtree(path_to_tmp_folder)

            token = token_hex(3)
            for det in amplitudes:
                amp = np.abs(amplitudes[det][-360:, :])
                amp = (amp - amp.min())
                amp = amp * 255 / (amp.max())
                fname = join(PATH_TO_SIGNAL_FOLDER, f'{det}_{token}.png')
                imwrite(fname, amp)

        except Exception as ex:
            print(ex)

    def generate_signals(self, n: int):
        with ProcessPoolExecutor() as p:
            for files in tqdm(p.map(self, range(n))):
                pass
        # GOSignalGenerator()
        # for _ in range(n):
        #     self.generate_single_signal()


if __name__ == '__main__':
    timesteps = GOTimestepGenerator().generate_timestamps()
    samples = 100
    GOSignalGenerator(timesteps).generate_signals(samples)
