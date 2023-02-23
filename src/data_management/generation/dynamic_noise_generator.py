# https://www.kaggle.com/code/vslaykovsky/g2net-realistic-simulation-of-test-noise

import pyfstat
from pyfstat.utils import get_sft_as_arrays
import copy
import numpy as np
from multiprocess import Pool
from os import makedirs
from src.data_management.generation.statistics import GOStaticNoise, GOStatistics

from src.helper.utils import PATH_TO_DYNAMIC_NOISE_FOLDER

class GODynamicNoiseGenerator:
    def __init__(self, timesteps: list, statistics: GOStaticNoise = GOStatistics().noise.static) -> None:
        self.constants = statistics
        self.timestep_count = len(timesteps)
        self.timesteps = timesteps

        self.c_sqrsx = 26.5

    def bucketize_real_noise_asd(self, sfts, ts, buckets=256):
        bucket_size = (ts.max() - ts.min()) // buckets
        idx = np.searchsorted(ts, [ts[0] + bucket_size * i for i in range(buckets)])
        global_noise_amp = np.mean(np.abs(sfts))
        return np.array([np.mean(np.abs(i)) if i.shape[1] > 0 else global_noise_amp for i in np.array_split(sfts, idx[1:], axis=1)]), bucket_size

    def generate_segment(self, writer_kwargs):
        writer = pyfstat.Writer(**writer_kwargs)
        writer.make_data()
        return writer.sftfilepath


    def simulate_real_noise(self, frequency, timestamps, fourier_data, detector, buckets=256):
        asd, bucket_size = self.bucketize_real_noise_asd(fourier_data, self.timestamps, buckets=buckets)
        # makedirs(PATH_TO_DYNAMIC_NOISE_FOLDER, exist_ok=True)
        writer_kwargs = {
            "outdir": PATH_TO_DYNAMIC_NOISE_FOLDER,
            # "tstart": timestamps[0],
            "detectors": detector,  # Detector to simulate, in this case LIGO Hanford
            "F0": np.mean(frequency),  # Central frequency of the band to be generated [Hz]
            'Band': 1/5.01,  # Frequency band-width around F0 [Hz]
            # "sqrtSX": 1e-23,  # Single-sided Amplitude Spectral Density of the noise
            "Tsft": 1800,  # Fourier transform time duration
            "SFTWindowType": "tukey",
            "SFTWindowBeta": 0.01,
            "duration": bucket_size
        }

        all_args = []
        for segment in range(buckets):
            args = copy.deepcopy(writer_kwargs)
            args["label"] = f"segment_{segment}"
            args["sqrtSX"] = asd[segment] / self.c_sqrsx
            args["tstart"] = timestamps[0] + segment * bucket_size
            all_args.append(args)

        with Pool() as p:
            sft_path = p.map(self.generate_segment, all_args)

        sft_path = ";".join(sorted(sft_path))  # Concatenate different files using ;
        frequency, timestamps, fourier_data = get_sft_as_arrays(sft_path)
        ts, sft = timestamps[detector], fourier_data[detector]
        if len(sft) == 361:
            # print('Cutting 361 to 360')
            sft = sft[1:]
            frequency = frequency[1:]
        return frequency, ts, sft

if __name__ == '__main__':
    GODynamicNoiseGenerator().simulate_real_noise()