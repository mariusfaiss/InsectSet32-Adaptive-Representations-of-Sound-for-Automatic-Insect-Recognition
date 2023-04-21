import librosa
import soundfile as sf
import os
import glob
import ntpath
import shutil
import random
from audiomentations import Compose, ApplyImpulseResponse, AddGaussianSNR, FrequencyMask
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

dataset = "Train"                                           # define dataset to augment
SR = 44100                                              # sample rate
n_augments = 10                                         # number of augmentations per file
random.seed(100)
np.random.seed(100)
base_path = "/Users/mariusfaiss/PycharmProjects/TDSproject/Training/Datasets"
in_folder = f"{base_path}/{dataset} Chunks/"
out_folder = f"{base_path}/{dataset} Chunks Augmented/"

# create output folder if it does not exist
if os.path.exists(out_folder):
    shutil.rmtree(out_folder)
os.makedirs(out_folder)

# define Augmentations
augment = Compose([FrequencyMask(p=0.5, min_frequency_band=0.06, max_frequency_band=0.22),
                   AddGaussianSNR(p=1, min_snr_in_db=25, max_snr_in_db=80)])

impulse = Compose([ApplyImpulseResponse(p=0.7, leave_length_unchanged=True, ir_path="44100 IRs")])

# loop over all files for n_augment times
counter = 1
while counter <= n_augments:
    print(f"{dataset} Augmentation {counter}")
    for filename in glob.glob(os.path.join(in_folder, '*.wav')):
        mix_ratio = random.uniform(0, 1)                                    # generate mix ratio for IRs
        print_mix = "{:.0f}".format(mix_ratio*100)                          # for status indicator
        name = ntpath.basename(filename[:-4])                               # get file name without path or extension
        signal, sr = librosa.load(filename, sr=SR)                          # import signal
        augmented_signal = augment(signal, sr)                              # augment signal with Noise and FMask
        IR_signal = impulse(augmented_signal, sr)                           # apply random IR to augmented signal
        mixed_signal = (mix_ratio * augmented_signal) + ((1 - mix_ratio) * IR_signal)   # mix augmented and IR signal
        sf.write(f"{out_folder}/{name}_aug{counter}.wav", mixed_signal, samplerate=SR)
    counter += 1                                                            # update counter

# copy original, unaugmented files into output folder
for filename in glob.glob(os.path.join(in_folder, '*.wav')):
    shutil.copy2(os.path.join(in_folder, filename), out_folder)
#
# # delete input folder
# shutil.rmtree(in_folder)
