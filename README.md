# InsectSet32: Adaptive Representations of Sound for Automatic Insect Recognition

This repository contains the code used for a masters thesis comparing the waveform-based frontend [LEAF](https://github.com/google-research/leaf-audio) to the classic mel-spectrogram approach for classifying insect sounds. A preprint of the thesis is published at [arxiv](https://arxiv.org/abs/2211.09503). The dataset that was compiled for this work is publicly available on zenodo as [InsectSet32](https://zenodo.org/record/7072196). This work has been expanded upon by compiling two more extensive datasets, [InsectSet47 & InsectSet66](https://zenodo.org/record/7828439), which are also available on zenodo. This expanded work was published in [PLOS Computational Biology](https://doi.org/10.1371/journal.pcbi.1011541) and the code is also available on [Github](https://github.com/mariusfaiss/InsectSet47-InsectSet66-Adaptive-Representations-of-Sound-for-Automatic-Insect-Recognition).

This repository includes the [classifier](https://github.com/mariusfaiss/Adaptive-Representations-of-Sound-for-Automatic-Insect-Recognition/blob/main/Mel_LEAF_InsectClassifier.py) that was built and tested, which can be used with a mel-spectrogram frontend as the standard approach, or with the adaptive, waveform based frontend [LEAF](https://github.com/google-research/leaf-audio) (using the [pytorch implementation](https://github.com/SarthakYadav/leaf-pytorch)) which achieved substantially better performance. For preparation of the input data, [a script that splits the audio files](https://github.com/mariusfaiss/Adaptive-Representations-of-Sound-for-Automatic-Insect-Recognition/blob/main/SplitAudioChunks.py) into overlapping five second long chunks of audio is included. This should be applied to all audio files to match the input length of the classifier. A [data augmentation script](https://github.com/mariusfaiss/Adaptive-Representations-of-Sound-for-Automatic-Insect-Recognition/blob/main/DataAugmentation.py) is inlcuded, which should be applied to the training set.

Below is the abstract of the [thesis](https://arxiv.org/abs/2211.09503) describing the project:

Insects are an integral part of our ecosystem. These often small and evasive animals have a big impact on their surroundings, providing a large part of the present biodiversity and pollination duties, forming the foundation of the food chain and many biological and ecological processes. Due to factors of human influence, population numbers and biodiversity have been rapidly declining with time. Monitoring this decline has become increasingly important for conservation measures to be effectively implemented. But monitoring methods are often invasive, time and resource intense, and prone to various biases. Many insect species produce characteristic mating sounds that can easily be detected and recorded without large cost or effort. Using deep learning methods, insect sounds from field recordings could be automatically detected and classified to monitor biodiversity and species distribution ranges. In this project, I implement this using existing datasets of insect sounds (Orthoptera and Cicadidae) and machine learning methods and evaluate their potential for acoustic insect monitoring. I compare the performance of the conventional spectrogram-based deep learning method against the new adaptive and waveform-based approach LEAF. The waveform-based frontend achieved significantly better classification performance than the Mel-spectrogram frontend by adapting its feature extraction parameters during training. This result is encouraging for future implementations of deep learning technology for automatic insect sound recognition, especially if larger datasets become available.

The IR files used for data augmentation are sourced from the [OpenAIR](https://www.openairlib.net) library.

[Gill Heads Mine](https://www.openair.hosted.york.ac.uk/?page_id=494)

44100_dales_site1_4way_mono.wav

44100_dales_site2_4way_mono.wav

44100_dales_site3_4way_mono.wav

[Koli National Park - Winter](https://www.openair.hosted.york.ac.uk/?page_id=584)

44100_koli_snow_site1_4way_mono.wav

44100_koli_snow_site2_4way_mono.wav

44100_koli_snow_site3_4way_mono.wav

44100_koli_snow_site4_4way_mono.wav

[Koli National Park - Summer](https://www.openair.hosted.york.ac.uk/?page_id=577)

44100_koli_summer_site1_4way_mono.wav

44100_koli_summer_site2_4way_mono.wav

44100_koli_summer_site3_4way_mono.wav

44100_koli_summer_site4_4way_mono.wav
