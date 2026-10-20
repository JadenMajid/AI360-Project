Quick tour of what this repo looks like (hopefully up to date).

### /data

This is where datasets go, particularly the Intel multimodal dataset for now. Look at the dataset card at HuggingFace

https://huggingface.co/datasets/IntelLabs/Intel_Robotic_Welding_Multimodal_Dataset


### /models

This is where our encoders are. They were all mentioned in the MERL paper (the winning submission for DCASE). They're all ViT-based, meaning they take patch tokens over audio, not sure about audio.

BEATs is from Microsoft and downloaded via OneDrive. Find it here: https://github.com/microsoft/unilm/tree/master/beats

EAT is from HuggingFace and can be found here: https://github.com/cwx-worst-one/EAT

Note that these encoders are trained to handle audio. Intel uses SlowFastR101 for video encoding. It's from 2019 and trained on human motion, very different from close-up welding shots. So as a stretch goal we could perhaps fine-tune our own encoder (maybe a JEPA or DiNO-based one) and crush their baseline :p


### /papers

Where the INTEL technical report and the MERL report are located. Core idea is that INTEL published the dataset and provided the baselines, but using methods from the MERL report should beat their baseline, especially on audio-only.