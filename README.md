# iSTFTNet : Fast and Lightweight Mel-spectrogram Vocoder Incorporating Inverse Short-time Fourier Transform
This repository is based on the [opensource implementation](https://github.com/rishikksh20/iSTFTNet-pytorch) of iSTFTNet (model `C8C8I`). Our contribution to the repository:
- we changed the `logging` – added `loguru` & `wandb`; 
- added `Docerfile` for faster env set up;
- updated the code with several scripts to `compute mel-spectrograms` and `convert the model to .onnx`;
- we share the weights of the model we trained on robust internal dataset consists of Russian speech recorded in different acoustic conditions.

## Setup env

### Docker

      bash run_docker.sh
      
      
### Conda 

      conda create —name istft-vocoder python=3.9
      # change the link if you have different version of cuda / no cuda
      pip install torch torchvision torchaudio —extra-index-url https://download.pytorch.org/whl/cu116
      pip install -r requirements.txt
      

## Citations :
```
@inproceedings{kaneko2022istftnet,
title={{iSTFTNet}: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time Fourier Transform},
author={Takuhiro Kaneko and Kou Tanaka and Hirokazu Kameoka and Shogo Seki},
booktitle={ICASSP},
year={2022},
}
```

## References:
* https://github.com/rishikksh20/iSTFTNet-pytorch
