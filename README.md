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
      
      
## Inference 

### Download checkpoints

      bash download_checkpoints.sh
      
Your file structure should look like:

      ├── data                                                                                                                                                                                 
      │   ├── awesome_checkpoints                                                                                                                                                              
      │   │   ├── do_02900000                                                                                                                                                                  
      │   │   ├── g_02900000                                                                                                                                                                   
      │   │   └── g_02900000.onnx                                                                                                                                                              
      │   ├── deep_voices_mel                                                                                                                                                                  
      │   │   ├── andrey_preispolnilsya.npy                                                                                                                                                    
      │   │   ├── egor_dora.npy
      │   │   └── kirill_lunch.npy
      │   └── deep_voices_wav
      │       ├── andrey_preispolnilsya.wav
      │       ├── egor_dora.wav
      │       └── kirill_lunch.wav
      
      
 ### Running inference 
 
To run inference with downloaded test-files:

       python -m src.inference
       
       
To run inference with your own files specify parameters:

| Parameter  | Description |
| ------------- | ------------- |
| input_wavs_dir | Directory with your wav files to synthesize.  |
| input_mels_dir  | Directory with pre-computed mel-spectrograms to synthesize mel. Note that mel-spectrograms should be computed with [compute_mels_from_audio.py](iSTFTNet-pytorch/scripts/compute_mels_from_audio.py) script.|
|compute_mels| Pass `--no-compute_mels` if you precomputed mels, if not specified mels will be computed from the audios in input_wavs_dir.|
|onnx_inference| If specified, checkpoint file should be `.onnx` file|
|onnx_provider| Used if onnx_inference is specified, default provider is `CPUExecutionProvider` for `CPU` inference.
|checkpoint_file| Path to the generator checkpoint or `.onnx` model|
     

## Train 

To train the model:
1. Login from CLI to Wanb account: `wandb login`.
2. Create training manifects wiht [create_manifests.py](iSTFTNet-pytorch/scripts/create_manifests.py) script.


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
