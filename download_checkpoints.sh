#!/bin/bash
cur="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)" # to make running the script from anywhere available
cd $cur
cd src
# download awesome_checkpoints.zip (generator weights, discriminator weights, onnx models)
gdown https://drive.google.com/uc?id=1CQkS0AZyI_e8L68qf_cR3lNRWorspP3B;
unzip -qq awesome_checkpoints.zip;