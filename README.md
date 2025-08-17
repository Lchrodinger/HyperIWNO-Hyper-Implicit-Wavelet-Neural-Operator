# HyperIWNO-Hyper-Implicit-Wavelet-Neural-Operator

## Installation
1. First clone the directory.
2. Install dependencies. We recommand creating a new environment using conda, then install all the dependencies using conda or pip.
Install [pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets).

Install pytorch:
```
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

Install other dependencies:
```
pip install -r requirements.txt
```
## Dataset
All the source data used in the paper can be generated/downloaded from [OpenFWI](https://github.com/lanl/OpenFWI) and [here](https://github.com/lu-group/fourier-deeponet-fwi).

## Training
1. First, config the yaml file in the config directory.
2. Then, simply run the following code.
   ```
   python train_wno.py
   ```
   
