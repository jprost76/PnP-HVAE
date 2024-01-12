# PnP-HVAE

This repository contains the code to run the PnP-HVAE method proposed in

*Inverse problem regularization with hierarchical variational autoencoders*

## 1. Setup
### 1.1 Start a virtuel environment
1. create virtual environment
```
python -m venv pnphvae-env
```
Note: on windows, make sure you have virtualenv: `pip install virtualenv` and create environment with `virtualenv pnphvae-env`

2. activate virtual environment
```
source pnphvae-env/bin/activate
```
Note: on windows, run `.\pnphvae-env\Scripts\activate` 

3. install requirements:
```
pip install -r requirements.txt
```
Notes:
- if you experience issues with `pip install -r requirements.txt` try upgrading pip
    ```
    python -m pip install --upgrade pip
    ```
- if you experience issues with `Pillow` try
    ```
    python -m pip install --upgrade pillow
    ```
- make sure your install of `cupy` is compatible with your cuda version (see https://docs.cupy.dev/en/stable/install.html#installing-cupy)
    ```
    pip install cupy-cuda11x
    ```
- to run patchVDVAE, `hparams` library is required, if failed with requirements.txt run:
    ```
    pip install --upgrade git+https://github.com/Rayhane-mamah/hparams
    ```
- if you are experiencing issues with PIL import try to uninstall and reinstall pillow :
    ```
    pip uninstall pillow
    pip install pillow
    ```
### 1.2 Download VAE weights

- VDVAE:
```
cd VAEs/vdvae/saved_models
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq256-iter-1700000-model-ema.th
```
Note: for windows user, you can download the weight from your browser

- PatchVDVAE:
```
cd VAEs/patchVDVAE/src/saved_models
wget https://osf.io/download/udjny/?view_only=a152beb1784a4ee4b2c41f9993b306b7
```
Note : you can also paste the link in your browser to download the weights

## 2. Run the method

Run PnP-HVAE:
```
python main.py exp=face_inpainting
```
You can change the experience by changing the exp option (see the conf/exp file for the different experiments):

    - face_inpainting
    - face_deblurring
    - face_sr
    - bsd_deblurring

If you want to log the iterations 
```
python main.py exp=face_inpainting log_images=True
```

## 3. Acknowledgements
This repo is build upon [VDVAE original repository](https://github.com/openai/vdvae)  and [efficient-vdvae repository](https://github.com/Rayhane-mamah/Efficient-VDVAE).
