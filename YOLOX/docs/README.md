# YOLOX (+PolarNet)
This coed is based on official YOLOX code. Refer to [report on Arxiv](https://arxiv.org/abs/2107.08430).

Model weights `.pth` are available in [https://drive.google.com/drive/folders/1r2DuiVTLlEkGKB_9lWovkL3uAJu1iN0s](https://drive.google.com/drive/folders/1r2DuiVTLlEkGKB_9lWovkL3uAJu1iN0s).

## How to install
```
pip install pytorch torchvision
python setup.py --install # this line is optional in Win OS
```

## How to train
YOLOX
```
python train.py -f exps\glandular_dw_m.py
```
YOLOX + PolarNet
```
python train.py -f exps\glandular_dw_m_polar_1.py
```

## How to test
```
python tools\eval.py -f path\to\exp_config_file.py
```
