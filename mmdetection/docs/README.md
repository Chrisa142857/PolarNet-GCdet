# Other models (FCOS, Cascade RCNN)

This code is based on `mmdetection`.

Model weights `.pth` are in [https://drive.google.com/drive/folders/1r2DuiVTLlEkGKB_9lWovkL3uAJu1iN0s](https://drive.google.com/drive/folders/1r2DuiVTLlEkGKB_9lWovkL3uAJu1iN0s).

## How to install
```
pip install torch torchvision
python setup.py --install # this line is optional in Win OS
```

## How to train

FCOS:
```
python tools\train.py configs\glandular_exp\fcos_r50.py
```

FCOS + PolarNet:
```
python tools\train.py configs\glandular_exp\fcos_r50_polar.py
```

Cascade RCNN:
```
python tools\train.py configs\glandular_exp\cascade_rcnn_r50.py
```

Cascade RCNN + PolarNet:
```
python tools\train.py configs\glandular_exp\cascade_rcnn_r50_polar.py
```

## How to test

Choose a `\path\to\model_config.py` in dir `configs\`. Set the corresponding `\path\to\model_weights.pth`. Then
```
python tools\test.py \path\to\model_config.py \path\to\model_weights.pth --eval bbox --out \path\to\save_results.pkl
```
