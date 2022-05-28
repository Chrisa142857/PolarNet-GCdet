# Using PolarNet in Faster RCNN

This code is based on public code by AttFaster[1].

Model weights `.pth` are available in [https://drive.google.com/drive/folders/1r2DuiVTLlEkGKB_9lWovkL3uAJu1iN0s?usp=sharing](https://drive.google.com/drive/folders/1r2DuiVTLlEkGKB_9lWovkL3uAJu1iN0s?usp=sharing).

## How to install
```
pip install pytorch torchvision
```

## How to train
```
set MODEL_NAME=polar # saving name
set NUM_EPOCHS=100 # max epoch
set SAVE_MODEL_PATH=/path/to/best.pth # saving path of best model in val
set LOG_DIR=/path/to/log # saving path of tf log file

python run.py ^
    --model_name %MODEL_NAME% ^
    --num_epochs %NUM_EPOCHS% ^
    --save_model_path %SAVE_MODEL_PATH% ^
    --log_dir %LOG_DIR% ^
    --device "cuda" ^
    --train_batch_size 8 ^
    --use_attn 0 ^
    --use_polar 1 ^
    --start_epoch 0 ^
    --resume 0 ^
    --pretrained_resnet50_coco 1 ^
    SGD
```

## How to test
```
python test.py
```
Note: Here is no `argparse` to use, please set `model_name`, `model_path`, `csv_name`, `device` inside `test.py`

### Reference
[1] L. Cao, J. Yang, Z. Rong, L. Li, B. Xia, C. You, G. Lou, L. Jiang,
C. Du, H. Meng et al., “A novel attention-guided convolutional network
for the detection of abnormal cervical cells in cervical cancer screening,”
Medical image analysis, vol. 73, p. 102197, 2021

