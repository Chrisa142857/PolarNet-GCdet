# PolarNet, GC (Glandular Cell) detection from cytology WSI

Code available of a preprint paper *Cervical Glandular Cell Detection from Whole Slide Image with Out-Of-Distribution Data* (refer to [arXiv](https://arxiv.org/abs/2205.14625))

Plugin PolarNet using eight neighboring self-attention mechanism, for improving GC (glandular cell) detection from cytology WSI (Whole Slide Image). 

It can be plugged into any modern deep learning-based object detection model. Code available of a paper *Cervical GC detection from WSI with OOD data* (under review). Please cite it.

The implement of PolarNet in different detection frameworks:
- SOTA double-stage: `Faster RCNN/` (based on the github repo [AttFaster](https://github.com/cl2227619761/AttFPN-Ovarian-Cancer/tree/master))
- SOTA one-stage: `YOLOX/` (based on the github repo [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX))
- Others: `mmdetection/` (based on the github repo [mmdetection](https://github.com/open-mmlab/mmdetection))

Data is available except for images:
- COCO format: `data/mmdetection`
- CSV format: `data/YOLOX` & `data/Faster RCNN`

Please see the corresponding `README.md` inside folders.
