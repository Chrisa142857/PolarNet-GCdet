# PolarNet, GC (Glandular Cell) detection from cytology WSI

Code available of a preprint paper *Cervical Glandular Cell Detection from Whole Slide Image with Out-Of-Distribution Data* (refer to [arXiv](https://arxiv.org/abs/2205.14625)). Please cite it.

Plugin PolarNet using eight neighboring self-attention mechanism, for improving GC (glandular cell) detection from cytology WSI (Whole Slide Image). 

It can be plugged into any modern deep learning-based object detection model. The code of PolarNet `polar_net.py` in this project is easiest to read about the proposed network.

The implementation of PolarNet in different detection frameworks is identical:
- SOTA double-stage: `Faster RCNN/` (based on the github repo [AttFaster](https://github.com/cl2227619761/AttFPN-Ovarian-Cancer/tree/master))
- SOTA one-stage: `YOLOX/` (based on the github repo [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX))
- Others: `mmdetection/` (based on the github repo [mmdetection](https://github.com/open-mmlab/mmdetection))

Data is available except for images:
- COCO format annotation: `data/mmdetection`
- CSV format annotation: `data/YOLOX` & `data/Faster RCNN`

Please see the corresponding `README.md` inside folders.
