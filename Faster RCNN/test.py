from tools.engine import freeze_bn, custom_voc_evaluate
from utils.backbone_utils import resnet_fpn_backbone
from faster_rcnn import FasterRCNN, FastRCNNPredictor
from faster_polar_rcnn import FasterPolarRCNN
import _utils
from datasets.data import get_dataset
from run import get_transform
from polar_net import PolarNet

import torch
from thop import profile
from copy import deepcopy

def get_model_info(model, tsize):
    device = 'cuda'
    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=device) #next(model.parameters()).device
    flops, params = profile(deepcopy(model.to(device)), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info

if __name__ == '__main__':

    use_attn = 0
    use_polar = 1
    mtype = 'v4.0.1c'
    _mtype = 'v4c'
    tail = '_w_neg'
    # use_new_v = 1
    model_name = "r50_use_attn=%d_use_polar=%d%s" % (use_attn, use_polar, mtype)
    model_path = "results/polar%s_r50%s/best.pth" % (mtype, tail)
    CLASSES = {"__background__", "nGEC", "AGC"}
    val_csvname = "test_w_ood.csv"
    # val_csvname = "val.csv"
    val_batch_size = 1
    num_workers = 4
    torch.cuda.set_device(2)
    device = torch.device("cuda")
    print("====Loading data====")
    dataset_val = get_dataset(val_csvname,
                              datatype="val",
                              transform=get_transform(train=False))
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers,
        collate_fn=_utils.collate_fn
    )


    print("====Loading model====")
    backbone = resnet_fpn_backbone("resnet50", False, use_attn=use_attn)
    if use_polar == 1:
        polar_net = PolarNet(mtype=_mtype)
        model = FasterPolarRCNN(backbone, polar_net, mtype=_mtype, num_classes=len(CLASSES))
    else:
        model = FasterRCNN(backbone, num_classes=len(CLASSES))

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                      len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)

    model.apply(freeze_bn)
    print(get_model_info(model, (1024, 1024)))

    val_mAP = custom_voc_evaluate(
        model, dataloader_val, device=device,
        gt_csv_path=val_csvname,
        cls_csv_path="temp/test_%s_%s_cls.csv" % (model_name, val_csvname),
        loc_csv_path="temp/test_%s_%s_loc.csv" % (model_name, val_csvname)
    )

    print(val_mAP)