import torch
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

Object = namedtuple("Object",
                    ["image_path", "object_id", "object_type",
                     "coordinates"])
Prediction = namedtuple("Prediction",
                        ["image_path", "probability", "polarity", "object_type", "coordinates"])

def voc_ap(recall, precision, use_07_metric=False):
    """
    Calculate the AP value using recall and precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p/11.

    else:
        mrec = np.concatenate(([0.], recall, [1.]))
        mprec = np.concatenate(([0.], precision, [0.]))
        for i in range(mprec.size - 1, 0, -1):
            mprec[i-1] = np.maximum(mprec[i-1], mprec[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i+1] - mrec[i]) * mprec[i+1])

    return ap

def custom_voc_eval(gt_csv, pred_csv, pscore_pth, num_class=2, ovthresh=0.5, use_07_metric=False):

    pscores = torch.load(pscore_pth)['p_score']

    num_image = 0
    num_object = 0
    num_object_subclass = [0 for _ in range(num_class)]
    object_dict = {}
    with open(gt_csv) as f:
        # skip header
        next(f)
        for line in f:
            image_path, annotation = line.strip("\n").split(",")
            if annotation == "":
                num_image += 1
                continue
            
            object_annos = annotation.split(";")
            for object_anno in object_annos:
                fields = object_anno.split(" ")  # one box
                if len(fields) == 0 or fields[0] == '': 
                  continue

                object_type = int(fields[0])
                coords = np.array(list(map(float, fields[1:])))
                if 0 in coords.shape: 
                  continue
                x1, y1, x2, y2 = coords
                # if not is_box_valid_after_crop(crop_x, crop_y, x1, x2, y1, y2, new_side):
                #   continue
                # one box info
                obj = Object(image_path, num_object, object_type, coords)
                if image_path in object_dict:
                    object_dict[image_path].append(obj)
                else:
                    object_dict[image_path] = [obj]
                num_object += 1
                num_object_subclass[int(object_type)] += 1
            num_image += 1

    # parse prediction csv, by parsing pred csv, we get the pre box info
    preds = []
    empty_pred_num = 0
    with open(pred_csv) as f:
        # print(len(pscores), len(f.read().split('\n'))-2)
        # skip header
        next(f)
        for li, line in enumerate(f):
            pscore = pscores[li-empty_pred_num]
            image_path, prediction = line.strip("\n").split(",")
            
            if prediction == "":
                if len(pscore) != 0:
                    empty_pred_num += 1
                continue

            coord_predictions = prediction.split(";")
            if len(pscore) != len(coord_predictions):
                print("No.", li, "line")
                print(len(pscore),len(coord_predictions))
                exit()
            for ci, coord_prediction in enumerate(coord_predictions):
                polarity = pscore[ci]
                fields = coord_prediction.split(" ")
                probability, object_type, x1, y1, x2, y2 = list(map(float, fields))
                probability = probability*2 - polarity
                pred = Prediction(image_path, probability, polarity, object_type,
                                  np.array([x1, y1, x2, y2]))
                preds.append(pred)
                
    # sort prediction by probability, decrease order
    preds = sorted(preds, key=lambda x: x.probability, reverse=True)
    nd = len(preds)  # total number of pred boxes
    object_hitted = set()
    object_hitted_nonclass = set()
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    tp_nonclass = np.zeros(nd)
    fp_nonclass = np.zeros(nd)
    tp_subclass = [np.zeros(nd) for _ in range(num_class)]
    fp_subclass = [np.zeros(nd) for _ in range(num_class)]

    num_ood_fp = 0 
    true_pscore = []
    false_pscore = []
    # loop over each pred box to see if it matches one ground box
    if nd > 0:
        for d in range(nd):
            # one pred box coords
            bb = preds[d].coordinates.astype(float)
            pred_type = preds[d].object_type
            image_path = preds[d].image_path
            pscore = preds[d].polarity
            if preds[d].image_path in object_dict:
               # set the initial max overlap iou
               ovmax = -np.inf
               # ground box on the image
               R = [i.coordinates for i in object_dict[image_path] if 0 not in i.coordinates.shape]
               try:
                   BBGT = np.stack(R, axis=0)
               except ValueError:
                   import ipdb;ipdb.set_trace()
               R_img_id = [i.object_id for i in object_dict[image_path]]
               R_object_type = [i.object_type for i in object_dict[image_path]]
               BBGT_hitted_flag = np.stack(R_img_id, axis=0)
               BBGT_object_type = np.stack(R_object_type, axis=0)

               if BBGT.size > 0:
                   # cal the iou between pred box and all the gt boxes on
                   # the image
                   ixmin = np.maximum(BBGT[:, 0], bb[0])
                   iymin = np.maximum(BBGT[:, 1], bb[1])
                   ixmax = np.minimum(BBGT[:, 2], bb[2])
                   iymax = np.minimum(BBGT[:, 3], bb[3])

                   # cal inter area width
                   iw = np.maximum(ixmax - ixmin + 1., 0.)
                   ih = np.maximum(iymax - iymin + 1., 0.)
                   inters = iw * ih  # inter area

                   # cal iou
                   union = (
                       (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) * (BBGT[:, 3] - BBGT[:, 1] + 1.) - \
                       inters
                   )
                   overlaps = inters / union
                   # find the max iou
                   ovmax = np.max(overlaps)
                   # find the index of the max iou
                   jmax = np.argmax(overlaps)

               if ovmax > ovthresh:
                    # if the max iou greater than the iou thresh
                    if BBGT_hitted_flag[jmax] not in object_hitted_nonclass:
                        tp_nonclass[d] = 1.
                        object_hitted_nonclass.add(BBGT_hitted_flag[jmax])
                    else:
                        fp_nonclass[d] = 1.
                    if BBGT_hitted_flag[jmax] not in object_hitted and BBGT_object_type[jmax] == pred_type:
                        tp_subclass[int(pred_type)][d] = 1.
                        tp[d] = 1.
                        object_hitted.add(BBGT_hitted_flag[jmax])
                    else:
                        fp_subclass[int(pred_type)][d] = 1.
                        fp[d] = 1.
               else:
                    fp_subclass[int(pred_type)][d] = 1.
                    fp_nonclass[d] = 1.
                    fp[d] = 1.
            else:
                fp_subclass[int(pred_type)][d] = 1.
                fp_nonclass[d] = 1.
                fp[d] = 1.
                num_ood_fp += 1
            if tp[d] == 1:
                true_pscore.append(pscore)
            else:
                false_pscore.append(pscore)
            
    print("num_ood_fp", num_ood_fp)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    fp_nonclass = np.cumsum(fp_nonclass)
    tp_nonclass = np.cumsum(tp_nonclass)
    fp_subclass = [np.cumsum(sub) for sub in fp_subclass]
    tp_subclass = [np.cumsum(sub) for sub in tp_subclass]

    recs, precs, aps = {}, {}, {}

    rec = tp / float(num_object)
    prec = tp / np.maximum(tp+fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    recs['all_%d'%(ovthresh*100)] = rec
    precs['all_%d'%(ovthresh*100)] = prec
    aps['all_%d_ap'%(ovthresh*100)] = ap
    rec = tp_nonclass / float(num_object)
    prec = tp_nonclass / np.maximum(tp_nonclass+fp_nonclass, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    # recs['nonclass_%d'%(ovthresh*100)] = rec
    # precs['nonclass_%d'%(ovthresh*100)] = prec
    # aps['nonclass_%d'%(ovthresh*100)] = ap

    for i, (tp, fp, num_object) in enumerate(zip(tp_subclass, fp_subclass, num_object_subclass)):
      rec = tp / float(num_object)
      prec = tp / np.maximum(tp+fp, np.finfo(np.float64).eps)
      ap = voc_ap(rec, prec, use_07_metric)
    #   recs['class%d_%d'%(i, ovthresh*100)] = rec
    #   precs['class%d_%d'%(i, ovthresh*100)] = prec
    #   aps['class%d_%d'%(i, ovthresh*100)] = ap

    return recs, precs, aps, true_pscore, false_pscore


if __name__ == '__main__':
    # gt_csv = "val.csv"
    # pred_csv = "temp/polarv4_resnet50_epoch%d_loc.csv"
    # pscore_pth = "temp/polarv4_resnet50_epoch%d_pscore.pth"
    gt_csv = "test_w_ood.csv"
    pred_csv = "temp/test_r50_use_attn=0_use_polar=1v4_test_w_ood.csv_loc.csv"
    pscore_pth = "temp/test_r50_use_attn=0_use_polar=1v4_test_w_ood.pth_pscore.pth"
    
    outputs = custom_voc_eval(gt_csv, pred_csv, pscore_pth)
    true_pscore, false_pscore = outputs[-2:]
    true_pscore, false_pscore = torch.stack(true_pscore), torch.stack(false_pscore)
    Acc = 0
    Tpr = 0
    Fpr = 0
    Thr = 0
    for i in range(100):
        tpr = len([p for p in true_pscore if p >= i/100]) / len(true_pscore)
        fpr = len([p for p in false_pscore if p < i/100]) / len(false_pscore)
        acc = (tpr + fpr) / 2
        if acc > Acc:
            Acc = acc
            Tpr = tpr
            Fpr = fpr
            Thr = i
    print("%.16f" % true_pscore.max(), "%.16f" % true_pscore.min(), "%.16f" % true_pscore.mean(), "%.16f" % true_pscore.std())
    print("%.16f" % false_pscore.max(), "%.16f" % false_pscore.min(), "%.16f" % false_pscore.mean(), "%.16f" % false_pscore.std())
    print(outputs[2]['all_50_ap'], Acc, Tpr, Fpr, Thr)
    # plt.figure()
    # for i in range(10):
    #     num = len([p for p in true_pscore if i/10 <= p < (i+1)/10])
    #     x = (i+0.25)/10
    #     if i < 9:
    #         plt.bar(x, num, width=0.01, color='b')
    #     else:
    #         plt.bar(x, num, width=0.01, color='b', label='TP')
    # for i in range(10):
    #     num = len([p for p in false_pscore if i/10 <= p < (i+1)/10])
    #     x = (i+0.75)/10
    #     if i < 9:
    #         plt.bar(x, num, width=0.01, color='r')
    #     else:
    #         plt.bar(x, num, width=0.01, color='r', label='FP')
    # plt.legend()
    # plt.savefig("temp.jpg")
    exit()
    for i in range(9,45):
        print("Epoch", i)
        outputs = custom_voc_eval(gt_csv, pred_csv%i, pscore_pth%i)
        true_pscore, false_pscore = outputs[-2:]
        true_pscore, false_pscore = torch.stack(true_pscore), torch.stack(false_pscore)
        Acc = 0
        Tpr = 0
        Fpr = 0
        Thr = 0
        for i in range(100):
            tpr = len([p for p in true_pscore if p >= i/100]) / len(true_pscore)
            fpr = len([p for p in false_pscore if p < i/100]) / len(false_pscore)
            acc = (tpr + fpr) / 2
            if acc > Acc:
                Acc = acc
                Tpr = tpr
                Fpr = fpr
                Thr = i
        # print(len([p for p in true_pscore if p != 0]) / len(true_pscore), true_pscore.mean())
        # print(len([p for p in false_pscore if p != 0]) / len(false_pscore), false_pscore.mean())
        print(outputs[2]['all_50_ap'], Acc, Tpr, Fpr, Thr)
    # plt.figure()
    # for i in range(100):
    #     num = len([p for p in true_pscore if i/10 <= p < (i+1)/10])
    #     x = (i+0.25)/10
    #     if i < 9:
    #         plt.bar(x, num, width=0.01, color='b')
    #     else:
    #         plt.bar(x, num, width=0.01, color='b', label='TP')
    # for i in range(10):
    #     num = len([p for p in false_pscore if i/10 <= p < (i+1)/10])
    #     x = (i+0.75)/10
    #     if i < 9:
    #         plt.bar(x, num, width=0.01, color='r')
    #     else:
    #         plt.bar(x, num, width=0.01, color='r', label='FP')
    # plt.legend()
    # plt.savefig("temp.jpg")