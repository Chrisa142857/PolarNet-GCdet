#!/usr/bin/env python
# coding=utf-8
"""
进行evaluation使用
"""
import xml.etree.ElementTree as ET
from collections import defaultdict
import os
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

import sys


def parse_rec(filename):
    """
    作用：解析给定的xml文件
    """
    tree = ET.parse(filename)
    targets = parse_voc_xml(tree.getroot())
    all_objects = []
    anno = targets['doc']
    img_width = float(anno['size']['width'])
    img_height = float(anno['size']['height'])
    objs = anno['outputs']['object']['item']
    if isinstance(objs, dict):
        objs = [objs]
    for obj in objs:
        if 'bndbox' in obj:
            obj_struct = {}
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])
            bbox_ws = xmax - xmin
            bbox_hs = ymax - ymin
            name = obj['name']
            if name == '异常':
                name = "Positive"
            if bbox_ws > 0 and bbox_hs > 0:
                obj_struct['bbox'] = [xmin, ymin, xmax, ymax]
                obj_struct['name'] = name
                obj_struct['difficult'] = 0
                obj_struct['img_width'] = img_width
                obj_struct['img_height'] = img_height
            if obj_struct:
                all_objects.append(obj_struct)

    return all_objects


def parse_voc_xml(node):
    voc_dict = {}
    children = list(node)
    if children:
        def_dic = defaultdict(list)
        for dc in map(parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        voc_dict = {
            node.tag:
            {ind: v[0] if len(v) == 1 else v
            for ind, v in def_dic.items()}
        }
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict


def voc_ap(recall, precision, use_07_metric=False):
    """
    使用得到的召回率和准确率计算该类的ap值

    参数:
        recall: 召回率
        precision: 查准率
        use_07_metric: 是否使用07年的计算方式，默认否；使用的是
            10年的计算方式
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
        # 使用10年的计算方式,即PR曲线下面积
        mrec = np.concatenate(([0.], recall, [1.]))
        mprec = np.concatenate(([0.], precision, [0.]))

        # 将pr曲线进行平滑，弄成矩形
        for i in range(mprec.size - 1, 0, -1):
            mprec[i-1] = np.maximum(mprec[i-1], mprec[i])

        # 找到recall变化的点
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # 计算矩形的面积
        ap = np.sum((mrec[i+1] - mrec[i]) * mprec[i+1])

    return ap


def custom_voc_eval(classname,
                   detpath,
                   imagesetfile,
                   annopath='',
                   ovthresh=0.5,
                   use_07_metric=False):
    """
    作用: 计算指定类别的tp, fp等

    参数:
        classname: 指定的类别
        detpath: 预测的结果存储的路径
        imagesetfile: 文件名所在的txt文件，每一行是一个图片的名字，无后缀
        annopath: xml文件的路径
        ovthresh: iou的阈值，默认为0.5
        use_07_metric: 是否使用VOC07的方式计算，默认为False
    """
    # recs为一个字典，键为图片名(无后缀),值为该图片的xml解析后的信息
    # recs存储的是所有图片的目标物体信息
    recs = {}
    # 读取图片
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
        imagenames = [x.strip() for x in lines]

        # 加载xml文件并解析，存入recs
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))

    # 从recs中提取出每张图片中该类(指定类)的目标物体信息
    class_recs = {}
    npos = 0  # 所有图片该类不是difficult的目标的个数
    try:
        for imagename in imagenames:
            record = [obj for obj in recs[imagename]
                     if obj['name'] == classname]
            img_width = record[0]['img_width']
            img_height = record[0]['img_height']
            # 该类目标框的坐标集合
            bbox = np.array([x['bbox'] for x in record])
            bbox[:, [0, 2]] = np.clip(
                bbox[:, [0, 2]], a_min=0, a_max=img_width
            )
            bbox[:, [1, 3]] = np.clip(
                bbox[:, [1, 3]], a_min=0, a_max=img_height
            )
            difficult = np.array(
                [x['difficult'] for x in record]
            ).astype(np.bool)
            det = [False] * len(record)  # 用来在后面匹配的时候作为是否被匹配过的标志
            npos = npos + sum(~difficult)  # 去除difficult的目标
            # 每张图片的名字作为键名，其值为该图片中的该类目标的坐标框
            # 框的difficult与否,是否检测到
            class_recs[imagename] = {
                "bbox": bbox,
                "difficult": difficult,
                "det": det
            }
    except KeyError as e:
        import ipdb;ipdb.set_trace()

    # 读取模型的检测结果
    # 注意：现在是针对其中一类进行操作的
    detfile = detpath.format(classname)  # 该类的检测结果？
    with open(detfile, 'r') as f:
        lines = f.readlines()
    # 记录的格式应该是：第一列为图片的id
    # 第二列应为预测的得分(应为该图片其中的一个框)
    # 第三列及其后面为该预测框的坐标
    splitlines = [x.strip().split(' ') for x in lines]  # 每行的记录
    image_ids = [x[0] for x in splitlines]  # 每张图片的id
    # 每个预测框得分
    confidence = np.array(
        [float(x[1]) for x in splitlines]
    )
    # 每隔预测框坐标
    BB = np.array(
        [[float(z) for z in x[2:]] for x in splitlines]
    )

    nd = len(image_ids)  # 该预测文件中一共有多少行，即多少个该类预测目标
    tp = np.zeros(nd)  # 判断正确的个数，初始为0（真阳性）
    fp = np.zeros(nd)  # 判断错误的个数，初始为0（假阳性）
    
    # 开始遍历该类的每个预测框看是否匹配上真实框（每个真实框匹配一个预测框）
    if BB.shape[0] > 0:
        # 先对预测框的坐标按照预测框得分高低排序,从大到小排序
        sorted_idxs = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_idxs, :]
        # 每个框所在的图片索引
        image_ids = [image_ids[x] for x in sorted_idxs]

        for d in range(nd):
            # 该预测框所在图片的所有该类真实框标注信息
            R = class_recs[image_ids[d]]
            # 该预测框的坐标
            bb = BB[d, :].astype(float)
            # 设定初始的最大iou
            ovmax = -np.inf
            # 该图片上所有该类真实框的坐标
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # 计算该预测框与该张图片上所有该类真实框的iou
                # 重叠区域的xmin
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                # 重叠区域的ymin
                iymin = np.maximum(BBGT[:, 1], bb[1])
                # 重叠区域的xmax
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                # 重叠区域的ymax
                iymax = np.minimum(BBGT[:, 3], bb[3])
                
                # 重叠区域的宽度
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih  # 重叠区域的面积

                # 计算iou
                union = (
                    (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
                    (BBGT[:, 2] - BBGT[:, 0] + 1.) * (BBGT[:, 3] - BBGT[:, 1] + 1.) - \
                    inters
                )
                overlaps = inters / union
                # 找出最大的iou
                ovmax = np.max(overlaps)
                # 最大的iou对应的真实框的索引
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                # 若该iou大于设定的阈值0.5
                if not R['difficult'][jmax]:
                    # 且该真实框不是difficult的框
                    if not R['det'][jmax]:
                        # 且该真实框没有被匹配过
                        tp[d] = 1.  # 则该预测框预测正确
                        R['det'][jmax] = 1  # 将该真实框标记为已匹配
                    else:
                        # 若该真实框已经被其它预测框匹配过
                        fp[d] = 1.  # 则该预测框定为预测错误
            else:
                # 若该iou小于设定的阈值
                fp[d] = 1.  # 则该预测框预测错误
        
        # 详细解释见笔记
        fp = np.cumsum(fp)  
        tp = np.cumsum(tp)

        # 该类的召回率
        rec = tp / float(npos)  # npos为该类所有非difficult真实框的数量
        # 该类的precision
        prec = tp / np.maximum(tp+fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)

        return rec, prec, ap


def write_custom_voc_results_file(data_loader,
                                 all_boxes,
                                 image_index,
                                 root,
                                 classes,
                                 voc_results_dir,
                                 thread=0.3):
    """
    作用:
        将验证的结果存储起来

    参数:
        thread: 设定的阈值，绘制预测框的时候的概率阈值
            也是写入预测结果的阈值
    """
    if os.path.exists(voc_results_dir):
        shutil.rmtree(voc_results_dir)
    os.makedirs(voc_results_dir)
    print("====将预测结果写入%s====" % voc_results_dir, end="\r")

    # 建立存储预测图片及其预测框的路径
    # os.makedirs(OPT.output_img_dir, exist_ok=True)
    # 绘制预测框的时候的颜色，由于只有一类，可以根据类别定义
    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for cls_ind, cls in enumerate(classes):
        new_image_index, all_boxes[cls_ind] = \
                zip(*sorted(zip(image_index, all_boxes[cls_ind]),
                            key=lambda x: x[0]))

        if cls == "__background__":
            continue

        images_dir = data_loader.dataset.image_dir
        # 要存储的该类预测结果的结果输出路径
        filename = os.path.join(
            voc_results_dir, ("det_test_{:s}.txt").format(cls)
        )

        print("====写入预测结果====")
        start = time.time()
        with open(filename, 'wt') as f:
            prev_index = ''
            for im_ind, index in enumerate(new_image_index):
                # 按照图片索引读取图片
                # img = cv2.imread(
                #     os.path.join(
                #         images_dir, index + OPT.image_format_extension
                #     )
                # )
                # h, w, _ = img.shape

                # 检查重复的输入并删除
                if prev_index == index: continue
                prev_index = index
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                dets = dets[0]

                for k in range(dets.shape[0]):
                    # 遍历该图片上的该类预测框
                    f.write(
                        "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".\
                        format(
                            index, dets[k, -1],
                            dets[k, 0] + 1, dets[k, 1] + 1,
                            dets[k, 2] + 1, dets[k, 3] + 1
                        )
                    )
                    if dets[k, -1] < thread:
                        continue
        end = time.time()
        print('====写入预测结果结束====，用时: %.4f' % (end-start))
                    # x1, x2 = dets[k, 0], dets[k, 2]
                    # y1, y2 = dets[k, 1], dets[k, 3]

                    # color = colors[cls_ind]
                    # thick = int((h+w)/300)  # 框的线条粗细
                    # cv2.rectangle(
                    #     img, (x1, y1), (x2, y2),
                    #     color, thick
                    # )
                    # message = '%s: %.3f' % (cls, dets[k, -1])
                    # cv2.putText(
                    #     img, message, (x1, y1-12),
                    #     0, 1e-3 * h, color, thick // 3
                    # )
                # filename = index
                # cv2.imwrite(
                    # os.path.join(
                    #     OPT.output_img_dir, 'output-%s.png' % filename
                    # ), img
                # )


def do_python_eval(data_loader, voc_results_dir,
                   use_07_metric=False,
                   savefig_flag=False):

    imagesetfile = os.path.join(
        data_loader.dataset.voc_imgset_file,
        data_loader.dataset.image_set+".txt"
    )
    annopath = os.path.join(
        data_loader.dataset.annotation_dir, '{:s}.xml'
    )

    # 类别名
    classes = data_loader.dataset.classes
    aps = []  # ap值
    fig = plt.figure()

    for cls in classes:
        if cls == "__background__":
            continue
        filename = voc_results_dir + '/det_test_{:s}.txt'.format(cls)
        # 该类的ap, recall, precision
        rec, prec, ap = custom_voc_eval(
            cls, filename, imagesetfile, annopath,
            ovthresh=0.5, use_07_metric=use_07_metric
        )
        print("+ Class {} - AP: {}".format(cls, ap))
        plt.plot(rec, prec, label='{}'.format(cls))
        aps += [ap]
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend()
    if savefig_flag:
        plt.savefig(os.path.join(voc_results_dir,'pr_curve'), format='png',
                   transparent=True, dpi=300, pad_inches=0)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    return np.mean(aps), fig
    

if __name__ == "__main__":
    parse_rec(
        "/home/stat-caolei/code/FasterDetection/data/VOC2007_/Annotations/1792bj129_0091.xml"
    )
    import ipdb;ipdb.set_trace()
