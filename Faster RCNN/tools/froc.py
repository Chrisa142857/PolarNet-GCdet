import argparse
from collections import namedtuple
import numpy as np
from skimage.measure import points_in_poly
import matplotlib.pyplot as plt


Object = namedtuple('Object',
                    ['image_path', 'object_id', 'object_type', 'coordinates',
                     'hit_flag'])
Prediction = namedtuple('Prediction',
                        ['image_path', 'probability', 'coordinates'])


parser = argparse.ArgumentParser(description='Compute FROC')
parser.add_argument('gt_csv', default=None, metavar='GT_CSV',
                    type=str, help="Path to the ground truch csv file")
parser.add_argument('pred_csv', default=None, metavar='PRED_PATH',
                    type=str, help="Path to the predicted csv file")
parser.add_argument('--fps', default='0.125,0.25,0.5,1,2,4,8', type=str,
                    help='False positives per image to compute FROC, comma '
                    'seperated, default "0.125,0.25,0.5,1,2,4,8"')


def inside_object(pred, obj):
    # bounding box
    if obj.object_type == '0':
        x1, y1, x2, y2 = obj.coordinates
        x, y = pred.coordinates
        return x1 <= x <= x2 and y1 <= y <= y2
    # bounding ellipse
    if obj.object_type == '1':
        x1, y1, x2, y2 = obj.coordinates
        x, y = pred.coordinates
        x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
        x_axis, y_axis = (x2 - x1) / 2, (y2 - y1) / 2
        return ((x - x_center)/x_axis)**2 + ((y - y_center)/y_axis)**2 <= 1
    # mask/polygon
    if obj.object_type == '2':
        num_points = len(obj.coordinates) // 2
        poly_points = obj.coordinates.reshape(num_points, 2, order='C')
        return points_in_poly(pred.coordinates.reshape(1, 2), poly_points)[0]


def main():
    args = parser.parse_args()

    # iou overlap threshold, we set 0.5
    ovthresh = 0.5
    # parse ground truth csv
    num_image = 0
    num_object = 0
    object_dict = {}
    with open(args.gt_csv) as f:
        # header
        next(f)
        for line in f:
            image_path, annotation = line.strip('\n').split(',')

            if annotation == '':
                num_image += 1
                continue

            object_annos = annotation.split(';')
            for object_anno in object_annos:
                fields = object_anno.split(' ')
                object_type = fields[0]
                coords = np.array(list(map(float, fields[1:])))
                hit_flag = False
                obj = Object(image_path, num_object, object_type, coords,
                             hit_flag)
                if image_path in object_dict:
                    object_dict[image_path].append(obj)
                else:
                    object_dict[image_path] = [obj]
                num_object += 1
            num_image += 1

    # parse prediction truth csv
    preds = []
    with open(args.pred_csv) as f:
        # header
        next(f)
        for line in f:
            image_path, prediction = line.strip('\n').split(',')

            if prediction == '':
                continue

            coord_predictions = prediction.split(';')
            for coord_prediction in coord_predictions:
                fields = coord_prediction.split(' ')
                probability, x1, y1, x2, y2 = list(map(float, fields))
                pred = Prediction(image_path, probability,
                                  np.array([x1, y1, x2, y2]))
                preds.append(pred)

    # sort prediction by probabiliyt
    preds = sorted(preds, key=lambda x: x.probability, reverse=True)

    # compute hits and false positives
    hits = 0
    false_positives = 0
    fps_idx = 0
    object_hitted = set()
    # fps = list(map(float, args.fps.split(',')))
    # fps = [0.125,0.25,0.5,1.,2.]
    # fps = list(np.arange(0.1, 2.1, 0.1))
    # fps = list(np.arange(0, 2.1, 0.1))
    fps = list(np.arange(0, 3.1, 0.1))
    # fps = np.arang(0,4.5,0.5).tolist()
    fps_flag = [str(i) for i in fps]
    froc = []
    for i in range(len(preds)):
        pred = preds[i]
        if pred.image_path in object_dict:
            objs = object_dict[pred.image_path]
            # gt boxes coords in this image
            BBGT = np.array([k.coordinates for k in objs]).astype(float)
            # gt boxes hit flag
            BBGT_HIT_FLAG = [k.hit_flag for k in objs]
            # this pred box coord
            bb = pred.coordinates.astype(float)
            # set the initial max iou
            ovmax = -np.inf

            if BBGT.size > 0:
                # calculate ious between this box and every gt boxes
                # on this image
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])

                # calculate overlap area width
                iw = np.maximum(ixmax-ixmin+1., 0.)
                # overlap area height
                ih = np.maximum(iymax-iymin+1., 0.)
                # overlap areas
                inters = iw * ih
                ## calculate ious
                # union
                union = ((bb[2]-bb[0]+1.)*(bb[3]-bb[1]+1.)+\
                        (BBGT[:, 2]-BBGT[:, 0]+1.)*(BBGT[:, 3]-BBGT[:, 1]+1.)-\
                        inters)
                # ious
                overlaps = inters / union
                # find the maximum iou
                ovmax = np.max(overlaps)
                # find the maximum iou index
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not BBGT_HIT_FLAG[jmax]:
                    BBGT_HIT_FLAG[jmax] = True
                    hits += 1
                else:
                    false_positives += 1
            else:
                false_positives += 1

            if false_positives / num_image >= fps[fps_idx]:
                sensitivity = hits / float(num_object)
                froc.append(sensitivity)
                fps_idx += 1

                if len(fps) == len(froc):
                    break
                
    while len(froc) < len(fps):
        froc.append(froc[-1])

    print("False positives per image:")
    print("\t".join(fps_flag))
    print("Sensitivity:")
    print("\t".join(map(lambda x: "{:.3f}".format(x), froc)))
    print("FROC:")
    print(np.mean(froc))
    fig, ax = plt.subplots(
        subplot_kw=dict(xlim=[0, 3.], ylim=[0, 1], aspect="equal"),
        figsize=(6,6)
    )
    ax.plot(fps, froc)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(linestyle="dashed")
    fig.savefig("xiugao_att169_froc.svg", dpi=300, format="svg")


if __name__ == '__main__':
    main()
