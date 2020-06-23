import sys
from collections import Counter
from enum import Enum
from typing import List, Dict

import numpy as np


class Box:
    def __init__(self, xtl: float, ytl: float, xbr: float, ybr: float):
        """
                    0,0 ------> x (width)
             |
             |  (Left,Top)
             |      *_________
             |      |         |
                    |         |
             y      |_________|
          (height)            *
                        (Right,Bottom)

        Args:
            xtl: the X top-left coordinate of the bounding box.
            ytl: the Y top-left coordinate of the bounding box.
            xbr: the X bottom-right coordinate of the bounding box.
            ybr: the Y bottom-right coordinate of the bounding box.
        """
        assert xtl < xbr, f'xtl < xbr: xtl:{xtl}, xbr:{xbr}'
        assert ytl < ybr, f'ytl < ybr: xtl:{ytl}, xbr:{ybr}'

        self.xtl = xtl
        self.ytl = ytl
        self.xbr = xbr
        self.ybr = ybr

    @property
    def width(self) -> float:
        return self.xbr - self.xtl

    @property
    def height(self) -> float:
        return self.ybr - self.ytl

    @property
    def area(self) -> float:
        return (self.xbr - self.xtl) * (self.ybr - self.ytl)

    def __str__(self):
        return 'Box[xtl={},ytl={},xbr={},ybr={}]'.format(self.xtl, self.ytl, self.xbr, self.ybr)

    @classmethod
    def intersection_over_union(cls, box1: 'Box', box2: 'Box') -> float:
        """
        Intersection Over Union (IOU) is measure based on Jaccard Index that evaluates the overlap between
        two bounding boxes.
        """
        # if boxes dont intersect
        if not Box.is_intersecting(box1, box2):
            return 0
        intersection = Box.intersection_area(box1, box2)
        union = Box.union_areas(box1, box2, intersection_area=intersection)
        # intersection over union
        iou = intersection / union
        assert iou >= 0, '{} = {} / {}, box1={}, box2={}'.format(iou, intersection, union, box1, box2)
        return iou

    @classmethod
    def is_intersecting(cls, box1: 'Box', box2: 'Box') -> bool:
        if box1.xtl > box2.xbr:
            return False  # boxA is right of boxB
        if box2.xtl > box1.xbr:
            return False  # boxA is left of boxB
        if box1.ybr < box2.ytl:
            return False  # boxA is above boxB
        if box1.ytl > box2.ybr:
            return False  # boxA is below boxB
        return True

    @classmethod
    def intersection_area(cls, box1: 'Box', box2: 'Box') -> float:
        xtl = max(box1.xtl, box2.xtl)
        ytl = max(box1.ytl, box2.ytl)
        xbr = min(box1.xbr, box2.xbr)
        ybr = min(box1.ybr, box2.ybr)
        # intersection area
        return (xbr - xtl) * (ybr - ytl)

    @staticmethod
    def union_areas(box1: 'Box', box2: 'Box', intersection_area: float = None) -> float:
        if intersection_area is None:
            intersection_area = Box.intersection_area(box1, box2)
        return box1.area + box2.area - intersection_area


class MethodAveragePrecision(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.

        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    AllPointsInterpolation = 1
    ElevenPointsInterpolation = 2


class BoundingBox(Box):
    def __init__(self, image_name: str, label: str, xtl: float, ytl: float, xbr: float, ybr: float,
                 score=None):
        """Constructor.
        Args:
            image_name: the image name.
            label: class id.
            xtl: the X top-left coordinate of the bounding box.
            ytl: the Y top-left coordinate of the bounding box.
            xbr: the X bottom-right coordinate of the bounding box.
            ybr: the Y bottom-right coordinate of the bounding box.
            score: (optional) the confidence of the detected class.
        """
        super().__init__(xtl, ytl, xbr, ybr)
        self.image_name = image_name
        self.score = score
        self.label = label


class MetricPerClass:
    def __init__(self):
        self.label = None
        self.precision = None
        self.recall = None
        self.ap = None
        self.interpolated_precision = None
        self.interpolated_recall = None
        self.num_groundtruth = None
        self.num_detection = None
        self.tp = None
        self.fp = None

    @staticmethod
    def mAP(results: Dict[str, 'MetricPerClass']):
        return np.average([m.ap for m in results.values() if m.num_groundtruth > 0])


def get_pascal_voc_metrics(gold_standard: List[BoundingBox],
                           predictions: List[BoundingBox],
                           iou_threshold: float = 0.5,
                           method: MethodAveragePrecision = MethodAveragePrecision.AllPointsInterpolation
                           ) -> Dict[str, MetricPerClass]:
    """Get the metrics used by the VOC Pascal 2012 challenge.

    Args:
        gold_standard: ground truth bounding boxes;
        predictions: detected bounding boxes;
        iou_threshold: IOU threshold indicating which detections will be considered TP or FP (default value = 0.5);
        method: It can be calculated as the implementation in the official PASCAL VOC toolkit (EveryPointInterpolation),
            or applying the 11-point interpolation as described in the paper "The PASCAL Visual Object Classes(VOC)
            Challenge" or AllPointsInterpolation" (ElevenPointInterpolation);
    Returns:
        A dictionary containing metrics of each class.
    """
    ret = {}  # list containing metrics (precision, recall, average precision) of each class

    # Get all classes
    classes = sorted(set(b.label for b in gold_standard + predictions))

    # Precision x Recall is obtained individually by each class
    # Loop through by classes
    for c in classes:
        preds = [b for b in predictions if b.label == c]  # type: List[BoundingBox]
        golds = [b for b in gold_standard if b.label == c]  # type: List[BoundingBox]
        npos = len(golds)

        # sort detections by decreasing confidence
        preds = sorted(preds, key=lambda b: b.score, reverse=True)
        tps = np.zeros(len(preds))
        fps = np.zeros(len(preds))

        # create dictionary with amount of gts for each image
        counter = Counter([cc.image_name for cc in golds])
        for key, val in counter.items():
            counter[key] = np.zeros(val)

        # Loop through detections
        for i in range(len(preds)):
            # Find ground truth image
            gt = [b for b in golds if b.image_name == preds[i].image_name]
            max_iou = sys.float_info.min
            mas_idx = -1
            for j in range(len(gt)):
                iou = Box.intersection_over_union(preds[i], gt[j])
                if iou > max_iou:
                    max_iou = iou
                    mas_idx = j
            # Assign detection as true positive/don't care/false positive
            if max_iou >= iou_threshold:
                if counter[preds[i].image_name][mas_idx] == 0:
                    tps[i] = 1  # count as true positive
                    counter[preds[i].image_name][mas_idx] = 1  # flag as already 'seen'
                else:
                    # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
                    fps[i] = 1  # count as false positive
            else:
                fps[i] = 1  # count as false positive
        # compute precision, recall and average precision
        cumulative_fps = np.cumsum(fps)
        cumulative_tps = np.cumsum(tps)
        recalls = np.divide(cumulative_tps, npos, out=np.full_like(cumulative_tps, np.nan), where=npos != 0)
        precisions = np.divide(cumulative_tps, (cumulative_fps + cumulative_tps))
        # Depending on the method, call the right implementation
        if method == MethodAveragePrecision.AllPointsInterpolation:
            ap, mpre, mrec, _ = calculate_all_points_average_precision(recalls, precisions)
        else:
            ap, mpre, mrec = calculate_11_points_average_precision(recalls, precisions)
        # add class result in the dictionary to be returned
        r = MetricPerClass()
        r.label = c
        r.precision = precisions
        r.recall = recalls
        r.ap = ap
        r.interpolated_recall = np.array(mrec)
        r.interpolated_precision = np.array(mpre)
        r.tp = np.sum(tps)
        r.fp = np.sum(fps)
        r.num_groundtruth = len(golds)
        r.num_detection = len(preds)
        ret[c] = r
    return ret


def calculate_all_points_average_precision(recall, precision):
    """
    All-point interpolated average precision

    Returns:
        average precision
        interpolated precision
        interpolated recall
        interpolated points
    """
    mrec = [0] + [e for e in recall] + [1]
    mpre = [0] + [e for e in precision] + [0]
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii


def calculate_11_points_average_precision(recall, precision):
    """
    11-point interpolated average precision

    Returns:
        average precision
        interpolated precision
        interpolated recall
    """
    mrec = [e for e in recall]
    mpre = [e for e in precision]
    recall_values = np.linspace(0, 1, 11)
    recall_values = list(recall_values[::-1])
    rho_interp = []
    recall_valid = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in recall_values:
        # Obtain all recall values higher or equal than r
        arg_greater_recalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        # If there are recalls above r
        if arg_greater_recalls.size != 0:
            pmax = max(mpre[arg_greater_recalls.min():])
        recall_valid.append(r)
        rho_interp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rho_interp) / 11
    # Generating values for the plot
    rvals = [recall_valid[0]] + [e for e in recall_valid] + [0]
    pvals = [0] + [e for e in rho_interp] + [0]
    # rhoInterp = rhoInterp[::-1]
    cc = []
    for i in range(len(rvals)):
        p = (rvals[i], pvals[i - 1])
        if p not in cc:
            cc.append(p)
        p = (rvals[i], pvals[i])
        if p not in cc:
            cc.append(p)
    recall_values = [i[0] for i in cc]
    rho_interp = [i[1] for i in cc]
    return ap, rho_interp, recall_values
