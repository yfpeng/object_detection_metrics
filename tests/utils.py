import json
from typing import List, Dict, Tuple

from podm.podm import BoundingBox, MetricPerClass
import numpy as np


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def load_data(pathname) -> List[BoundingBox]:
    with open(pathname, 'r') as fp:
        objs = json.load(fp)

    boxes = []
    for fig in objs:
        for box in fig['boxes']:
            if 'score' in box:
                score = box['score']
            else:
                score = 1
            bb = BoundingBox(
                fig['name'],
                box['label'],
                box['xtl'],
                box['ytl'],
                box['xbr'],
                box['ybr'],
                score)
            boxes.append(bb)
    return boxes


def load_data_coco(gd_pathname, rt_pathname=None) -> Tuple[List[BoundingBox], List[BoundingBox]]:
    with open(gd_pathname) as fp:
        objs = json.load(fp)

    label_map ={c['id']: c['name'] for c in objs['categories']}

    gt_boxes = []
    for box in objs['annotations']:
        bb = BoundingBox(
            box['image_id'],
            label_map[box['category_id']],
            box['bbox'][0],
            box['bbox'][1],
            box['bbox'][0] + box['bbox'][2],
            box['bbox'][1] + box['bbox'][3])
        gt_boxes.append(bb)

    rt_boxes = []
    if rt_pathname is not None:
        with open(rt_pathname) as fp:
            objs = json.load(fp)

        for box in objs:
            bb = BoundingBox(
                box['image_id'],
                label_map[box['category_id']],
                box['bbox'][0],
                box['bbox'][1],
                box['bbox'][0] + box['bbox'][2],
                box['bbox'][1] + box['bbox'][3],
                box['score']
            )
            rt_boxes.append(bb)

    return gt_boxes, rt_boxes


def assert_results(actuals: Dict[str, MetricPerClass], expecteds, key, classes=None):
    if classes is None:
        classes = set(m.label for m in actuals.values())

    for m in actuals.values():
        label = m.label
        if label in classes:
            print(f'{label}, {key}: ', end='')
            actual = m.__dict__[key]
            expected = expecteds[label][key]
            try:
                if np.allclose(actual, expected, rtol=1e-1, equal_nan=True):
                    print(f'{bcolors.OKGREEN}Passed{bcolors.ENDC}')
                else:
                    print(f'{bcolors.FAIL}Failed. Expected:{expected}, Actual:{actual}{bcolors.ENDC}')
                    exit(1)
            except Exception as e:
                print(f'{bcolors.FAIL}Failed. Expected:{expected}, Actual:{actual}{bcolors.ENDC}')
                exit(1)

