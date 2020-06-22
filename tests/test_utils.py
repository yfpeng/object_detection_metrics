import json
from typing import List, Dict

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


def test_results(actuals: Dict[str, MetricPerClass], expecteds, key, classes=None):
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

