import json
from typing import List, Tuple

from podm import BoundingBox


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

    # categroies
    label_map = {c['id']: c['name'] for c in objs['categories']}
    # images
    image_map = {c['id']: c['file_name'] for c in objs['images']}

    gt_boxes = []
    for box in objs['annotations']:
        bb = BoundingBox(
            image_map[box['image_id']],
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
                image_map[box['image_id']],
                label_map[box['category_id']],
                box['bbox'][0],
                box['bbox'][1],
                box['bbox'][0] + box['bbox'][2],
                box['bbox'][1] + box['bbox'][3],
                box['score']
            )
            rt_boxes.append(bb)

    return gt_boxes, rt_boxes