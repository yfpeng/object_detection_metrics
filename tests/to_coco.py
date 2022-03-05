import json
import zipfile
from enum import Enum
from pathlib import Path
from typing import Dict
import io

from pycocotools.coco import COCO

from src.podm import PCOCOImage, PCOCODataset, PCOCOCategory, PCOCOAnnotation
from src.podm import pcoco


class BBFormat(Enum):
    """
    Class representing the format of a bounding box.
    It can be (X,Y,width,height) => XYWH
    or (X1,Y1,X2,Y2) => XYX2Y2

        Developed by: Rafael Padilla
        Last modification: May 24 2018
    """
    XYWH = 1
    X1Y1X2Y2 = 2


def convert_pascal_voc_to_coco_gold(src, dest, format=BBFormat.X1Y1X2Y2):
    image_id = 1
    ann_id = 1
    cat_map = {}  # type: Dict[str, int]

    dataset = PCOCODataset()
    with zipfile.ZipFile(src, 'r') as myzip:
        namelist = myzip.namelist()
        for name in namelist:
            if not name.endswith('.txt'):
                continue
            img = PCOCOImage()
            img.id = image_id
            img.file_name = name[name.find('/')+1:] + '.jpg'
            dataset.images.append(img)

            with myzip.open(name,'r') as fp:
                items_file = io.TextIOWrapper(fp)
                for line in items_file:
                    tok = line.strip().split(' ')
                    if len(tok) == 5:
                        xtl = int(tok[1])
                        ytl = int(tok[2])
                        xbr = int(tok[3])
                        ybr = int(tok[4])
                        label = tok[0]
                    else:
                        raise ValueError

                    if format == BBFormat.XYWH:
                        xbr += xtl
                        ybr += ytl

                    if label not in cat_map:
                        cat_map[label] = len(cat_map) + 1
                        cat = PCOCOCategory()
                        cat.id = cat_map[label]
                        cat.name = label
                        dataset.categories.append(cat)

                    ann = PCOCOAnnotation()
                    ann.image_id = image_id
                    ann.id = ann_id
                    ann.category_id = cat_map[label]
                    ann.xtl = xtl
                    ann.ytl = ytl
                    ann.xbr = xbr
                    ann.ybr = ybr
                    dataset.annotations.append(ann)

                    ann_id += 1

            image_id += 1

    with open(dest, 'w') as fp:
        json.dump(dataset.to_dict(), fp, indent=2)



def convert_pascal_voc_to_coco_pred(coco_gold_file, src, dest, format=BBFormat.X1Y1X2Y2):
    with open(coco_gold_file) as fp:
        dataset = pcoco.load(fp)

    cat_map = dataset.cat_name_to_id
    img_map = dataset.img_name_to_id

    annotations = []
    ann_id = 1
    with zipfile.ZipFile(src, 'r') as myzip:
        namelist = myzip.namelist()
        for name in namelist:
            if not name.endswith('.txt'):
                continue
            file_name = name[name.find('/') + 1:] + '.jpg'
            image_id = img_map[file_name]

            with myzip.open(name,'r') as fp:
                items_file = io.TextIOWrapper(fp)
                for line in items_file:
                    tok = line.strip().split(' ')
                    if len(tok) == 6:
                        xtl = int(tok[2])
                        ytl = int(tok[3])
                        xbr = int(tok[4])
                        ybr = int(tok[5])
                        label = tok[0]
                        score = float(tok[1])
                    else:
                        raise ValueError

                    if format == BBFormat.XYWH:
                        xbr += xtl
                        ybr += ytl

                    if label not in cat_map:
                        cat_map[label] = len(cat_map) + 1
                        cat = PCOCOCategory()
                        cat.id = cat_map[label]
                        cat.name = label
                        dataset.categories.append(cat)
                    try:
                        ann = PCOCOAnnotation()
                        ann.image_id = image_id
                        ann.id = ann_id
                        ann.category_id = cat_map[label]
                        ann.xtl = xtl
                        ann.ytl = ytl
                        ann.xbr = xbr
                        ann.ybr = ybr
                        ann.score = score
                        annotations.append(ann)
                    except:
                        print(label)
                    ann_id += 1

            image_id += 1

    with open(dest, 'w') as fp:
        json.dump([a.to_dict() for a in annotations], fp, indent=2)

    with open(coco_gold_file, 'w') as fp:
        json.dump(dataset.to_dict(), fp, indent=2)


if __name__ == '__main__':
    dir = Path('tests/sample_2')
    convert_pascal_voc_to_coco_gold(dir / 'groundtruths.zip',
                                    dir / 'groundtruths_coco.json',
                                    BBFormat.X1Y1X2Y2)
    convert_pascal_voc_to_coco_pred(dir / 'groundtruths_coco.json',
                                    dir / 'detections.zip',
                                    dir / 'detections_coco.json',
                                    BBFormat.X1Y1X2Y2)