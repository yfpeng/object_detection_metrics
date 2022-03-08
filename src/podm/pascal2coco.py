"""
Convert from a PASCAL VOC zip file to a COCO file.

Usage:
    pascal2coco gold --gold=<file> --output-gold=<file>
    pascal2coco pred --gold=<file> --pred=<file> --output-gold=<file> --output-pred=<file>

Options:
    --gold=<file>           PASCAL VOC groundtruths zip file
    --pred=<file>           PASCAL VOC predictions zip file
    --output-gold=<file>    Groundtruths JSON file
    --output-pred=<file>    Predictions JSON file
"""
import json
import zipfile
from enum import Enum
from typing import Dict
import io
import docopt
import tqdm

from pcoco import PCOCOImage, PCOCODataset, PCOCOCategory, PCOCOAnnotation, load as pcoco_load


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
        for name in tqdm.tqdm(namelist):
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


def convert_pascal_voc_to_coco_pred(src_gold, src_pred, dest_gold, dest_pred, format=BBFormat.X1Y1X2Y2):
    convert_pascal_voc_to_coco_gold(src_gold, dest_gold)

    with open(dest_gold) as fp:
        dataset = pcoco_load(fp)

    cat_map = dataset.cat_name_to_id
    img_map = dataset.img_name_to_id

    annotations = []
    ann_id = 1
    with zipfile.ZipFile(src_pred, 'r') as myzip:
        namelist = myzip.namelist()
        for name in tqdm.tqdm(namelist):
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

                    ann_id += 1

            image_id += 1

    with open(dest_pred, 'w') as fp:
        json.dump([a.to_dict() for a in annotations], fp, indent=2)

    with open(dest_gold, 'w') as fp:
        json.dump(dataset.to_dict(), fp, indent=2)


def main():
    argv = docopt.docopt(__doc__)
    if argv['gold']:
        convert_pascal_voc_to_coco_gold(argv['--gold'], argv['--output-gold'], BBFormat.X1Y1X2Y2)
    if argv['pred']:
        convert_pascal_voc_to_coco_pred(argv['--gold'], argv['--pred'],
                                        argv['--output-gold'], argv['--output-pred'],
                                        BBFormat.X1Y1X2Y2)


if __name__ == '__main__':
    main()
