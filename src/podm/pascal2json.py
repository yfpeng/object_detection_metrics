"""
Convert from a PASCAL VOC zip file to one json file.

Usage:
    pascal2json --input=<file> --output=<file>

Options:
    --input=<file>      PASCAL VOC zip file
    --output=<file>     JSON file
"""
import io
import zipfile
from enum import Enum
from pathlib import Path
import os
import json
from typing import Iterator

import docopt
import tqdm


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


def parse_single_file(items_file: Iterator[str], format: BBFormat):
    boxes = []
    for i, line in enumerate(items_file):
        tok = line.strip().split(' ')
        if len(tok) == 6:
            b = {
                'xtl': int(tok[2]),
                'ytl': int(tok[3]),
                'xbr': int(tok[4]),
                'ybr': int(tok[5]),
                'label': tok[0],
                'score': float(tok[1])
            }
        elif len(tok) == 5:
            b = {
                'xtl': int(tok[1]),
                'ytl': int(tok[2]),
                'xbr': int(tok[3]),
                'ybr': int(tok[4]),
                'label': tok[0],
            }
        else:
            raise ValueError('%s: ill-formatted. Should have 5 or 6 field. %s' % i, line)
        if format == BBFormat.XYWH:
            b['xbr'] += b['xtl']
            b['ybr'] += b['ytl']
        boxes.append(b)
    return boxes


def convert_pascal_voc_to_json_zip(src, dest, format=BBFormat.X1Y1X2Y2):
    data = []
    with zipfile.ZipFile(src, 'r') as myzip:
        namelist = myzip.namelist()
        for name in tqdm.tqdm(namelist):
            if not name.endswith('.txt'):
                continue
            x = {
                'name': name,
                'boxes': []
            }
            with myzip.open(name, 'r') as fp:
                items_file = io.TextIOWrapper(fp)
                boxes = parse_single_file(items_file, format)
                x['boxes'].extend(boxes)
            data.append(x)

    with open(dest, 'w') as fp:
        json.dump(data, fp, indent=2)


def convert_pascal_voc_to_json(dirname, dest, format: BBFormat = BBFormat.X1Y1X2Y2):
    data = []
    for entry in os.scandir(dirname):
        x = {
            'name': Path(entry.path).stem,
            'boxes': []
        }
        with open(entry.path) as fp:
            boxes = parse_single_file(fp, format)
            x['boxes'].extend(boxes)
        data.append(x)

    with open(dest, 'w') as fp:
        json.dump(data, fp, indent=2)


def main():
    argv = docopt.docopt(__doc__)
    convert_pascal_voc_to_json_zip(argv['--input'], argv['--output'], BBFormat.X1Y1X2Y2)


if __name__ == '__main__':
    main()
