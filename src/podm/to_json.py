from enum import Enum
from pathlib import Path
import os
import json


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


def convert_pascal_voc_to_json(dirname, dest, format=BBFormat.X1Y1X2Y2):
    data = []
    for entry in os.scandir(dirname):
        name = Path(entry.path).stem
        x = {
            'name': name,
            'boxes': []
        }
        with open(entry.path) as fp:
            for line in fp:
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
                if format == BBFormat.XYWH:
                    b['xbr'] += b['xtl']
                    b['ybr'] += b['ytl']
                x['boxes'].append(b)
        data.append(x)

    with open(dest, 'w') as fp:
        json.dump(data, fp, indent=2)


if __name__ == '__main__':
    dir = Path('tests/sample_2')
    convert_pascal_voc_to_json(dir / 'groundtruths', dir / 'groundtruths.json', BBFormat.X1Y1X2Y2)
    convert_pascal_voc_to_json(dir / 'detections', dir / 'detections.json', BBFormat.X1Y1X2Y2)
