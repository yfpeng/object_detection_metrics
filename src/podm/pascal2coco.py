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
import io
import warnings

import docopt
import tqdm
import pandas as pd
from podm import coco_encoder
from podm.box import BBFormat, Box
from podm.coco import PCOCOBoundingBoxDataset, PCOCOImage, PCOCOCategory, PCOCOBoundingBox


def convert_pascal_to_df(src):
    rows = []
    with zipfile.ZipFile(src, 'r') as myzip:
        namelist = myzip.namelist()
        for name in tqdm.tqdm(namelist):
            if not name.endswith('.txt'):
                continue
            with myzip.open(name, 'r') as fp:
                name = name[name.find('/') + 1:]
                items_file = io.TextIOWrapper(fp)
                for line in items_file:
                    toks = line.strip().split(' ')
                    if len(toks) == 5:
                        row = {
                            'name': name,
                            'label': toks[0],
                            'xtl': int(toks[1]),
                            'ytl': int(toks[2]),
                            'xbr': int(toks[3]),
                            'ybr': int(toks[4])
                        }
                    elif len(toks) == 6:
                        row = {
                            'name': name,
                            'label': toks[0],
                            'score': float(toks[1]),
                            'xtl': int(toks[2]),
                            'ytl': int(toks[3]),
                            'xbr': int(toks[4]),
                            'ybr': int(toks[5])
                        }
                    else:
                        raise ValueError
                    rows.append(row)
    return pd.DataFrame(rows)


class PascalVoc2COCO:
    def __init__(self, format: BBFormat = BBFormat.X1Y1X2Y2):
        self.format = format

    def convert_gold(self, src) -> PCOCOBoundingBoxDataset:
        df = convert_pascal_to_df(src)

        dataset = PCOCOBoundingBoxDataset()
        # add image
        for i, name in enumerate(df['name'].unique()):
            img = PCOCOImage()
            img.id = i
            img.file_name = name
            dataset.add_image(img)
        # add category
        for i, label in enumerate(df['label'].unique()):
            cat = PCOCOCategory()
            cat.id = i
            cat.name = label
            dataset.add_category(cat)
        # add annotation
        for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            box = Box.of_box(row['xtl'], row['ytl'], row['xbr'], row['ybr'])
            if self.format == BBFormat.XYWH:
                box.xbr += box.xtl
                box.ybr += box.ytl
            ann = PCOCOBoundingBox()
            ann.image_id = dataset.get_image_name(row['name']).id
            ann.id = i
            ann.category_id = dataset.get_category_name(row['label']).id
            ann.set_box(box)
            dataset.add_annotation(ann)
        return dataset

    def convert_gold_file(self, src, dest):
        dataset = self.convert_gold(src)
        with open(dest, 'w') as fp:
            coco_encoder.dump(dataset, fp)

    def convert_gold_pred(self, src_gold, src_pred):
        gold_dataset = self.convert_gold(src_gold)

        df = convert_pascal_to_df(src_pred)
        # check cat
        subrows = []
        for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            if gold_dataset.get_category_name(row['label']) is None:
                warnings.warn('%s: Category does not exist' % row['label'])
                continue
            if gold_dataset.get_image_name(row['name']) is None:
                warnings.warn('%s: Image does not exist' % row['name'])
                continue
            subrows.append(row)
        if len(subrows) < len(df):
            warnings.warn('Remove %s rows' % (len(df) - len(subrows)))

        annotations = []
        for i, row in tqdm.tqdm(enumerate(subrows), total=len(subrows)):
            box = Box.of_box(row['xtl'], row['ytl'], row['xbr'], row['ybr'])
            if self.format == BBFormat.XYWH:
                box.xbr += box.xtl
                box.ybr += box.ytl
            ann = PCOCOBoundingBox()
            ann.image_id = gold_dataset.get_image_name(row['name']).id
            ann.id = i
            ann.category_id = gold_dataset.get_category_name(row['label']).id
            ann.score = row['score']
            ann.set_box(box)
            annotations.append(ann)

        pred_dataset = gold_dataset.get_new_dataset(annotations)
        return gold_dataset, pred_dataset

    def convert_gold_pred_file(self, src_gold, src_pred, dest_gold, dest_pred):
        gold_dataset, pred_dataset = self.convert_gold_pred(src_gold, src_pred)
        with open(dest_gold, 'w') as fp:
            coco_encoder.dump(gold_dataset, fp)

        with open(dest_pred, 'w') as fp:
            json.dump(pred_dataset.annotations, fp, cls=coco_encoder.PCOCOJSONEncoder, indent=2)


def main():
    argv = docopt.docopt(__doc__)
    converter = PascalVoc2COCO(BBFormat.X1Y1X2Y2)
    if argv['gold']:
        converter.convert_gold_file(argv['--gold'], argv['--output-gold'])
    if argv['pred']:
        converter.convert_gold_pred_file(argv['--gold'], argv['--pred'], argv['--output-gold'], argv['--output-pred'])


if __name__ == '__main__':
    main()
