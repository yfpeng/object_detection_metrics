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
import docopt
import tqdm

from podm.pcoco import PCOCOImage, PCOCOObjectDetectionDataset, PCOCOCategory, PCOCOAnnotation, PCOCOObjectDetection, \
    PCOCOObjectDetectionResult
from podm.box import BBFormat, Box
from podm.pcoco_encoder import dump as pcoco_dump, PCOCOJSONEncoder


def convert_pascal_voc_to_coco_gold(src, dest, format=BBFormat.X1Y1X2Y2) -> PCOCOObjectDetectionDataset:
    image_id = 1
    ann_id = 1
    cat_id = 1

    dataset = PCOCOObjectDetectionDataset()
    with zipfile.ZipFile(src, 'r') as myzip:
        namelist = myzip.namelist()

        # add image
        for name in tqdm.tqdm(namelist):
            if not name.endswith('.txt'):
                continue
            img = PCOCOImage()
            img.id = image_id
            img.file_name = name[name.find('/')+1:] + '.jpg'
            dataset.add_image(img)
            image_id += 1

            # add annotation
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

                    # add cat
                    cat = dataset.get_category_name(label)
                    if cat is None:
                        cat = PCOCOCategory()
                        cat.id = cat_id
                        cat.name = label
                        dataset.add_category(cat)
                        cat_id += 1

                    ann = PCOCOObjectDetection()
                    ann.image_id = img.id
                    ann.id = ann_id
                    ann.category_id = cat.id
                    ann.add_box(Box(xtl, ytl, xbr, ybr))
                    dataset.add_annotation(ann)

                    ann_id += 1

            image_id += 1

    with open(dest, 'w') as fp:
        pcoco_dump(dataset, fp)

    return dataset


def convert_pascal_voc_to_coco_pred(src_gold, src_pred, dest_gold, dest_pred, format=BBFormat.X1Y1X2Y2):
    gold_dataset = convert_pascal_voc_to_coco_gold(src_gold, dest_gold)
    cat_id = gold_dataset.get_max_category_id() + 1

    annotations = []
    ann_id = 1
    with zipfile.ZipFile(src_pred, 'r') as myzip:
        namelist = myzip.namelist()
        for name in tqdm.tqdm(namelist):
            if not name.endswith('.txt'):
                continue
            file_name = name[name.find('/') + 1:] + '.jpg'
            image = gold_dataset.get_image_name(file_name)

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

                    cat = gold_dataset.get_category_name(label)
                    if cat is None:
                        cat = PCOCOCategory()
                        cat.id = cat_id
                        cat.name = label
                        gold_dataset.add_category(cat)
                        cat_id += 1

                    ann = PCOCOObjectDetectionResult()
                    ann.image_id = image.id
                    ann.id = ann_id
                    ann.category_id = cat.id
                    ann.add_box(Box(xtl, ytl, xbr, ybr))
                    ann.score = score
                    annotations.append(ann)

                    ann_id += 1

    pred_dataset = gold_dataset.get_new_dataset(annotations)
    with open(dest_pred, 'w') as fp:
        json.dump(pred_dataset.annotations, fp, cls=PCOCOJSONEncoder, indent=2)


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
