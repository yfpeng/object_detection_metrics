import copy
import json
from typing import Dict

from podm.coco import PCOCOLicense, PCOCOInfo, PCOCOImage, PCOCOCategory, PCOCOObjectDetection, \
    PCOCOObjectDetectionDataset


def parse_infon(obj: Dict) -> PCOCOInfo:
    info = PCOCOInfo()
    info.contributor = obj['contributor']
    info.description = obj['description']
    info.url = obj['url']
    info.date_created = obj['date_created']
    info.version = obj['version']
    info.year = obj['year']
    return info


def parse_license(obj: Dict) -> PCOCOLicense:
    lic = PCOCOLicense()
    lic.id = obj['id']
    lic.name = obj['name']
    lic.url = obj['url']
    return lic


def parse_image(obj: Dict) -> PCOCOImage:
    img = PCOCOImage()
    img.id = obj['id']
    img.height = obj['height']
    img.width = obj['width']
    img.file_name = obj['file_name']
    img.flickr_url = obj['flickr_url']
    img.coco_url = obj['coco_url']
    img.date_captured = obj['date_captured']
    img.license = obj['license']
    return img


def parse_bounding_box(obj: Dict) -> PCOCOBoundingBox:
    ann = PCOCOBoundingBox()
    ann.id = obj['id']
    ann.category_id = obj['category_id']
    ann.image_id = obj['image_id']
    ann.xtl = obj['bbox'][0]
    ann.ytl = obj['bbox'][1]
    ann.xbr = ann.xtl + obj['bbox'][2]
    ann.ybr = ann.ytl + obj['bbox'][3]
    if 'contributor' in obj:
        ann.contributor = obj['contributor']
    if 'score' in obj:
        ann.score = obj['score']
    if 'attributes' in obj:
        ann.attributes = obj['attributes']
    return ann


def parse_segments(obj: Dict) -> PCOCOSegments:
    ann = PCOCOSegments()
    ann.id = obj['id']
    ann.category_id = obj['category_id']
    ann.image_id = obj['image_id']
    ann.iscrowd = obj['iscrowd']
    ann.segmentation = obj['segmentation']
    if 'score' in obj:
        ann.score = obj['score']
    if 'contributor' in obj:
        ann.contributor = obj['contributor']
    if 'attributes' in obj:
        ann.attributes = obj['attributes']
    return ann


def parse_category(obj: Dict) -> PCOCOCategory:
    cat = PCOCOCategory()
    cat.id = obj['id']
    cat.name = obj['name']
    cat.supercategory = obj['supercategory']
    return cat


def parse_object_detection_dataset(coco_obj: Dict) -> PCOCOObjectDetectionDataset:
    dataset = PCOCOObjectDetectionDataset()
    dataset.info = parse_infon(coco_obj['info'])

    for lic_obj in coco_obj['licenses']:
        lic = parse_license(lic_obj)
        dataset.licenses.append(lic)

    for img_obj in coco_obj['images']:
        img = parse_image(img_obj)
        dataset.images.append(img)

    for ann_obj in coco_obj['annotations']:
        if 'segmentation' in ann_obj and len(ann_obj['segmentation']) > 0:
            ann = parse_segments(ann_obj)
        else:
            ann = parse_bounding_box(ann_obj)
        dataset.add_annotation(ann)

    for cat_obj in coco_obj['categories']:
        cat = parse_category(cat_obj)
        dataset.categories.append(cat)

    return dataset


def load_true_object_detection_dataset(fp, **kwargs) -> PCOCOObjectDetectionDataset:
    coco_obj = json.load(fp, **kwargs)
    return parse_object_detection_dataset(coco_obj)


def load_pred_object_detection_dataset(fp, dataset: PCOCOObjectDetectionDataset, **kwargs) \
        -> PCOCOObjectDetectionDataset:
    new_dataset = PCOCOObjectDetectionDataset()
    new_dataset.info = copy.deepcopy(dataset.info)
    new_dataset.licenses = copy.deepcopy(dataset.licenses)
    new_dataset.images = copy.deepcopy(dataset.images)
    new_dataset.categories = copy.deepcopy(dataset.categories)
    # check annotation
    coco_obj = json.load(fp, **kwargs)
    annotations = []
    for obj in coco_obj:
        ann = parse_bounding_box(obj)
        if new_dataset.get_image(id=ann.image_id) is None:
            print('%s: Cannot find image' % ann.image_id)
        if new_dataset.get_category(id=ann.category_id) is None:
            print('%s: Cannot find category' % ann.category_id)
        annotations.append(ann)
    new_dataset.annotations = annotations
    return new_dataset
