import copy
import json
from typing import Dict

from podm.pcoco import PCOCOLicense, PCOCOInfo, PCOCOObjectDetectionDataset, PCOCOImage, PCOCOObjectDetection, \
    PCOCOCategory, PCOCOObjectDetectionResult


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


def parse_object_detection_annotation(obj: Dict) -> PCOCOObjectDetection:
    ann = PCOCOObjectDetection()
    ann.id = obj['id']
    ann.category_id = obj['category_id']
    ann.image_id = obj['image_id']
    ann.iscrowd = obj['iscrowd']
    ann.segmentation = obj['segmentation']
    return ann


def parse_object_detection_annotation_result(obj: Dict) -> PCOCOObjectDetectionResult:
    ann = PCOCOObjectDetectionResult()
    ann.id = obj['id']
    ann.category_id = obj['category_id']
    ann.image_id = obj['image_id']
    ann.iscrowd = obj['iscrowd']
    ann.segmentation = obj['segmentation']
    ann.score = obj['score']
    return ann


def parse_category(obj: Dict) -> PCOCOCategory:
    cat = PCOCOCategory()
    cat.id = obj['id']
    cat.name = obj['name']
    cat.supercategory = obj['supercategory']
    return cat


def parse_object_detection(coco_obj: Dict) -> PCOCOObjectDetectionDataset:
    dataset = PCOCOObjectDetectionDataset()
    dataset.info = parse_infon(coco_obj['info'])

    for lic_obj in coco_obj['licenses']:
        lic = parse_license(lic_obj)
        dataset.licenses.append(lic)

    for img_obj in coco_obj['images']:
        img = parse_image(img_obj)
        dataset.images.append(img)

    for ann_obj in coco_obj['annotations']:
        ann = parse_object_detection_annotation(ann_obj)
        dataset.annotations.append(ann)

    for cat_obj in coco_obj['categories']:
        cat = parse_category(cat_obj)
        dataset.categories.append(cat)

    return dataset


def loads_object_detection(s, **kwargs) -> PCOCOObjectDetectionDataset:
    coco_obj = json.loads(s, **kwargs)
    return parse_object_detection(coco_obj)


def load_object_detection(fp, **kwargs) -> PCOCOObjectDetectionDataset:
    coco_obj = json.load(fp, **kwargs)
    return parse_object_detection(coco_obj)


def load_object_detection_result(fp, dataset: PCOCOObjectDetectionDataset, **kwargs) -> PCOCOObjectDetectionDataset:
    new_dataset = PCOCOObjectDetectionDataset()
    new_dataset.info = copy.deepcopy(dataset.info)
    new_dataset.licenses = copy.deepcopy(dataset.licenses)
    new_dataset.images = copy.deepcopy(dataset.images)
    new_dataset.categories = copy.deepcopy(dataset.categories)
    new_dataset.create_index()
    # check annotation
    coco_obj = json.load(fp, **kwargs)
    annotations = []
    for obj in coco_obj:
        ann = parse_object_detection_annotation_result(obj)
        if not new_dataset.has_image_id(ann.image_id):
            print('%s: Cannot find image' % ann.image_id)
        if not new_dataset.has_category_id(ann.category_id):
            print('%s: Cannot find category' % ann.category_id)
        annotations.append(ann)
    new_dataset.annotations = annotations
    return new_dataset
