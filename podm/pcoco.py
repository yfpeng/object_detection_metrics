import json
from typing import List

from podm.podm import Box


class PCOCODataset:
    def __init__(self):
        self.annotations = []  # type: List[PCOCOAnnotation]
        self.images = []  # type: List[PCOCOImage]
        self.categories = []  # type: List[PCOCOCategory]
        self.licenses = []  # type: List[PCOCOLicense]
        self.contributor = ''
        self.description = ''
        self.url = ''
        self.date_created = ''
        self.version = ''
        self.year = 0

    def to_dict(self):
        return {
            "licenses": [i.to_dict() for i in self.licenses],
            "info": {
                "contributor": self.contributor,
                "description": self.contributor,
                "url": self.url,
                "date_created": self.date_created,
                "version": self.version,
                "year": self.year
            },
            'annotations': [i.to_dict() for i in self.annotations],
            'images': [i.to_dict() for i in self.images],
            'categories': [i.to_dict() for i in self.categories],
        }

    @property
    def cat_name_to_id(self):
        return {v.name: v.id for v in self.categories}

    @property
    def img_name_to_id(self):
        return {v.file_name: v.id for v in self.images}


class PCOCOImage:
    def __init__(self):
        self.width = 0  # type:int
        self.height = 0  # type:int
        self.flickr_url = ''  # type:str
        self.coco_url = ''  # type:str
        self.file_name = ''  # type:str
        self.license = 0  # type:int
        self.id = 0  # type:int
        self.date_captured = 0

    def to_dict(self):
        return {
            "width": self.width,
            "height": self.height,
            "flickr_url": self.flickr_url,
            "coco_url": self.coco_url,
            "file_name": self.file_name,
            "date_captured": self.date_captured,
            "license": self.license,
            "id": self.id,
        }


class PCOCOLicense:
    def __init__(self):
        self.id = 0  # type:int
        self.name = ''  # type:str
        self.url = ''  # type:str

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
        }


class PCOCOCategory:
    def __init__(self):
        self.id = 0  # type:int
        self.name = ''  # type:str
        self.supercategory = ''  # type:str

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "supercategory": self.supercategory,
        }


class PCOCOAnnotation:
    def __init__(self):
        self.id = 0  # type:int
        self.image_id = 0  # type:int
        self.category_id = 0  # type:int
        self.iscrowd = 0  # type:int
        self.score = 0.  # type:float
        self.xtl = 0
        self.ytl = 0
        self.xbr = 0
        self.ybr = 0

    @property
    def area(self) -> float:
        return (self.xbr - self.xtl) * (self.ybr - self.ytl)

    def to_dict(self):
        return {
            "id": self.id,
            "image_id": self.image_id,
            "area": self.area,
            "category_id": self.category_id,
            "bbox": [self.xtl, self.ytl, self.xbr - self.xtl, self.ybr - self.ytl],
            "segmentation": [[self.xtl, self.ytl, self.xbr, self.ytl, self.xbr, self.ybr, self.xtl, self.ybr]],
            "iscrowd": self.iscrowd,
            "score": self.score
        }


# def dump(dataset: PCOCODataset, fp, **kwargs):
#     json.dump(dataset.to_dict(), fp, **kwargs)


def load(fp, **kwargs) -> PCOCODataset:
    coco_obj = json.load(fp, **kwargs)

    dataset = PCOCODataset()
    dataset.contributor = coco_obj['info']['contributor']
    dataset.description = coco_obj['info']['description']
    dataset.url = coco_obj['info']['url']
    dataset.date_created = coco_obj['info']['date_created']
    dataset.version = coco_obj['info']['version']
    dataset.year = coco_obj['info']['year']

    for ann_obj in coco_obj['annotations']:
        ann = PCOCOAnnotation()
        ann.id = ann_obj['id']
        ann.category_id = ann_obj['category_id']
        ann.image_id = ann_obj['image_id']
        ann.iscrowd = ann_obj['iscrowd']
        ann.xtl = ann_obj['bbox'][0]
        ann.ytl = ann_obj['bbox'][1]
        ann.xbr = ann_obj['bbox'][0] + ann_obj['bbox'][2]
        ann.ybr = ann_obj['bbox'][1] + ann_obj['bbox'][3]
        if 'score' in ann_obj:
            ann.score = ann_obj['score']
        dataset.annotations.append(ann)

    for cat_obj in coco_obj['categories']:
        cat = PCOCOCategory()
        cat.id = cat_obj['id']
        cat.name = cat_obj['name']
        cat.supercategory = cat_obj['supercategory']
        dataset.categories.append(cat)

    for img_obj in coco_obj['images']:
        img = PCOCOImage()
        img.id = img_obj['id']
        img.height = img_obj['height']
        img.width = img_obj['width']
        img.file_name = img_obj['file_name']
        img.flickr_url = img_obj['flickr_url']
        img.coco_url = img_obj['coco_url']
        img.date_captured = img_obj['date_captured']
        img.license = img_obj['license']
        dataset.images.append(img)

    for lic_obj in coco_obj['licenses']:
        lic = PCOCOLicense()
        lic.id = lic_obj['id']
        lic.name = lic_obj['name']
        lic.url = lic_obj['url']
        dataset.licenses.append(lic)

    return dataset
