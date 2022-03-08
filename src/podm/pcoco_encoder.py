import json
from typing import Union, TextIO, Dict

from .pcoco import PCOCOImage, PCOCOLicense, PCOCOInfo,\
    PCOCOObjectDetectionDataset, PCOCOCategory, PCOCOObjectDetection, PCOCOObjectDetectionResult

PCOCO_OBJ = Union[PCOCOImage, PCOCOLicense, PCOCOInfo,
                  PCOCOObjectDetectionDataset, PCOCOCategory, PCOCOObjectDetection, PCOCOObjectDetectionResult]


def dumps(obj: PCOCO_OBJ, **kwargs) -> str:
    """
    Serialize a BioC ``obj`` to a JSON formatted ``str``. kwargs are passed to json.
    """
    return json.dumps(obj, cls=PCOCOJSONEncoder, indent=2, **kwargs)


def dump(obj: PCOCO_OBJ, fp: TextIO, **kwargs):
    """
    Serialize ``obj`` as a JSON formatted stream to ``fp``
    (a ``.write()``-supporting file-like object). kwargs are passed to json.
    """
    return json.dump(obj, fp, cls=PCOCOJSONEncoder, indent=2, **kwargs)


class PCOCOJSONEncoder(json.JSONEncoder):
    """
    Extensible BioC JSON encoder for BioC data structures.
    """

    def default(self, o):
        if isinstance(o, PCOCOImage):
            return {
                "width": o.width,
                "height": o.height,
                "flickr_url": o.flickr_url,
                "coco_url": o.coco_url,
                "file_name": o.file_name,
                "date_captured": o.date_captured,
                "license": o.license,
                "id": o.id,
            }
        if isinstance(o, PCOCOLicense):
            return {
                "id": o.id,
                "name": o.name,
                "url": o.url,
            }
        if isinstance(o, PCOCOInfo):
            return {
                "year": o.year,
                "version": o.version,
                "description": o.description,
                "contributor": o.contributor,
                "url": o.url,
                "date_created": o.date_created,
            }
        if isinstance(o, PCOCOCategory):
            return {
                "id": o.id,
                "name": o.name,
                "supercategory": o.supercategory,
            }
        if isinstance(o, PCOCOObjectDetection):
            bb = o.bbox
            return {
                "id": o.id,
                "image_id": o.image_id,
                "category_id": o.category_id,
                "segmentation": o.segmentation,
                "bbox": [bb.xtl, bb.ytl, bb.width, bb.height],
                "area": bb.area,
                "iscrowd": o.iscrowd,
            }
        if isinstance(o, PCOCOObjectDetectionResult):
            bb = o.bbox
            return {
                "id": o.id,
                "image_id": o.image_id,
                "category_id": o.category_id,
                "segmentation": o.segmentation,
                "bbox": [bb.xtl, bb.ytl, bb.width, bb.height],
                "area": bb.area,
                "iscrowd": o.iscrowd,
                "score": o.score
            }
        if isinstance(o, PCOCOObjectDetectionDataset):
            return {
                "info": self.default(o.info),
                'images': [self.default(img) for img in o.images],
                "licenses": [self.default(l) for l in o.licenses],
                'annotations': [self.default(ann) for ann in o.annotations],
                'categories': [self.default(cat) for cat in o.categories],
            }
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)


def toJSON(o) -> Dict:
    """
    Convert a pcoco obj to a Python `dict`
    """
    return PCOCOJSONEncoder().default(o)
