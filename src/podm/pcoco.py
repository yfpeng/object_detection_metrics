import copy
import warnings
from abc import ABC
from typing import List
from datetime import date, datetime

from podm import box


class PCOCOInfo:
    def __init__(self):
        self.year = date.today().year  # type:int
        self.version = ''  # type: str
        self.description = ''  # type: str
        self.contributor = ''  # type: str
        self.url = ''  # type: str
        self.date_created = datetime.now().strftime('%m/%d/%Y')  # type:str


class PCOCOAnnotation(ABC):
    pass


class PCOCOImage:
    def __init__(self):
        self.id = None  # type:int or None
        self.width = 0  # type:int
        self.height = 0  # type:int
        self.file_name = ''  # type:str
        self.license = None  # type:int or None
        self.flickr_url = ''  # type:str
        self.coco_url = ''  # type:str
        self.date_captured = datetime.now().strftime('%m/%d/%Y')  # type:str


class PCOCOLicense:
    def __init__(self):
        self.id = None  # type:int or None
        self.name = ''  # type:str
        self.url = ''  # type:str


class PCOCOCategory:
    def __init__(self):
        self.id = None  # type:int or None
        self.name = ''  # type:str
        self.supercategory = ''  # type:str


class PCOCODataset(ABC):
    def __init__(self):
        self.info = PCOCOInfo()  # type: PCOCOInfo or None
        self.images = []  # type: List[PCOCOImage]
        self.licenses = []  # type: List[PCOCOLicense]

    def add_image(self, image: PCOCOImage):
        for img in self.images:
            if img.id == image.id or img.file_name == image.file_name:
                warnings.warn('%s: Image exists' % img.id)
        self.images.append(image)

    def get_image_id(self, image_id: int, default=None) -> PCOCOImage:
        for img in self.images:
            if img.id == image_id:
                return img
        return default

    def get_image_name(self, image_file_name: str, default=None) -> PCOCOImage:
        for img in self.images:
            if img.file_name == image_file_name:
                return img
        return default


##############################################################################
# object detection
##############################################################################

class PCOCOObjectDetection(PCOCOAnnotation):
    def __init__(self):
        self.id = None  # type:int or None
        self.image_id = None  # type:int or None
        self.category_id = None  # type:int or None
        self.segmentation = []  # type: List[List[int]]
        self.iscrowd = False  # type:bool

    def add_box(self, box: box.Box):
        self.segmentation = [[box.xtl, box.ytl, box.xtl, box.ybr, box.xbr, box.ybr, box.xbr, box.ytl]]

    @property
    def box(self) -> 'box.Box' or None:
        if len(self.segmentation) == 0:
            return None
        else:
            bb = self.box_polygon(self.segmentation[0])
            for polygon in self.segmentation[1:]:
                bb = box.union(bb, self.box_polygon(polygon))
            return bb

    @property
    def bbox(self) -> 'box.BoundingBox' or None:
        b = self.box
        if b is None:
            return b
        return box.BoundingBox(self.image_id, self.category_id, b.xtl, b.ytl, b.xbr, b.ybr)

    @classmethod
    def box_polygon(cls, polygon: List[int]) -> 'box.Box':
        xtl = min(polygon[i] for i in range(0, len(polygon), 2))
        ytl = min(polygon[i] for i in range(1, len(polygon), 2))
        xbr = max(polygon[i] for i in range(0, len(polygon), 2))
        ybr = max(polygon[i] for i in range(1, len(polygon), 2))
        return box.Box(xtl, ytl, xbr, ybr)


class PCOCOObjectDetectionResult(PCOCOObjectDetection):
    def __init__(self):
        super(PCOCOObjectDetectionResult, self).__init__()
        self.score = 0.  # type:float


class PCOCOObjectDetectionDataset(PCOCODataset):
    def __init__(self):
        super(PCOCOObjectDetectionDataset, self).__init__()
        self.annotations = []  # type: List[PCOCOObjectDetection]
        self.categories = []  # type: List[PCOCOCategory]

    def add_annotation(self, annotation: 'PCOCOObjectDetection'):
        for ann in self.annotations:
            if ann.id == annotation.id:
                warnings.warn('%s: Annotation exists' % ann.id)
        self.annotations.append(annotation)

    def add_category(self, category: PCOCOCategory):
        for cat in self.categories:
            if cat.id == category.id or cat.name == category.name:
                warnings.warn('%s: Category exists' % cat.id)
        self.categories.append(category)

    def get_category_name(self, category_name: str, default=None) -> PCOCOCategory:
        for cat in self.categories:
            if cat.name == category_name:
                return cat
        return default

    def get_max_category_id(self):
        return max(cat.id for cat in self.categories)

    def get_category_id(self, category_id: int, default=None) -> PCOCOCategory:
        for cat in self.categories:
            if cat.id == category_id:
                return cat
        return default

    def get_new_dataset(self, annotations: List[PCOCOObjectDetection]):
        new_dataset = PCOCOObjectDetectionDataset()
        new_dataset.info = copy.deepcopy(self.info)
        new_dataset.licenses = copy.deepcopy(self.licenses)
        new_dataset.images = copy.deepcopy(self.images)
        new_dataset.categories = copy.deepcopy(self.categories)
        for ann in annotations:
            new_dataset.add_annotation(ann)
        return new_dataset

    def bboxes(self):
        return [ann.bbox for ann in self.annotations]
