from abc import ABC
from typing import List
from datetime import date, datetime

from podm import box


class PCOCODataset(ABC):
    def __init__(self):
        self.info = PCOCOInfo()  # type: PCOCOInfo or None
        self.images = []  # type: List[PCOCOImage]
        self.licenses = []  # type: List[PCOCOLicense]
        # flag
        self.img_name_to_id = {}
        self.img_id_to_name = {}
        # self.img_name_to_id_dirty = False

    def create_index(self):
        self.img_name_to_id = {v.file_name: v.id for v in self.images}
        self.img_id_to_name = {v.id: v.file_name for v in self.images}

    def add_image(self, image: 'PCOCOImage'):
        self.images.append(image)

    def has_image_id(self, image_id: int):
        return image_id in self.img_id_to_name


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
        self.date_captured = datetime.now()  # type:datetime


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


##############################################################################
# object detection
##############################################################################

class PCOCOObjectDetectionDataset(PCOCODataset):
    def __init__(self):
        super(PCOCOObjectDetectionDataset, self).__init__()
        self.annotations = []  # type: List[PCOCOAnnotation]
        self.categories = []  # type: List[PCOCOCategory]
        # flag
        self.category_name_to_id = {}
        self.category_id_to_name = {}

    def add_annotation(self, annotation: 'PCOCOObjectDetection'):
        self.annotations.append(annotation)

    def create_index(self):
        super(PCOCOObjectDetectionDataset, self).create_index()
        self.category_name_to_id = {v.name: v.id for v in self.categories}
        self.category_id_to_name = {v.id: v.name for v in self.categories}

    def add_category(self, category: PCOCOCategory):
        self.categories.append(category)

    def get_category_id(self, category_name: str):
        return self.category_name_to_id[category_name]

    def has_category_id(self, category_id: int):
        return category_id in self.category_id_to_name


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
    def bbox(self) -> box.Box or None:
        if len(self.segmentation) == 0:
            return None
        else:
            bb = self.bbox_polygon(self.segmentation[0])
            for polygon in self.segmentation[1:]:
                bb = box.union(bb, self.bbox_polygon(polygon))
            return bb

    @classmethod
    def bbox_polygon(cls, polygon: List[int]) -> box.Box:
        xtl = min(polygon[i] for i in range(0, len(polygon), 2))
        ytl = min(polygon[i] for i in range(1, len(polygon), 2))
        xbr = max(polygon[i] for i in range(0, len(polygon), 2))
        ybr = max(polygon[i] for i in range(1, len(polygon), 2))
        return box.Box(xtl, ytl, xbr, ybr)


class PCOCOObjectDetectionResult(PCOCOObjectDetection):
    def __init__(self):
        super(PCOCOObjectDetectionResult, self).__init__()
        self.score = 0.  # type:float
