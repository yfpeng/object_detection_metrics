import copy
import itertools
from abc import ABC
from typing import List, Union, NewType, Tuple
from datetime import date, datetime

from podm import box
from podm.utils import _isArrayLike, _default_argument

ListStr = NewType('ListStr', Union[List[str], str])
ListInt = NewType('ListInt', Union[List[int], int])
Range = NewType('Range', Tuple[float, float])


class PCOCOInfo:
    def __init__(self):
        self.year = date.today().year  # type:int
        self.version = ''  # type: str
        self.description = ''  # type: str
        self.contributor = ''  # type: str
        self.url = ''  # type: str
        self.date_created = datetime.now().strftime('%m/%d/%Y')  # type:str


class PCOCOAnnotation(ABC):
    def __init__(self):
        self.id = None  # type:int or None
        self.image_id = None  # type:int or None
        self.score = None  # type:float or None
        self.contributor = ''  # type: str


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
                raise KeyError('%s: Image exists' % img.id)
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

    def loadImgs(self, ids: ListInt=None) -> List[PCOCOImage]:
        """
        Load anns with the specified ids.
        :param ids: integer ids specifying img
        :return: imgs: loaded img objects
        """
        ids = _default_argument(ids)
        return [img for img in self.images if img.id in ids]


##############################################################################
# object detection
##############################################################################


class PCOCOBoundingBox(PCOCOAnnotation, box.Box):
    def __init__(self):
        super(PCOCOBoundingBox, self).__init__()
        self.category_id = None  # type:int or None


class PCOCOSegments(PCOCOAnnotation):
    def __init__(self):
        super(PCOCOSegments, self).__init__()
        self.category_id = None  # type:int or None
        self.segmentation = []  # type: List[List[float]]
        self.iscrowd = False  # type:bool

    def add_box(self, box: box.Box):
        self.add_segmenation([box.xtl, box.ytl, box.xtl, box.ybr, box.xbr, box.ybr, box.xbr, box.ytl])

    def add_segmenation(self, segmentation: List[float]):
        self.segmentation.append(segmentation)

    @property
    def bbox(self) -> 'box.Box' or None:
        if len(self.segmentation) == 0:
            return None
        else:
            b = self.box_polygon(self.segmentation[0])
            for polygon in self.segmentation[1:]:
                b = box.union(b, self.box_polygon(polygon))
            return b

    @classmethod
    def box_polygon(cls, polygon: List[float]) -> 'box.Box':
        xtl = min(polygon[i] for i in range(0, len(polygon), 2))
        ytl = min(polygon[i] for i in range(1, len(polygon), 2))
        xbr = max(polygon[i] for i in range(0, len(polygon), 2))
        ybr = max(polygon[i] for i in range(1, len(polygon), 2))
        return box.Box.of_box(xtl, ytl, xbr, ybr)


class PCOCOImageCaptioning(PCOCOAnnotation):
    def __init__(self):
        super(PCOCOImageCaptioning, self).__init__()
        self.caption = None  # type:str or None


class PCOCOBoundingBoxDataset(PCOCODataset):
    def __init__(self):
        super(PCOCOBoundingBoxDataset, self).__init__()
        self.annotations = []  # type: List[PCOCOBoundingBox]
        self.categories = []  # type: List[PCOCOCategory]

    def add_annotation(self, annotation: 'PCOCOBoundingBox'):
        for ann in self.annotations:
            if ann.id == annotation.id:
                raise KeyError('%s: Annotation exists' % ann.id)
        self.annotations.append(annotation)

    def add_category(self, category: PCOCOCategory):
        for cat in self.categories:
            if cat.id == category.id or cat.name == category.name:
                raise KeyError('%s: Category exists' % cat.id)
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

    def get_new_dataset(self, annotations: List[PCOCOBoundingBox]):
        new_dataset = PCOCOBoundingBoxDataset()
        new_dataset.info = copy.deepcopy(self.info)
        new_dataset.licenses = copy.deepcopy(self.licenses)
        new_dataset.images = copy.deepcopy(self.images)
        new_dataset.categories = copy.deepcopy(self.categories)
        for ann in annotations:
            new_dataset.add_annotation(ann)
        return new_dataset

    def get_category_ids(self, category_names: ListStr = None,
                         supercategory_name: ListStr = None,
                         category_ids: ListInt = None) -> List[int]:
        """
        filtering parameters. default skips that filter.
        :param category_names: get cats for given cat names
        :param supercategory_name: get cats for given supercategory names
        :param category_ids: get cats for given cat ids
        :return: integer array of cat ids
        """
        category_names = _default_argument(category_names)
        supercategory_name = _default_argument(supercategory_name)
        category_ids = _default_argument(category_ids)

        if len(category_names) == len(supercategory_name) == len(category_ids) == 0:
            cats = self.categories
        else:
            cats = self.categories
            cats = cats if len(category_names) == 0 else [cat for cat in cats if cat.name in category_names]
            cats = cats if len(supercategory_name) == 0 else [cat for cat in cats if cat.supercategory in supercategory_name]
            cats = cats if len(category_ids) == 0 else [cat for cat in cats if cat.id in category_ids]
        ids = [cat.id for cat in cats]
        return ids

    def get_annotation_ids(self, image_ids: ListInt = None,
                           category_ids: ListInt = None,
                           area_range: Range = None) -> List[int]:
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param image_ids: get anns for given imgs
        :param category_ids: get anns for given cats
        :param area_range: get anns for given area range (e.g. [0 inf])
        :param iscrowd: get anns for given crowd label (False or True)
        :return: integer array of ann ids
        """
        image_ids = _default_argument(image_ids)
        category_ids = _default_argument(category_ids)

        if len(image_ids) == len(category_ids) == len(area_range) == 0:
            anns = self.annotations
        else:
            if not len(image_ids) == 0:
                anns = [ann for ann in self.annotations if ann.image_id in image_ids]
            else:
                anns = self.annotations
            anns = anns if len(category_ids) == 0 else [ann for ann in anns if ann.category_id in category_ids]
            anns = anns if len(area_range) == 0 else [ann for ann in anns if area_range[0] < ann.area < area_range[1]]
        ids = [ann.id for ann in anns]
        return ids

    def get_image_ids(self, image_ids: ListInt = None, category_ids: ListInt = None) -> List[int]:
        """
        Get img ids that satisfy given filter conditions.
        :param image_ids: get imgs for given ids
        :param category_ids: get imgs with all given cats
        :return: ids: integer array of img ids
        """
        image_ids = _default_argument(image_ids)
        category_ids = _default_argument(category_ids)

        if len(image_ids) == len(category_ids) == 0:
            ids = [img.id for img in self.images]
        else:
            ids = set(image_ids)
            for i, catId in enumerate(category_ids):
                if i == 0 and len(ids) == 0:
                    ids = set(ann.image_id for ann in self.annotations if ann.category_id == catId)
                else:
                    ids &= set(ann.image_id for ann in self.annotations if ann.category_id == catId)
        return list(ids)

    def load_annotations(self, ids: ListInt = None) -> List[PCOCOAnnotation]:
        """
        Load anns with the specified ids.
        :param ids: integer ids specifying anns
        :return: anns: loaded ann objects
        """
        ids = _default_argument(ids)
        return [ann for ann in self.annotations if ann.id in ids]

    def load_categories(self, ids: ListInt = None) -> List[PCOCOCategory]:
        """
        Load cats with the specified ids.
        :param ids: integer ids specifying cats
        :return: cats: loaded cat objects
        """
        ids = _default_argument(ids)
        return [cat for cat in self.categories if cat.id in ids]

    # def bboxes(self, use_name: bool = True) -> List[box.Box]:
    #     if use_name:
    #         bboxes = [ann.bbox for ann in self.annotations]
    #         for bb in bboxes:
    #             bb.image = self.get_image_id(bb.image).file_name
    #             bb.category = self.get_category_id(bb.category).name
    #         return bboxes
    #     else:
    #         return [ann.bbox for ann in self.annotations]
