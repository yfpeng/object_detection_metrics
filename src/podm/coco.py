import copy
from abc import ABC
from typing import List, Tuple, Set, Collection
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

    def add_license(self, license: PCOCOLicense):
        for lic in self.licenses:
            if lic.id == license.id or lic.name == license.name:
                raise KeyError('%s: License exists' % lic.id)
        self.licenses.append(license)

    def add_image(self, image: PCOCOImage):
        for img in self.images:
            if img.id == image.id or img.file_name == image.file_name:
                raise KeyError('%s: Image exists' % img.id)
        self.images.append(image)

    def get_image(self, id: int = None, file_name: str = None, default=None) -> PCOCOImage:
        if id is None and file_name is None:
            raise KeyError('%s %s: Cannot set both to None' % (id, file_name))
        if id is not None and file_name is not None:
            raise KeyError('%s %s: Cannot set both' % (id, file_name))

        imgs = self.images
        if id is not None:
            imgs = [img for img in imgs if img.id == id]
            if len(imgs) == 0:
                return default
            elif len(imgs) == 1:
                return next(iter(imgs))
            else:
                raise KeyError('%s: more than one image with the same id' % id)

        if file_name is not None:
            imgs = [img for img in imgs if img.file_name == file_name]
            if len(imgs) == 0:
                return default
            elif len(imgs) == 1:
                return next(iter(imgs))
            else:
                raise KeyError('%s: more than one image with the same name' % file_name)

        raise Exception('Should not be here')

    def get_images(self, ids: Collection[int] = None) -> List[PCOCOImage]:
        """
        Load anns with the specified ids.
        :param ids: integer ids specifying img
        :return: imgs: loaded img objects
        """
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
        self.add_segmentation(box.segment)

    def add_segmentation(self, segmentation: List[float]):
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

    def get_max_category_id(self):
        return max(cat.id for cat in self.categories)

    def get_category(self, id: int = None, name: str = None, default=None) -> PCOCOCategory:
        if id is None and name is None:
            raise KeyError('%s %s: Cannot set both to None' % (id, name))
        if id is not None and name is not None:
            raise KeyError('%s %s: Cannot set both' % (id, name))

        cats = self.categories
        if id is not None:
            cats = [cat for cat in cats if cat.id == id]
            if len(cats) == 0:
                return default
            elif len(cats) == 1:
                return next(iter(cats))
            else:
                raise KeyError('%s: more than one category with the same id' % id)

        if name is not None:
            cats = [cat for cat in cats if cat.name == name]
            if len(cats) == 0:
                return default
            elif len(cats) == 1:
                return next(iter(cats))
            else:
                raise KeyError('%s: more than one category with the same name' % name)

        raise Exception('Should not be here')

    def get_annotation(self, id: int, default=None) -> PCOCOAnnotation:
        anns = [ann for ann in self.annotations if ann.id == id]
        if len(anns) == 0:
            return default
        elif len(anns) == 1:
            return next(iter(anns))
        else:
            raise KeyError('%s: more than one annotation' % id)

    def get_new_dataset(self, annotations: Collection[PCOCOBoundingBox]):
        new_dataset = PCOCOBoundingBoxDataset()
        new_dataset.info = copy.deepcopy(self.info)
        new_dataset.licenses = copy.deepcopy(self.licenses)
        new_dataset.images = copy.deepcopy(self.images)
        new_dataset.categories = copy.deepcopy(self.categories)
        for ann in annotations:
            new_dataset.add_annotation(ann)
        return new_dataset

    def get_category_ids(self, category_names: Collection[str] = None,
                         supercategory_names: Collection[str] = None) -> Collection[int]:
        """
        filtering parameters. default skips that filter.
        :param category_names: get cats for given cat names
        :param supercategory_names: get cats for given supercategory names
        :return: integer array of cat ids
        """
        itr = iter(self.categories)
        if category_names is not None:
            itr = filter(lambda x: x.name in category_names, itr)
        if supercategory_names is not None:
            itr = filter(lambda x: x.supercategory in supercategory_names, itr)
        return [cat.id for cat in itr]

    def get_annotation_ids(self, image_ids: Collection[int] = None,
                           category_ids: Collection[int] = None,
                           area_range: Tuple[float, float] = None) -> Collection[int]:
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param image_ids: get anns for given imgs
        :param category_ids: get anns for given cats
        :param area_range: get anns for given area range (e.g. [0 inf])
        :return: integer array of ann ids
        """
        # image_ids = convert_array_argument(image_ids)
        # category_ids = convert_array_argument(category_ids)
        # area_range = convert_array_argument(area_range)

        itr = iter(self.annotations)
        if image_ids is not None:
            itr = filter(lambda x: x.image_id in image_ids, itr)
        if category_ids is not None:
            itr = filter(lambda x: x.category_id in category_ids, itr)
        if area_range is not None:
            itr = filter(lambda x: area_range[0] <= x.area <= area_range[1], itr)
        return [ann.id for ann in itr]

    def get_image_ids(self, category_ids: Collection[int] = None) -> Collection[int]:
        """
        Get img ids that satisfy given filter conditions.
        :param category_ids: get imgs with all given cats
        :return: ids: integer array of img ids
        """
        ids = set(img.id for img in self.images)
        for i, cat_id in enumerate(category_ids):
            if i == 0 and len(ids) == 0:
                ids = set(ann.image_id for ann in self.annotations if ann.category_id == cat_id)
            else:
                ids &= set(ann.image_id for ann in self.annotations if ann.category_id == cat_id)
        return list(ids)

    def get_annotations(self, ids: Collection[int] = None) -> Collection[PCOCOAnnotation]:
        """
        Load anns with the specified ids.
        :param ids: integer ids specifying anns
        :return: anns: loaded ann objects
        """
        return [ann for ann in self.annotations if ann.id in ids]

    def get_categories(self, ids: Collection[int] = None) -> Collection[PCOCOCategory]:
        """
        Load cats with the specified ids.
        :param ids: integer ids specifying cats
        :return: cats: loaded cat objects
        """
        return [cat for cat in self.categories if cat.id in ids]
