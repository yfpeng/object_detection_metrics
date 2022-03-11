from enum import Enum


class Box:
    """
                0,0 ------> x (width)
         |
         |  (Left,Top)
         |      *_________
         |      |         |
                |         |
         y      |_________|
      (height)            *
                    (Right,Bottom)

    xtl: the X top-left coordinate of the bounding box.
    ytl: the Y top-left coordinate of the bounding box.
    xbr: the X bottom-right coordinate of the bounding box.
    ybr: the Y bottom-right coordinate of the bounding box.
    """
    def __init__(self):
        self.xtl = None  # type: float or None
        self.ytl = None  # type: float or None
        self.xbr = None  # type: float or None
        self.ybr = None  # type: float or None

    @classmethod
    def of_box(cls, xtl: float, ytl: float, xbr: float, ybr: float) -> 'Box':
        """
        :param xtl: the X top-left coordinate of the bounding box.
        :param ytl: the Y top-left coordinate of the bounding box.
        :param xbr: the X bottom-right coordinate of the bounding box.
        :param ybr: the Y bottom-right coordinate of the bounding box.
        """
        box = Box()
        box.xtl = xtl
        box.ytl = ytl
        box.xbr = xbr
        box.ybr = ybr
        box.verify()
        return box

    def set_box(self, box: 'Box'):
        self.xtl = box.xtl
        self.ytl = box.ytl
        self.xbr = box.xbr
        self.ybr = box.ybr

    def verify(self):
        assert self.xtl <= self.xbr, f'xtl < xbr: xtl:{self.xtl}, xbr:{self.xbr}'
        assert self.ytl <= self.ybr, f'ytl < ybr: ytl:{self.ytl}, xbr:{self.ybr}'

    @property
    def segment(self):
        return [self.xtl, self.ytl, self.xtl, self.ybr, self.xbr, self.ybr, self.xbr, self.ytl]

    @property
    def width(self) -> float:
        return self.xbr - self.xtl

    @property
    def height(self) -> float:
        return self.ybr - self.ytl

    @property
    def area(self) -> float:
        return (self.xbr - self.xtl) * (self.ybr - self.ytl)

    def __str__(self):
        return 'Box[xtl={},ytl={},xbr={},ybr={}]'.format(self.xtl, self.ytl, self.xbr, self.ybr)

    def __eq__(self, other):
        if not isinstance(other, Box):
            return False
        return self.xtl == other.xtl and self.ytl == other.ytl and self.xbr == other.xbr and self.ybr == other.ybr


def intersection_over_union(box1: 'Box', box2: 'Box') -> float:
    """
    Intersection Over Union (IOU) is measure based on Jaccard Index that evaluates the overlap between
    two bounding boxes.
    """
    # if boxes dont intersect
    if not is_intersecting(box1, box2):
        return 0
    intersection_area = intersection(box1, box2).area
    union = union_areas(box1, box2, intersection_area=intersection_area)
    # intersection over union
    iou = intersection_area / union
    assert iou >= 0, '{} = {} / {}, box1={}, box2={}'.format(iou, intersection, union, box1, box2)
    return iou


def is_intersecting(box1: 'Box', box2: 'Box') -> bool:
    if box1.xtl > box2.xbr:
        return False  # boxA is right of boxB
    if box2.xtl > box1.xbr:
        return False  # boxA is left of boxB
    if box1.ybr < box2.ytl:
        return False  # boxA is above boxB
    if box1.ytl > box2.ybr:
        return False  # boxA is below boxB
    return True


def union_areas(box1: 'Box', box2: 'Box', intersection_area: float = None) -> float:
    if intersection_area is None:
        intersection_area = intersection(box1, box2).area
    return box1.area + box2.area - intersection_area


def union(box1: 'Box', box2: 'Box'):
    xtl = min(box1.xtl, box2.xtl)
    ytl = min(box1.ytl, box2.ytl)
    xbr = max(box1.xbr, box2.xbr)
    ybr = max(box1.ybr, box2.ybr)
    return Box.of_box(xtl, ytl, xbr, ybr)


def intersection(box1: 'Box', box2: 'Box'):
    xtl = max(box1.xtl, box2.xtl)
    ytl = max(box1.ytl, box2.ytl)
    xbr = min(box1.xbr, box2.xbr)
    ybr = min(box1.ybr, box2.ybr)
    return Box.of_box(xtl, ytl, xbr, ybr)


class BBFormat(Enum):
    """
    Class representing the format of a bounding box.
    It can be (X,Y,width,height) => XYWH
    or (X1,Y1,X2,Y2) => XYX2Y2

        Developed by: Rafael Padilla
        Last modification: May 24 2018
    """
    XYWH = 1
    X1Y1X2Y2 = 2


# class BoundingBox(Box):
#     def __init__(self):
#         """Constructor.
#         Args:
#             image: image.
#             category: category.
#             xtl: the X top-left coordinate of the bounding box.
#             ytl: the Y top-left coordinate of the bounding box.
#             xbr: the X bottom-right coordinate of the bounding box.
#             ybr: the Y bottom-right coordinate of the bounding box.
#             score: (optional) the confidence of the detected class.
#         """
#         super(BoundingBox, self).__init__()
#         self.image = None
#         self.category = None
#         self.score = None  # type: float or None
#
#     @classmethod
#     def of_bbox(cls, image, category, xtl: float, ytl: float, xbr: float, ybr: float, score: float = None) \
#             -> 'BoundingBox':
#         bbox = BoundingBox()
#         bbox.xtl = xtl
#         bbox.ytl = ytl
#         bbox.xbr = xbr
#         bbox.ybr = ybr
#         bbox.image = image
#         bbox.score = score
#         bbox.category = category
#         return bbox