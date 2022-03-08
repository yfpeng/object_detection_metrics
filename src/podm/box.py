from enum import Enum


class Box:
    def __init__(self, xtl: float, ytl: float, xbr: float, ybr: float):
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

        Args:
            xtl: the X top-left coordinate of the bounding box.
            ytl: the Y top-left coordinate of the bounding box.
            xbr: the X bottom-right coordinate of the bounding box.
            ybr: the Y bottom-right coordinate of the bounding box.
        """
        assert xtl <= xbr, f'xtl < xbr: xtl:{xtl}, xbr:{xbr}'
        assert ytl <= ybr, f'ytl < ybr: ytl:{ytl}, xbr:{ybr}'

        self.xtl = xtl
        self.ytl = ytl
        self.xbr = xbr
        self.ybr = ybr

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
    return Box(xtl, ytl, xbr, ybr)


def intersection(box1: 'Box', box2: 'Box'):
    xtl = max(box1.xtl, box2.xtl)
    ytl = max(box1.ytl, box2.ytl)
    xbr = min(box1.xbr, box2.xbr)
    ybr = min(box1.ybr, box2.ybr)
    return Box(xtl, ytl, xbr, ybr)


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


class BoundingBox(Box):
    def __init__(self, image_id, category_id, xtl: float, ytl: float, xbr: float, ybr: float,
                 score: float = None):
        """Constructor.
        Args:
            image_id: image id.
            category_id: category id.
            xtl: the X top-left coordinate of the bounding box.
            ytl: the Y top-left coordinate of the bounding box.
            xbr: the X bottom-right coordinate of the bounding box.
            ybr: the Y bottom-right coordinate of the bounding box.
            score: (optional) the confidence of the detected class.
        """
        super().__init__(xtl, ytl, xbr, ybr)
        self.image_id = image_id
        self.score = score
        self.category_id = category_id