from podm.coco import PCOCOObjectDetectionDataset, PCOCOObjectDetection


def coco2labelme(cocodataset: PCOCOObjectDetectionDataset):
    objs = []
    for img in cocodataset.images:
        obj = {
            "version": "5.0.1",
            "flags": {},
            "imagePath": img.file_name,
            "imageData": None,
            "imageHeight": img.height,
            "imageWidth": img.width,
            "shapes": []
        }
        for annid in cocodataset.get_annotation_ids(image_ids=[img.id]):
            ann = cocodataset.get_annotation(annid)
            if isinstance(ann, PCOCOObjectDetection):
                if ann.is_rectangle:
                    bbox = ann.bbox
                    points = [[bbox.xtl, bbox.ytl], [bbox.xbr, bbox.ybr]]
                    shape_type = "rectangle"
                else:
                    points = ann.segmentation[0].exterior.coords
                    shape_type = "polygon"
                shape = {
                    "label": ann.attributes['ID'],
                    "points": points,
                    "group_id": None,
                    "shape_type": shape_type,
                    "flags": {}
                }
            else:
                raise TypeError
            obj['shapes'].append(shape)

        objs.append(obj)

    return objs