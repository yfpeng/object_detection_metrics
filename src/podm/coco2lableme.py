from podm.coco import PCOCOObjectDetectionDataset, PCOCOBoundingBox, PCOCOSegments


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
            if isinstance(ann, PCOCOBoundingBox):
                shape = {
                    "label": ann.attributes['ID'],
                    "points": [[ann.xtl, ann.ytl], [ann.xbr, ann.ybr]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
            elif isinstance(ann, PCOCOSegments):
                shape = {
                    "label": ann.attributes['ID'],
                    "points": [[ann.segmentation[0][i], ann.segmentation[0][i+1]]
                               for i in range(0, len(ann.segmentation[0]), 2)],
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
            else:
                raise TypeError
            obj['shapes'].append(shape)

        objs.append(obj)

    return objs