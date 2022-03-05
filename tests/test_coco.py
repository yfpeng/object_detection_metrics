import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':
    dir = Path(r'ImageCLEF2016\total')
    annFile = dir / 'instances_default.json'
    coco = COCO(annFile)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))

    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['cxr'])
    imgIds = coco.getImgIds(catIds=catIds)
    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    print(img)

    # I = io.imread(img['coco_url'])
    # plt.axis('off')
    # plt.imshow(I)
    # plt.show()

    # load and display instance annotations
    # plt.imshow(I)
    # plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    print(anns)
    # coco.showAnns(anns)

    res = []
    print(coco.getCatIds(catNms='cxr'))
    annIds = coco.getAnnIds()
    for ann in coco.loadAnns(annIds):
        ann['score'] = np.random.uniform(.5, 1)
        res.append(ann)

    rstFile = tempfile.mktemp()
    with open(rstFile, 'w') as fp:
        json.dump(res, fp, indent=2)

    cocoDt = coco.loadRes(rstFile)
    cocoEval = COCOeval(coco, cocoDt, iouType='bbox')

    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print(cocoEval.stats[1])
    # print(cocoEval.eval)
    # print(cocoEval.ious)