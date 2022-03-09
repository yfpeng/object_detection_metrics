[![Build status](https://github.com/yfpeng/object_detection_metrics/actions/workflows/pytest.yml/badge.svg)](https://github.com/yfpeng/object_detection_metrics/)
[![Latest version on PyPI](https://img.shields.io/pypi/v/object_detection_metrics.svg)](https://pypi.python.org/pypi/object_detection_metrics)
[![License](https://img.shields.io/pypi/l/object_detection_metrics.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/object_detection_metrics.svg)](https://pypi.python.org/pypi/object_detection_metrics)
[![Pythong version](https://img.shields.io/pypi/pyversions/object_detection_metrics)](https://pypi.python.org/pypi/object_detection_metrics)
[![codecov](https://codecov.io/gh/yfpeng/object_detection_metrics/branch/master/graph/badge.svg?token=m4mJ9fD88s)](https://codecov.io/gh/yfpeng/object_detection_metrics)
[![Hits](https://hits.dwyl.com/yfpeng/object_detection_metrics.svg)](https://hits.dwyl.com/yfpeng/object_detection_metrics)


This project was forked from [rafaelpadilla/Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics).

## Getting started

Installing `object_detection_metrics`

```shell
$ pip install object_detection_metrics
```

Reading COCO file

```python
from podm import coco_decoder
with open('tests/sample/groundtruths_coco.json') as fp:
    gold_dataset = coco_decoder.load_true_bounding_box_dataset(fp)
```

PASCAL VOC Metrics

```python
from podm import coco_decoder
from podm.metrics import get_pascal_voc_metrics, MetricPerClass, get_bounding_boxes

with open('tests/sample/groundtruths_coco.json') as fp:
    gold_dataset = coco_decoder.load_true_bounding_box_dataset(fp)
with open('tests/sample/detections_coco.json') as fp:
    pred_dataset = coco_decoder.load_pred_bounding_box_dataset(fp, gold_dataset)

gt_BoundingBoxes = get_bounding_boxes(gold_dataset)
pd_BoundingBoxes = get_bounding_boxes(pred_dataset)
results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)
```

ap, precision, recall, tp, fp, etc

```python
for cls, metric in results.items():
    label = metric.category
    print('ap', metric.ap)
    print('precision', metric.precision)
    print('interpolated_recall', metric.interpolated_recall)
    print('interpolated_precision', metric.interpolated_precision)
    print('tp', metric.tp)
    print('fp', metric.fp)
    print('num_groundtruth', metric.num_groundtruth)
    print('num_detection', metric.num_detection)
```

mAP

```python
from podm.metrics import MetricPerClass
mAP = MetricPerClass.mAP(results)
```

IoU

```python
from podm.box import Box, intersection_over_union

box1 = Box.of_box(0., 0., 10., 10.)
box2 = Box.of_box(1., 1., 11., 11.)
intersection_over_union(box1, box2)
```

Official COCO Eval

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_gld = COCO('tests/sample/groundtruths_coco.json')
coco_rst = coco_gld.loadRes('tests/sample/detections_coco.json')
cocoEval = COCOeval(coco_gld, coco_rst, iouType='bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
```

## Implemented metrics

[Tutorial](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173)

- Intersection Over Union (IOU)
- TP and FP
  - True Positive (TP): IOU ≥ *IOU threshold* (default: 0.5)
  - False Positive (FP): IOU \< *IOU threshold* (default: 0.5)
- Precision and Recall
- Average Precision
  - 11-point AP
  - all-point AP
- Official COCO Eval

## License

Copyright BioNLP Lab at Weill Cornell Medicine, 2022.

Distributed under the terms of the [MIT](https://github.com/yfpeng/object_detection_metrics/blob/master/LICENSE)
license, this is free and open source software.
