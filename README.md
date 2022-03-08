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

Reading Josn file

```python
import podm
bounding_boxes = podm.load_data('tests/sample_2/groundtruths.json')
```

Reading COCO file

```python
import podm
bounding_boxes = podm.load_data_coco('tests/sample_2/groundtruths_coco.json')
```

PASCAL VOC Metrics

```python
import podm
gt_BoundingBoxes = podm.load_data('tests/sample_2/groundtruths.json')
pd_BoundingBoxes = podm.load_data('tests/sample_2/detections.json')
results = podm.get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)
```

ap, precision, recall, tp, fp, etc

```python
for cls, metric in actuals.items():
    label = m.category_id
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
from podm import MetricPerClass
mAP = MetricPerClass.mAP(results)
```

IoU

```python
box1 = Box(0., 0., 10., 10.)
box2 = Box(1., 1., 11., 11.)
Box.intersection_over_union(box1, box2)
```

## Implemented metrics

[Tutorial](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173)

-   Intersection Over Union (IOU)
-   TP and FP
    -   True Positive (TP): IOU ≥ *IOU threshold* (default: 0.5)
    -   False Positive (FP): IOU \< *IOU threshold* (default: 0.5)
-   Precision and Recall
-   Average Precision
    -   11-point AP
    -   all-point AP
