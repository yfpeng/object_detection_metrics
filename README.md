[![Latest version on PyPI](https://img.shields.io/pypi/v/object_detection_metrics.svg)](https://pypi.python.org/pypi/object_detection_metrics)
[![License](https://img.shields.io/pypi/l/object_detection_metrics.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Downloads](https://img.shields.io/pypi/dm/object_detection_metrics.svg)](https://pypi.python.org/pypi/object_detection_metrics)
[![Pythong version](https://img.shields.io/pypi/pyversions/object_detection_metrics)](https://pypi.python.org/pypi/object_detection_metrics)
[![Hits](https://hits.dwyl.com/yfpeng/object_detection_metrics.svg)](https://hits.dwyl.com/yfpeng/object_detection_metrics)


This project was forked from [rafaelpadilla/Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics).

## Getting started

Installing `object_detection_metrics`

```bash
$ pip install object_detection_metrics
```

```python
from podm.podm import get_pascal_voc_metrics

gt_BoundingBoxes = ... # type: List[BoundingBox]
pd_BoundingBoxes = ... # type: List[BoundingBox]
results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)
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
