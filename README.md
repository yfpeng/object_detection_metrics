This project was borrowed from [rafaelpadilla/Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)

## Getting started

Installing `pdom`

```bash
$ pip install pdom
```

```python
from podm.podm import get_pascal_voc_metrics

gt_BoundingBoxes = ... # type: List[BoundingBox]
pd_BoundingBoxes = ... # type: List[BoundingBox]
results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)
```

## Implemented metrics

[Tutorial](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173)

* Intersection Over Union (IOU)
* TP and FP
    * True Positive (TP): IOU ≥ _IOU threshold_ (default: 0.5)
    * False Positive (FP): IOU < _IOU threshold_ (default: 0.5)
* Precision and Recall
* Average Precision
    * 11-point AP
    * all-point AP

