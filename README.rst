This project was forked from `rafaelpadilla/Object-Detection-Metrics <https://github.com/rafaelpadilla/Object-Detection-Metrics>`_.

Getting started
===============

Installing `object_detection_metrics`

.. code:: bash

    $ pip install object_detection_metrics

.. code:: python

    from podm.podm import get_pascal_voc_metrics

    gt_BoundingBoxes = ... # type: List[BoundingBox]
    pd_BoundingBoxes = ... # type: List[BoundingBox]
    results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)


Implemented metrics
===================

`Tutorial <https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173>`_.

- Intersection Over Union (IOU)
- TP and FP
    - True Positive (TP): IOU ≥ *IOU threshold* (default: 0.5)
    - False Positive (FP): IOU < *IOU threshold* (default: 0.5)
- Precision and Recall
- Average Precision
    - 11-point AP
    - all-point AP

