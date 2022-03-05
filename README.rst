.. image:: https://img.shields.io/pypi/v/object_detection_metrics.svg
   :target: https://pypi.python.org/pypi/object_detection_metrics
   :alt: Latest version on PyPI

.. image:: https://img.shields.io/pypi/l/object_detection_metrics.svg
   :alt: License
   :target: https://opensource.org/licenses/BSD-3-Clause

.. image:: https://hits.dwyl.com/yfpeng/object_detection_metrics.svg
   :alt: Hits
   :target: https://hits.dwyl.com/yfpeng/object_detection_metrics

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

* Intersection Over Union (IOU)
* TP and FP

  * True Positive (TP): IOU ≥ *IOU threshold* (default: 0.5)
  * False Positive (FP): IOU < *IOU threshold* (default: 0.5)
  
* Precision and Recall
* Average Precision

  * 11-point AP
  * all-point AP
