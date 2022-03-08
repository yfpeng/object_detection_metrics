from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from podm.metrics import MethodAveragePrecision, MetricPerClass


def plot_precision_recall_curve(result: MetricPerClass,
                                dest,
                                method: MethodAveragePrecision = MethodAveragePrecision.AllPointsInterpolation,
                                show_ap: bool=False,
                                show_interpolated_precision: bool=False):
    """PlotPrecisionRecallCurve
    Plot the Precision x Recall curve for a given class.
    Args:
        result: metric per class
        dest: the plot will be saved as an image in this path
        method: method for interpolation
        show_ap: if True, the average precision value will be shown in the title of
            the graph (default = False);
        show_interpolated_precision (optional): if True, it will show in the plot the interpolated
            precision (default = False);
    """
    mpre = result.interpolated_precision
    mrec = result.interpolated_recall

    plt.close()
    if show_interpolated_precision:
        if method == MethodAveragePrecision.AllPointsInterpolation:
            plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')
        elif method == MethodAveragePrecision.ElevenPointsInterpolation:
            # Uncomment the line below if you want to plot the area
            # plt.plot(mrec, mpre, 'or', label='11-point interpolated precision')
            # Remove duplicates, getting only the highest precision of each recall value
            nrec = []
            nprec = []
            for idx in range(len(mrec)):
                r = mrec[idx]
                if r not in nrec:
                    idxEq = np.argwhere(mrec == r)
                    nrec.append(r)
                    nprec.append(max([mpre[int(id)] for id in idxEq]))
            plt.plot(nrec, nprec, 'or', label='11-point interpolated precision')
    plt.plot(result.recall, result.precision, '-o')
    # add a new penultimate point to the list (mrec[-2], 0.0)
    # since the last line segment (and respective area) do not affect the AP value
    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
    area_under_curve_y = mpre[:-1] + [0.0] + [mpre[-1]]
    plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if show_ap:
        ap_str = "{0:.2f}%".format(result.ap * 100)
        plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (result.label, ap_str))
    else:
        plt.title('Precision x Recall curve \nClass: %s' % result.label)
    # plt.legend(shadow=True)
    plt.grid()
    plt.savefig(str(dest))


def plot_precision_recall_curve_all(results: Dict[Any, MetricPerClass],
                                    dest_dir,
                                    method: MethodAveragePrecision = MethodAveragePrecision.AllPointsInterpolation,
                                    show_ap: bool=False,
                                    show_interpolated_precision: bool=False):
    """
    Plot the Precision x Recall curve for a given class.

    Args:
        results: metric per class
        dest_dir: the plot will be saved as an image in this path
        method: method for interpolation
        show_ap: if True, the average precision value will be shown in the title of
            the graph (default = False);
        show_interpolated_precision (optional): if True, it will show in the plot the interpolated
            precision (default = False);
    """
    for label, result in results.items():
        dest = str(dest_dir / (label + '_pr.png'))
        try:
            plot_precision_recall_curve(result, dest, method, show_ap, show_interpolated_precision)
        except:
            print(f'{label}: Cannot plot')
