"""This file contains functions for generating synthetic survival and time
series data, and functions that, using this synthetic data, generate
example SPI, and test the accuracy and tightness of SPIEs.
"""
import numpy as np
import matplotlib.pyplot as plt
from spie.core import olshen, two_sided_olshen, gspie, pointwise_bonferroni, proportion


def generate_survival_curves(num_points, num_dims):
    """Generate a set of survival curves from a common distribution.

    :param num_points: The number of samples to take.
    :type num_points: int
    :param num_dims: The number of time points.
    :type num_dims: int
    :return: A ``num_points`` by ``num_dims`` array representing a collection
             of survival curves sampled from a shared distribution.
    :rtype: numpy.ndarray
    """
    pmf = np.random.dirichlet(
        alpha=np.random.exponential(size=num_dims), size=num_points)
    return np.clip(np.cumsum(pmf, axis=-1)[:, ::-1], 0, 1)


def generate_time_series(num_points, num_dims):
    """Generate a set of time series from a common distribution.

    :param num_points: The number of samples to take.
    :type num_points: int
    :param num_dims: The number of time points.
    :type num_dims: int
    :return: A ``num_points`` by ``num_dims`` array representing a collection
             of time series sampled from a shared distribution.
    :rtype: numpy.ndarray
    """
    phase = np.random.randn()
    period = np.random.uniform()
    times = np.linspace(0, 10, num_dims)
    scale = np.random.exponential(size=num_points)
    return np.outer(scale, np.sin(times / period + phase))


def example_spi(num_points, num_dims, p, surv):
    """Draw a graph of example estimated SPIs.

    :param num_points: The number of samples to take.
    :type num_points: int
    :param num_dims: The number of time points.
    :type num_dims: int
    :param p: The prescribed coverage probability.
    :type p: float
    :param surv: Whether to use survival curves.
    :type surv: bool
    """
    if surv:
        cloud = generate_survival_curves(num_points, num_dims)
        ylabel = 'Survival Probability'
    else:
        cloud = generate_time_series(num_points, num_dims)
        ylabel = 'Value'
    orthotope1 = olshen(cloud, p, surv)
    orthotope2 = two_sided_olshen(cloud, p, surv)
    orthotope3 = gspie(cloud, p, surv)
    orthotope4 = pointwise_bonferroni(cloud, p, surv)
    x = np.arange(num_dims)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex=True, sharey=True)
    ax1.fill_between(x, orthotope1[0], orthotope1[1], color='b')
    ax1.set_ylabel(ylabel)
    ax1.set_title('Olshen')
    ax2.fill_between(x, orthotope2[0], orthotope2[1], color='orange')
    ax2.set_title('Two-Sided Olshen')
    ax3.fill_between(x, orthotope3[0], orthotope3[1], color='g')
    ax3.set_xlabel('Time')
    ax3.set_title('GSPIE')
    ax3.set_ylabel(ylabel)
    ax4.fill_between(x, orthotope4[0], orthotope4[1], color='r')
    ax4.set_xlabel('Time')
    ax4.set_title('Pointwise Bonferroni')
    plt.show()


def tightness(num_points, num_dims, p, surv, num_runs=100):
    """Draw a graph summarizing the tightness of estimated SPIs.

    :param num_points: The number of samples to take.
    :type num_points: int
    :param num_dims: The number of time points.
    :type num_dims: int
    :param p: The prescribed coverage probability.
    :type p: float
    :param surv: Whether to use survival curves.
    :type surv: bool
    :param num_runs: The number of estimated SPIs to sample for each SPIE.
    :type num_runs: int, optional
    """
    mu1 = []
    mu2 = []
    mu3 = []
    mu4 = []
    for t in range(num_runs):
        if surv:
            cloud = generate_survival_curves(num_points, num_dims)
        else:
            cloud = generate_time_series(num_points, num_dims)
        mu1.append(average_width(olshen(cloud, p, surv)))
        mu2.append(average_width(two_sided_olshen(cloud, p, surv)))
        mu3.append(average_width(gspie(cloud, p, surv)))
        mu4.append(average_width(pointwise_bonferroni(cloud, p, surv)))
    plt.boxplot([mu1, mu2, mu3, mu4])
    plt.xticks([1, 2, 3, 4],
               ['Olshen', 'Two-Sided Olshen', 'GSPIE', 'Pointwise Bonferroni'])
    plt.ylabel('Average Width')
    plt.show()


def accuracy(num_points, num_dims, p, surv, num_runs=100, num_test=10000):
    """Draw a graph summarizing the accuracy of estimated SPIs.

    :param num_points: The number of samples to take.
    :type num_points: int
    :param num_dims: The number of time points.
    :type num_dims: int
    :param p: The prescribed coverage probability.
    :type p: float
    :param surv: Whether to use survival curves.
    :type surv: bool
    :param num_runs: The number of estimated SPIs to sample for each SPIE.
    :type num_runs: int, optional
    :param num_test: The number os samples with which to judge observed
                     coverage.
    :type num_test: int, optional
    """
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    for t in range(num_runs):
        if surv:
            cloud = generate_survival_curves(num_points + num_test, num_dims)
        else:
            cloud = generate_time_series(num_points + num_test, num_dims)
        opt_cloud = cloud[:num_points]
        test_cloud = cloud[num_points:]
        p1.append(proportion(test_cloud, olshen(opt_cloud, p, surv)))
        p2.append(proportion(test_cloud, two_sided_olshen(opt_cloud, p, surv)))
        p3.append(proportion(test_cloud, gspie(opt_cloud, p, surv)))
        p4.append(
            proportion(test_cloud, pointwise_bonferroni(opt_cloud, p, surv)))
    plt.boxplot([p1, p2, p3, p4])
    plt.xticks([1, 2, 3, 4],
               ['Olshen', 'Two-Sided Olshen', 'GSPIE', 'Pointwise Bonferroni'])
    plt.ylabel('Observed Coverage')
    plt.xlabel(f"{p} Prescribed Coverage")
    plt.show()


def average_width(orthotope):
    """Compute the average width of the intervals of an orthotope.

    :param orthotope: A 2 by ``num_dims`` region representing SPI.
    :type orthotope: numpy.ndarray
    :return: The average width of the SPI induced by ``orthotope``.
    :rtype: float
    """
    return np.mean(orthotope[1] - orthotope[0])
