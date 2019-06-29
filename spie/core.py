"""This file contains the functions for estimating simultaneous prediction
intervals that were examined in the paper 'Simultaneous Prediction Intervals 
for Patient-Specific Survival Curves'.
"""
import numpy as np
import logging


def olshen(cloud, coverage, surv=False, B=10):
    """Estimate SPI via Olshen's method.

    :param cloud: A ``num_points`` by ``num_dims`` point cloud of samples.
    :type cloud: numpy.ndarray
    :param coverage: The prescribed probability or probabilities of a sample
                     lying within the SPI.
    :type coverage: Union[float, list]
    :param surv: Whether the samples are survival curves.
    :type surv: bool, optional
    :param B: The number of boostrapped datasets to use to determine ``k``.
    :type B: int, optional
    :return: A 2 by ``num_dims`` array where the index [0, i] gives the lower
             bound for interval i and the index [1, j] gives the upper bound
             for interval j. If a ``coverage`` is passed as a float an
             orthotope is returned. If ``coverage`` is passed as a list of
             floats a list of orthotopes is returned in corresponding order.
    :rtype: Union[numpy.ndarray, list]
    """
    cloud, fix = degenerate_fix_factory(cloud)
    bootstraps = np.random.choice(
        np.arange(cloud.shape[0]), size=(B, cloud.shape[0]))
    clouds = cloud[bootstraps]
    means = np.mean(clouds, axis=1)
    stds = np.std(clouds, axis=1, ddof=1)
    zscores = (clouds - means[:, np.newaxis, :]) / stds[:, np.newaxis, :]
    maxes = np.max(np.abs(np.nan_to_num(zscores)), axis=-1)
    mean = np.mean(cloud, axis=0)
    std = np.std(cloud, ddof=1, axis=0)

    def helper(p):
        k = np.quantile(maxes, q=p, interpolation='higher')
        lower_bounds = mean - k * std
        upper_bounds = mean + k * std
        orthotope = fix(np.array([lower_bounds, upper_bounds]))
        if surv:
            orthotope = surv_orthotope(orthotope)
        return orthotope

    return [helper(p)
            for p in coverage] if type(coverage) is list else helper(coverage)


def two_sided_olshen(cloud, coverage, surv=False, B=10):
    """Estimate SPI via two-sided Olshen's method.

    :param cloud: A ``num_points`` by ``num_dims`` point cloud of samples.
    :type cloud: numpy.ndarray
    :param coverage: The prescribed probability or probabilities of a sample
                     lying within the SPI.
    :type coverage: Union[float, list]
    :param surv: Whether the samples are survival curves.
    :type surv: bool, optional
    :param B: The number of boostrapped datasets to use to determine ``k``.
    :type B: int, optional
    :return: A 2 by ``num_dims`` array where the index [0, i] gives the lower
             bound for interval i and the index [1, j] gives the upper bound
             for interval j. If a ``coverage`` is passed as a float an
             orthotope is returned. If ``coverage`` is passed as a list of
             floats a list of orthotopes is returned in corresponding order.
    :rtype: Union[numpy.ndarray, list]
    """
    cloud, fix = degenerate_fix_factory(cloud)
    bootstraps = np.random.choice(
        np.arange(cloud.shape[0]), size=(B, cloud.shape[0]))
    clouds = cloud[bootstraps]
    maxes = np.zeros((B, cloud.shape[0]))
    for i, cloud_b in enumerate(clouds):
        zscores_ = np.zeros_like(cloud)
        for j, col in enumerate(cloud_b.T):
            median = np.median(col)
            above_mask = col >= median
            below_mask = col <= median
            above = col[above_mask]
            below = col[below_mask]
            zscores_[above_mask, j] = (above - median) / np.sqrt(
                (np.square(median - above)).sum() / (above.shape[0] - 1))
            zscores_[below_mask, j] = (below - median) / np.sqrt(
                (np.square(median - below)).sum() / (below.shape[0] - 1))
        maxes[i] = np.max(np.abs(np.nan_to_num(zscores_)), axis=-1)
    median_cloud = np.median(cloud, axis=0)
    sigma_minuses = np.zeros_like(median_cloud)
    sigma_pluses = np.zeros_like(median_cloud)
    for i, col in enumerate(cloud.T):
        median = median_cloud[i]
        above = col[col >= median]
        below = col[col <= median]
        sigma_pluses[i] = np.sqrt(
            np.square(above - median).sum() / (above.shape[0] - 1))
        sigma_minuses[i] = np.sqrt(
            np.square(median - below).sum() / (below.shape[0] - 1))

    def helper(p):
        k = np.quantile(maxes, q=p, interpolation='higher')
        upper_bounds = median_cloud + k * sigma_pluses
        lower_bounds = median_cloud - k * sigma_minuses
        orthotope = fix(np.array([lower_bounds, upper_bounds]))
        if surv:
            orthotope = surv_orthotope(orthotope)
        return orthotope

    return [helper(p)
            for p in coverage] if type(coverage) is list else helper(coverage)


def gspie(cloud, coverage, surv=False, val=0.5):
    """Estimate SPI via GSPIE.

    :param cloud: A ``num_points`` by ``num_dims`` point cloud of samples.
    :type cloud: numpy.ndarray
    :param coverage: The prescribed probability or probabilities of a sample
                     lying within the SPI.
    :type coverage: Union[float, list]
    :param surv: Whether the samples are survival curves.
    :type surv: bool, optional
    :param val: The proportion of the samples to use for accuracy validation.
                If ``cloud`` is small, a large validation proportion is needed
                to prevent overfitting. If ``cloud`` is large, small validation
                proportions also perform well.
    :type val: float, optional
    :return: A 2 by ``num_dims`` array where the index [0, i] gives the lower
             bound for interval i and the index [1, j] gives the upper bound
             for interval j. If a ``coverage`` is passed as a float an
             orthotope is returned. If ``coverage`` is passed as a list of
             floats a list of orthotopes is returned in corresponding order.
    :rtype: Union[numpy.ndarray, list]
    """
    cloud, fix = degenerate_fix_factory(cloud)
    cloud = np.random.permutation(cloud)
    opt_cloud = cloud[:int(cloud.shape[0] * (1 - val))]
    val_cloud = cloud[int(cloud.shape[0] * (1 - val)):]
    orthotope = np.array([opt_cloud.min(axis=0), opt_cloud.max(axis=0)])
    retract = update_retract(cloud, orthotope, orthotope.copy())
    counts = np.zeros_like(orthotope)
    for index in np.ndindex(*counts.shape):
        counts[index] = (orthotope[index] == opt_cloud[:, index[1]]).sum()
    ps = np.array(coverage) if type(coverage) is list else np.array([coverage])
    orthotopes = np.array(len(ps) * [fix(orthotope)])
    while np.abs(orthotope - retract).sum() > 0:
        opt_cloud, orthotope, retract, counts = update_orthotope(
            opt_cloud, orthotope, retract, counts)
        satisfaction = proportion(val_cloud, orthotope) >= ps
        if np.all(~satisfaction):
            break
        orthotopes[satisfaction] = surv_orthotope(
            fix(orthotope)) if surv else fix(orthotope)
    return orthotopes if type(coverage) is list else np.squeeze(orthotopes)


def pointwise_bonferroni(cloud, coverage, surv=False):
    """Estimate SPI by pointwise estimation with Bonferroni correction.

    :param cloud: A ``num_points`` by ``num_dims`` point cloud of samples.
    :type cloud: numpy.ndarray
    :param coverage: The prescribed probability or probabilities of a sample
                     lying within the SPI.
    :type coverage: Union[float, list]
    :param surv: Whether the samples are survival curves.
    :type surv: bool, optional
    :return: A 2 by ``num_dims`` array where the index [0, i] gives the lower
             bound for interval i and the index [1, j] gives the upper bound
             for interval j. If a ``coverage`` is passed as a float an
             orthotope is returned. If ``coverage`` is passed as a list of
             floats a list of orthotopes is returned in corresponding order.
    :rtype: Union[numpy.ndarray, list]
    """
    cloud, fix = degenerate_fix_factory(cloud)
    if type(coverage) is list:
        return [pointwise_bonferroni(cloud, p, surv) for p in coverage]
    adj_coverage = 1 - (((1 - coverage) / cloud.shape[1]) / 2)
    lower_bounds = np.quantile(cloud, 1 - adj_coverage, axis=0)
    upper_bounds = np.quantile(cloud, adj_coverage, axis=0)
    orthotope = fix(np.array([lower_bounds, upper_bounds]))
    if surv:
        orthotope = surv_orthotope(orthotope)
    return orthotope


def degenerate_fix_factory(cloud, epsilon=1e-6):
    """Handle dimensions along which there is negligible variance.

    :param cloud: A ``num_points`` by ``num_dims`` point cloud of samples.
    :type cloud: numpy.ndarray
    :param epsilon: The one-sided width of intervals along which
                    there is negligible variance.
    :type epsilon: float, optional
    :return: A length 2 tuple where the first entry is columns of ``cloud``
             along which there is nonnegligible variance and the second entry
             is a function that, given SPI for the first entry, returns SPI
             for ``cloud``.
    :rtype: tuple
    """
    degenerate = np.isclose(cloud.max(axis=0), cloud.min(axis=0))
    deg_cloud = cloud[:, degenerate]
    nondeg_cloud = cloud[:, ~degenerate]
    mean_deg = np.mean(deg_cloud, axis=0)
    lower_bounds = mean_deg - epsilon
    upper_bounds = mean_deg + epsilon

    def fix(nondeg_orthotope):
        fixed_orthotope = np.zeros((2, cloud.shape[1]))
        fixed_orthotope[:, ~degenerate] = nondeg_orthotope
        fixed_orthotope[:, degenerate] = np.array([lower_bounds, upper_bounds])
        return fixed_orthotope

    return nondeg_cloud, fix


def surv_orthotope(orthotope):
    """Project upper bounds and lower bounds into survival space.

    :param orthotope: A region representing SPI.
    :type orthotope: numpy.ndarray
    :return: A 2 by ``num_dims`` array where each component lies in [0, 1] and
             where the sequences given by the indices {[0, i]} and {[1, i]} are
             monotonically decreasing.
    :rtype: numpy.ndarray
    """
    orthotope = np.clip(orthotope, 0, 1)
    orthotope[0] = np.maximum.accumulate(orthotope[0][::-1])[::-1]
    orthotope[1] = np.minimum.accumulate(orthotope[1])
    return orthotope


def update_retract(cloud, orthotope, retract, index=None):
    """Update the retract of ``orthotope``.

    :param cloud: A ``num_points`` by ``num_dims`` point cloud of samples.
    :type cloud: numpy.ndarray
    :param orthotope: A 2 by ``num_dims`` region representing SPI.
    :type orthotope: numpy.ndarray
    :param retract: A retract of ``orthotope`` over every wall.
    :type retract: numpy.ndarray
    :param index: The index of ``retract`` to update.
    :type tuple:
    :return: An updated retract of ``orthotope``.
    :rtype: numpy.ndarray
    """
    if index is None:
        for idx in np.ndindex(*orthotope.shape):
            retract = update_retract(cloud, orthotope, retract, idx)
    else:
        if index[0] == 1:
            strict_ineq = lambda x, y: x < y
            extreme = lambda x: x.max()
        else:
            strict_ineq = lambda x, y: x > y
            extreme = lambda x: x.min()
        try:
            retract[index] = extreme(cloud[strict_ineq(
                cloud[:, index[1]], retract[index])][:, index[1]])
        except ValueError:
            pass
    return retract


def update_orthotope(cloud, orthotope, retract, counts):
    """Greedily apply one-wall retraction to ``orthotope``.

    :param cloud: A ``num_points`` by ``num_dims`` point cloud of samples.
    :type cloud: numpy.ndarray
    :param orthotope: A 2 by ``num_dims`` region representing SPI.
    :type orthotope: numpy.ndarray
    :param retract: A retract of ``orthotope`` over every wall.
    :type retract: numpy.ndarray
    :return: A greedy one-wall retract of ``orthotope``.
    :rtype: numpy.ndarray
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        diff = np.abs(orthotope - retract) / counts
    arg_best = np.unravel_index(diff.argmax(), diff.shape)
    new_wall = retract[arg_best]
    orthotope[arg_best] = new_wall
    counts[arg_best] = (new_wall == cloud[:, arg_best[1]]).sum()
    ineq = (lambda x, y: x <= y) if arg_best[0] == 1 else (lambda x, y: x >= y)
    cloud = cloud[ineq(cloud[:, arg_best[1]], new_wall)]
    retract = update_retract(cloud, orthotope, retract, arg_best)
    return cloud, orthotope, retract, counts


def proportion(cloud, orthotope):
    """Compute proportion of ``cloud`` lying within ``orthotope``.

    :param cloud: A ``num_points`` by ``num_dims`` point cloud of samples.
    :type cloud: numpy.ndarray
    :param orthotope: A 2 by ``num_dims`` region representing SPI.
    :type orthotope: numpy.ndarray
    :return: The proportion of ``cloud`` lying within ``orthotope``.
    :rtype: float
    """
    return (np.all(orthotope[0] <= cloud, axis=-1) & np.all(
        cloud <= orthotope[1], axis=-1)).sum() / cloud.shape[0]

