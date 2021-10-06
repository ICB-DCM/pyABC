"""Sensitivity sankey flow plot."""

import numpy as np
from typing import Dict, List, Union
try:
    import plotly.graph_objects as go
except ImportError:
    pass

import pyabc.predictor
import pyabc.sumstat
import pyabc.storage
import pyabc.util
import pyabc.distance


def plot_sensitivity_sankey(
    info_sample_log_file: str,
    t: Union[int, str],
    h: pyabc.storage.History,
    predictor: pyabc.predictor.Predictor,
    par_trafo: pyabc.util.ParTrafoBase = None,
    sumstat: pyabc.sumstat.Sumstat = None,
    subsetter: pyabc.sumstat.Subsetter = None,
    feature_normalization: str = pyabc.distance.InfoWeightedPNormDistance.MAD,
    normalize_by_par: bool = True,
    fd_deltas: Union[List[float], float] = None,
    scale_weights: Dict[int, np.ndarray] = None,
    title: str = "Data-parameter sensitivities",
    width: float = None,
    height: float = None,
):
    """Plot sensitivity matrix as a Sankey flow plot.

    This visualization allows to analyze the parameter-data relationships,
    unraveling how informative data points are considered, and of which
    parameters.

    We use `plotly` to generate this plot, which may need to be separately
    installed, e.g. via `pip install plotly`.

    Customization of e.g. colors, to group by e.g. observable or parameter,
    is easily possible, however at the moment not implemented.

    To store the generated figure, use e.g.
    `fig.write_image({filename}.{format})`.

    Parameters
    ----------
    info_sample_log_file:
        Base of summary statistics, parameters, and weights files names
        containing samples used for model training, e.g. as generated via the
        `info_sample_log_file` argument of
        :class:`pyabc.distance.InfoWeightedPNormDistance`.
    t:
        Time point at which the training was performed. Can also be an
        `info_log_file`, from which then the latest time is extracted.
    h:
        History object. Required to extract observed data.
    predictor:
        Predictor model.
    par_trafo:
        Parameter transformations applied. Should be the same as applied for
        generation of the training sample.
    sumstat:
        Summary statistic used on the raw model outputs. Defaults to identity.
    subsetter:
        Subset generation method used. Defaults to identity.
    feature_normalization:
        Feature normalization method, as in
        :class:`pyabc.distance.InfoWeightedPNormDistance`.
    normalize_by_par:
        Whether to normalize sensitivities by parameter (transformation).
    fd_deltas:
        Finite difference step sizes to evaluate.
    scale_weights:
        Scale weights. Only needed if `feature_normalization is "weights"`.
    title:
        Plot title.
    width:
        Plot width.
    height:
        Plot height.

    Returns
    -------
    fig:
        Generated figure.
    """
    # default arguments
    if par_trafo is None:
        par_trafo = pyabc.util.ParTrafo()
    if sumstat is None:
        sumstat = pyabc.sumstat.IdentitySumstat()
    if subsetter is None:
        subsetter = pyabc.sumstat.IdSubsetter()

    # extract latest fitting time point
    if not isinstance(t, int):
        info_dict = pyabc.storage.load_dict_from_json(t)
        t = max(info_dict.keys())

    # initialize objects
    sample = pyabc.Sample.from_population(h.get_population(t=max(0, t-1)))
    data = h.observed_sum_stat()
    sumstat.initialize(t=0, get_sample=lambda: sample, x_0=data, total_sims=0)
    par_keys = list(h.get_distribution()[0].columns)
    par_trafo.initialize(keys=par_keys)

    # read training samples
    sumstats, parameters, weights = [
        np.load(f"{info_sample_log_file}_{t}_{var}.npy")
        for var in ["sumstats", "parameters", "weights"]
    ]

    s_0 = sumstat(data)

    # normalize data
    ret = pyabc.distance.InfoWeightedPNormDistance.normalize_sample(
        sumstats=sumstats,
        parameters=parameters,
        weights=weights,
        s_0=s_0,
        t=t,
        subsetter=subsetter,
        feature_normalization=feature_normalization,
        scale_weights=scale_weights,
    )
    x, y, weights, use_ixs, x0 = \
        (ret[key] for key in ("x", "y", "weights", "use_ixs", "x0"))

    # learn predictor model
    predictor.fit(x=x, y=y, w=weights)

    # calculate all sensitivities of the predictor at the observed data
    sensis = pyabc.distance.InfoWeightedPNormDistance.calculate_sensis(
        predictor=predictor,
        fd_deltas=fd_deltas,
        x0=x0,
        n_x=x.shape[1],
        n_y=y.shape[1],
        par_trafo=par_trafo,
        normalize_by_par=normalize_by_par,
    )

    # plot stuff

    n_in, n_out = sensis.shape

    # define links via lists of sources, targets, and values indicating
    #  connection strengths
    source = []
    target = []
    value = []
    node_label = [*sumstat.get_ids(), *par_trafo.get_ids()]
    for i_in in range(n_in):
        for i_out in range(n_out):
            source.append(i_in)
            target.append(n_in + i_out)
            value.append(sensis[i_in, i_out])

    # generate figure
    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 15,
                    "thickness": 20,
                    "line": {"color": "black", "width": 0.5},
                    "label": node_label,
                },
                link={
                    "source": source,
                    "target": target,
                    "value": value,
                },
            ),
        ],
    )

    # layout prettifications
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        font_size=12,
        width=width,
        height=height,
        template="simple_white",
    )

    return fig
