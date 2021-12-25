"""Sensitivity sankey flow plot."""

from typing import Callable, Dict, List, Union

import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    pass

import pyabc.distance
import pyabc.predictor
import pyabc.storage
import pyabc.sumstat
import pyabc.util

from . import colors


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
    sumstat_color: Callable[[str], str] = None,
    par_color: Callable[[str], str] = None,
    node_kwargs: dict = None,
    layout_kwargs: dict = None,
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
    sumstat_color:
        Callable assigning a color code for a given flattened summary
        statistic id.
    par_color:
        Callable assigning a color code for a given parameter or parameter
        transformation id.
    node_kwargs:
        Arguments for `go.Sankey.nodes`.
    layout_kwargs:
        Arguments for `fig.update_layout`.

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

    if node_kwargs is None:
        node_kwargs = {}
    node_kwargs_all = {
        "pad": 15,
        "thickness": 20,
        "line": {
            "color": "black",
            "width": 0.5,
        },
    }
    node_kwargs_all.update(node_kwargs)

    if layout_kwargs is None:
        layout_kwargs = {}
    layout_kwargs_all = {
        "title_x": 0.5,
        "font_size": 12,
        "template": "simple_white",
    }
    layout_kwargs_all.update(layout_kwargs)

    # extract latest fitting time point
    if not isinstance(t, int):
        info_dict = pyabc.storage.load_dict_from_json(t)
        t = max(info_dict.keys())

    # initialize objects
    sample = pyabc.Sample.from_population(h.get_population(t=max(0, t - 1)))
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
    x, y, weights, use_ixs, x0 = (
        ret[key] for key in ("x", "y", "weights", "use_ixs", "x0")
    )

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

    # node colors

    sumstat_color_dict = {}

    def default_sumstat_color(id_: str):
        # extract summary statistic name
        base = id_.split(":")[0]

        if base in sumstat_color_dict:
            return sumstat_color_dict[base]

        i = len(sumstat_color_dict)
        color = getattr(
            colors,
            f"{colors.REDSORANGES[i % len(colors.REDSORANGES)]}400",
        )
        sumstat_color_dict[base] = color
        return color

    par_color_dict = {}

    def default_par_color(id_: str):
        # extract parameter base name
        #  this may require customization
        if "^" in id_:
            base = id_.split("^")[0]
        elif "(" in id_:
            base = id_.split("(")[1].split(")")[0]
        else:
            base = id_.split("_")[0]

        if base in par_color_dict:
            return par_color_dict[base]

        i = len(par_color_dict)
        color = getattr(
            colors,
            f"{colors.GREENSBLUES[i % len(colors.GREENSBLUES)]}400",
        )
        par_color_dict[base] = color
        return color

    if sumstat_color is None:
        sumstat_color = default_sumstat_color
    if par_color is None:
        par_color = default_par_color

    node_color = [
        *[sumstat_color(id_) for id_ in sumstat.get_ids()],
        *[par_color(id_) for id_ in par_trafo.get_ids()],
    ]

    # generate figure
    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "label": node_label,
                    "color": node_color,
                    **node_kwargs_all,
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
        width=width,
        height=height,
        **layout_kwargs_all,
    )

    return fig
