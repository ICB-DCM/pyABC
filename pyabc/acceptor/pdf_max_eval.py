import numpy as np


def pdf_max_use_default(**kwargs):
    """
    Just use the pdf_max value passed via default, usually originating
    from the distance function.
    """
    pdf_max = kwargs['default']
    return pdf_max


def pdf_max_use_max_found(**kwargs):
    """
    Take as pdf_max the value found so far in history, and in
    `get_weighted_distances`.

    .. note::
        When the pdf_max value used is not really a supremum on the
        distribution, then the distribution upon which acceptance is
        based is effectively not the true distribution, but flattened
        out with a uniform distribution at high values. This introduces
        an approximation error.
    """
    pdf_maxs = kwargs['pdf_maxs']
    get_weighted_distances = kwargs['get_weighted_distances']

    # execute function
    df = get_weighted_distances()

    pdfs = np.array(df['distance'])

    if len(pdf_maxs) == 0:
        max_prev = - np.inf
    else:
        max_prev = max(pdf_maxs.values())
    max_prev_iter = max(pdfs)

    pdf_max = max(max_prev, max_prev_iter)

    return pdf_max
