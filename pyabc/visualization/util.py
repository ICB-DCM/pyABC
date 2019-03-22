def get_histories_and_ids(histories, ids=None):
    """
    Make sure both are lists of the same length.
    If ids is None,  
    """
    if isinstance(histories, History):
        histories = [histories]
    n_history = len(histories)
    if ids is None:
        ids = [None] * n_history
    n_id = len(ids)
    if n_history == 1 and n_id > 1:
        histories =  [histories[0]] * n_id
    if n_id == 1 and n_history > 1:
        ids = [ids[0]] * n_history

    return histories, ids
