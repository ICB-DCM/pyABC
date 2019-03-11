import click
from .df_to_file import to_file
from .history import History


@click.command(name="abc-dump")
@click.option("--db", help="The db connection or file in which the pyABC data "
                           "is stored and from from which we want to to dump "
                           "to a file")
@click.option("--out", help="The file to which to dump")
@click.option("--format", 'out_format', default="feather",
              help="The format to which to dump, e.g. feather, "
                   "csv, hdf, json, html, msgpack, stata")
@click.option("--generation", default="last",
              help="The generation to dump. Can be "
                   "\"all\" or \"last\" or an integer "
                   "number")
@click.option("--model", default="all",
              help="The model number to dump. Defaults"
                   "to \"all\", which means all models are"
                   "dumped. Can be an integer, which"
                   "identifies the model number. Note that the first model "
                   "has number 0.")
@click.option("--id", default=1, type=int,
              help="The ABC-SMC run id which to dump. "
                   "Defaults to 1")
@click.option("--tidy", default=True, type=bool,
              help="If True, the individual parameter and summary statistic "
                   "names are pivoted. Only works for a single model and "
                   "time point.")
def main(db, out, out_format, generation="last",
         model=None, id=1, tidy=True):  # pylint: disable=W0622
    """
    Export from the SQLite database to different table formats.
    The function pyabc.History.get_population_extended() is used
    to extract the data to export from the database, see there for
    further details.
    """
    # check if db is a file or SQLAlchemy identifier
    if ":///" not in db:
        db = "sqlite:///" + db

    # parse model
    if model == "all":
        m = None
    else:
        m = int(model)

    # parse generation
    t = generation
    try:
        t = int(t)
    except ValueError:
        pass

    # open database
    history = History(db)
    history.id = id

    # extract dataframe for abc run id, model m, generation t
    df = history.get_population_extended(m=m, t=t, tidy=tidy)

    # convert dataframe to output file format
    to_file(df, out, file_format=out_format)
