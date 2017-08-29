import click
from .df_to_file import to_file
from .history import History


@click.command(name="abc-dump")
@click.option("--db", help="The db connection or file in which the pyABC data "
                           "is stored and from from which we want to to dump "
                           "to a file")
@click.option("--out", help="The file to which to dump")
@click.option("--format", default="feather",
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
def main(db, out, format, generation="last", model=None, id=1):
    if ":///" not in db:  # check if db is a file or SQLAlchemy identifier
        db = "sqlite:///" + db

    if model == "all":
        m = None
    else:
        m = int(model)

    t = generation
    try:
        t = int(t)
    except ValueError:
        pass

    history = History(db)
    history.id = id
    df = history.get_population_extended(m=m, t=t, tidy=True)
    to_file(df, out, file_format=format)


if __name__ == "__main__":
    main()
