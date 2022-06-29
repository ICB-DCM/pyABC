import pandas as pd
from io import StringIO, BytesIO
import csv
import numpy as np
import logging

logger = logging.getLogger("ABC.History")

try:
    import pyarrow
    import pyarrow.parquet as parquet
except ImportError:
    pyarrow = parquet = None
    logger.warning(
        "Can't find pyarrow, thus falling back to csv to store pandas data.",
    )


class DataFrameLoadException(Exception):
    pass


def df_to_bytes_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(quoting=csv.QUOTE_NONNUMERIC).encode()


def df_from_bytes_csv(bytes_: bytes) -> pd.DataFrame:
    try:
        s = StringIO(bytes_.decode())
        s.seek(0)
        return pd.read_csv(
            s, index_col=0, header=0, float_precision="round_trip",
            quotechar='"')
    except UnicodeDecodeError:
        raise DataFrameLoadException("Not a DataFrame")


def df_to_bytes_msgpack(df: pd.DataFrame) -> bytes:
    return df.to_msgpack()


def df_from_bytes_msgpack(bytes_: bytes) -> pd.DataFrame:
    try:
        df = pd.read_msgpack(BytesIO(bytes_))
    except UnicodeDecodeError:
        raise DataFrameLoadException("Not a DataFrame")
    if not isinstance(df, pd.DataFrame):
        raise DataFrameLoadException("Not a DataFrame")
    return df


def df_to_bytes_json(df: pd.DataFrame) -> bytes:
    return df.to_json().encode()


def df_from_bytes_json(bytes_: bytes) -> pd.DataFrame:
    return pd.read_json(bytes_.decode())


def df_to_bytes_parquet(df: pd.DataFrame) -> bytes:
    """
    pyarrow parquet is the standard conversion method of pandas
    DataFrames since pyabc 0.9.14, because msgpack became
    deprecated in pandas 0.25.0.
    """
    b = BytesIO()
    table = pyarrow.Table.from_pandas(df)
    parquet.write_table(table, b)
    b.seek(0)
    return b.read()


def df_from_bytes_parquet(bytes_: bytes) -> pd.DataFrame:
    """
    Since pyabc 0.9.14, pandas DataFrames are converted using
    pyarrow parquet. If the conversion to DataFrame fails,
    then `df_from_bytes_msgpack_` is tried, which was the formerly
    used method. This is in particular useful for databases that
    still employ the old format. In case errors occur here, it may
    be necessary to use a pandas version prior to 0.25.0.
    """
    try:
        b = BytesIO(bytes_)
        table = parquet.read_table(b)
        df = table.to_pandas()
    except pyarrow.lib.ArrowIOError:
        df = df_from_bytes_msgpack(bytes_)
    return df


def df_to_bytes_np_records(df: pd.DataFrame) -> bytes:
    b = BytesIO()
    rec = df.to_records()
    np.save(b, rec, allow_pickle=False)
    b.seek(0)
    return b.read()


def df_from_bytes_np_records(bytes_: bytes) -> pd.DataFrame:
    b = BytesIO(bytes_)
    rec = np.load(b)
    df = pd.DataFrame.from_records(rec, index="index")
    return df


def df_to_bytes(df: pd.DataFrame) -> bytes:
    """Write dataframe to bytes.

    Use pyarrow parquet if available, otherwise csv.
    """
    if pyarrow is None:
        return df_to_bytes_csv(df)
    return df_to_bytes_parquet(df)


def df_from_bytes(bytes_: bytes) -> pd.DataFrame:
    """Read dataframe from bytes.

    If pyarrow is not available, try csv.
    """
    if pyarrow is None:
        return df_from_bytes_csv(bytes_)
    return df_from_bytes_parquet(bytes_)
