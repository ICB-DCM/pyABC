"""Migrate database."""

import os
import shutil

import click

try:
    from alembic import command
    from alembic.config import Config
except ImportError:
    Config = command = None

SQLITE_STR = 'sqlite:///'


def _to_db_file(db: str) -> str:
    """Normalize a database identifier to a file name.

    Parameters
    ----------
    db: Database file name, or sqlite URL ``sqlite:///<file>``.

    Returns
    -------
    db_file: The database file name.

    Raises
    ------
    ValueError: If a URL of a dialect other than sqlite is passed.
    """
    if db.startswith(SQLITE_STR):
        return db[len(SQLITE_STR) :]
    if '://' in db:
        raise ValueError(
            f'Cannot handle database identifier {db}: migration currently '
            f'only supports sqlite databases, i.e. either a file name, or a '
            f'URL of the form {SQLITE_STR}<file>.'
        )
    return db


def _alembic_config(db: str) -> 'Config':
    """Create the alembic configuration operating on a database.

    Parameters
    ----------
    db: Database the migrations are applied to, either as a file name or as a
        sqlite URL ``sqlite:///<file>``.

    Returns
    -------
    cfg: The alembic configuration.
    """
    # config base path
    base_path = os.path.dirname(os.path.abspath(__file__))
    # read configuration file
    cfg = Config(os.path.join(base_path, 'alembic.ini'))
    # set absolute script location path
    cfg.set_main_option(
        'script_location', os.path.join(base_path, 'migrations')
    )
    # set target database file
    cfg.set_main_option('sqlalchemy.url', SQLITE_STR + _to_db_file(db))
    return cfg


@click.command(
    help='**Migrate pyABC database**\n\n'
    "Sometimes, changes to pyABC's storage format are unavoidable. "
    'In such cases, this tool is intended to allow migrating databases '
    'between versions. '
    'To avoid data loss in the unlikely case that migration does not '
    'work properly, we recommend keeping the original file by specifying '
    'a different destination.\n\n'
    'Note: Migration currently only supports sqlite databases.'
)
@click.option(
    '--src', required=True, type=str, help='Database to convert (filename)'
)
@click.option('--dst', required=True, type=str, help='Destination (filename)')
@click.option(
    '--version', default='head', type=str, help='Target database version'
)
def migrate(src: str, dst: str, version: str) -> None:
    """Migrate database.

    Parameters
    ----------
    src: Source, either a file name or a sqlite URL
    dst: Destination, either a file name or a sqlite URL
    version: Version to migrate to
    """
    if Config is None or command is None:
        print(
            'Error: migration tools not installed. Please run '
            '`pip install pyabc[migrate]`'
        )
        return

    # to file paths if URLs
    try:
        src, dst = _to_db_file(src), _to_db_file(dst)
    except ValueError as e:
        print(f'Error: {e}')
        return

    # copy file
    if src != dst:
        if os.path.exists(dst):
            print(f'Error: Destination file {dst} exists already.')
            return
        # copy source to destination
        shutil.copyfile(src=src, dst=dst)

    # run the actual upgrade
    command.upgrade(_alembic_config(dst), version)
