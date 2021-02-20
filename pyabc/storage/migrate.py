import click
import os
import shutil
from alembic.config import Config
from alembic import command

SQLITE_STR = "sqlite:///"


@click.command(
    help="**Migrate pyABC database**\n\n"
         "Sometimes, changes to pyABC's storage format are unavoidable. "
         "In such cases, this tool is intended to allow migrating databases "
         "between versions. "
         "To avoid data loss in the unlikely case that migration does not "
         "work properly, we recommend keeping the original file by specifying "
         "a different destination.\n\n"
         "Note: Migration currently only supports sqlite databases.")
@click.option(
    '--src', required=True, type=str, help="Database to convert (filename)")
@click.option(
    '--dst', required=True, type=str, help="Destination (filename)")
@click.option(
    '--version', default='head', type=str, help="Target database version")
def migrate(src: str, dst: str, version: str) -> None:
    """Migrate database.

    Parameters
    ----------
    src: Source
    dst: Destination
    version: Version to migrate to
    """
    # to file paths if URLs
    if src.startswith(SQLITE_STR):
        src = src[len(SQLITE_STR):]
    if dst.startswith(SQLITE_STR):
        dst = dst[len(SQLITE_STR):]

    # copy file
    if src != dst:
        if os.path.exists(dst):
            print(f"Error: Destination file {dst} exists already.")
            return
        # copy source to destination
        shutil.copyfile(src=src, dst=dst)

    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg = Config(os.path.join(base_path, 'alembic.ini'))
    cfg.set_main_option(
        'script_location', os.path.join(base_path, 'migrations'))
    cfg.set_main_option('sqlalchemy.url', SQLITE_STR + dst)

    command.upgrade(cfg, version)
