from pathlib import Path

from appdirs import user_cache_dir

from pyfishsensedev import __version__


def _get_cache_directory() -> Path:
    directory = Path(
        user_cache_dir(
            appname="pyFishSenseDev",
            appauthor="Engineers for Exploration",
            version=__version__,
        )
    )

    if not directory.exists():
        directory.mkdir(exist_ok=True, parents=True)

    return directory


CACHE_DIRECTORY = _get_cache_directory()
