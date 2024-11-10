import importlib
import importlib.metadata
from pathlib import Path

from appdirs import user_cache_dir


def _get_cache_directory() -> Path:
    directory = Path(
        user_cache_dir(
            appname="pyFishSenseDev",
            appauthor="Engineers for Exploration",
            version=importlib.metadata.version("pyfishsensedev"),
        )
    )

    if not directory.exists():
        directory.mkdir(exist_ok=True, parents=True)

    return directory


CACHE_DIRECTORY = _get_cache_directory()
