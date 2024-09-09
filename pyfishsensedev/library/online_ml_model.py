from abc import ABC, abstractmethod
from pathlib import Path

from requests import get

from pyfishsensedev.library.paths import CACHE_DIRECTORY


class OnlineMLModel(ABC):
    def __init__(self) -> None:
        super().__init__()

    @property
    def _model_cache_path(self) -> Path:
        return CACHE_DIRECTORY / "models"

    @property
    @abstractmethod
    def _model_path(self) -> Path:
        raise NotImplementedError

    @property
    @abstractmethod
    def _model_url(self) -> str:
        raise NotImplementedError

    def _download_file(self, url: str, path: Path) -> Path:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

            response = get(url)
            with path.open("wb") as file:
                file.write(response.content)

        return path.absolute()

    def download_model(self) -> Path:
        return self._download_file(
            self._model_url,
            self._model_path,
        ).as_posix()
