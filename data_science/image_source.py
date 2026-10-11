"""Kaggle as the source of article images.

Shared by the pipeline (`fetch_images.py`, fetches a season's images in bulk) and the API (fetches a missing image the
first time the app shows it). Deliberately light: no pandas or model code, so the API can import it.
"""
import shutil
import tempfile
import zipfile
from pathlib import Path

COMPETITION = "h-and-m-personalized-fashion-recommendations"


class ImageSourceUnavailable(RuntimeError):
    """The image source (Kaggle) can't be used, e.g. because there is no login."""


def kaggle_path(article_id: str) -> str:
    """'0768921001' → 'images/076/0768921001.jpg' (the layout of the Kaggle image set)."""
    return f"images/{article_id[:3]}/{article_id}.jpg"


def connect():
    """An authenticated Kaggle API client."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi   # imported here: importing kaggle requires credentials
        api = KaggleApi()
        api.authenticate()
        return api
    except (Exception, SystemExit) as error:                  # the kaggle package exits when there is no login
        raise ImageSourceUnavailable(
            f"Kaggle login missing or invalid ({error}). Run `kaggle auth login` (or set KAGGLE_API_TOKEN).") from None


def download(api, article_id: str, target: Path) -> None:
    """Download one article image to `target`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        api.competition_download_file(COMPETITION, kaggle_path(article_id), path=tmp, quiet=True)
        files = list(Path(tmp).iterdir())
        if not files:
            raise FileNotFoundError(kaggle_path(article_id))
        downloaded = files[0]
        if downloaded.suffix == ".zip":                       # Kaggle may deliver a single file zipped
            with zipfile.ZipFile(downloaded) as z:
                jpg = next(n for n in z.namelist() if n.endswith(".jpg"))
                target.write_bytes(z.read(jpg))
        else:
            shutil.move(downloaded, target)
