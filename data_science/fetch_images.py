"""Fetch the article images the app shows (listed in outputs/images_needed.csv by `python -m data_science.predict`).

Images are stored per style: outputs/images/<style_id>/<article_id>.jpg (style_id is stable across quarters, unlike rank).

Only images that are not yet in outputs/images/ are fetched, so re-running it each quarter downloads just the new ones.
The API also fetches any missing image the first time the app shows it; this script just does it upfront.

    python -m data_science.fetch_images                          # download from Kaggle (needs a Kaggle login)
    python -m data_science.fetch_images --from-dir <folder>      # copy from images already downloaded by hand

Kaggle login (once): `kaggle auth login`, or generate a token at kaggle.com → Settings → API and set it as
KAGGLE_API_TOKEN / save it to ~/.kaggle/access_token. You must have accepted the competition rules.
In production the images would come from the retailer's image store instead; see IMAGE_URL_TEMPLATE in the backend.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

from data_science.image_source import ImageSourceUnavailable, connect, download
from data_science.train import OUTPUTS

IMAGES_DIR = OUTPUTS / "images"

__all__ = ["fetch", "ImageSourceUnavailable"]


def _find_local(article_id: str, kaggle_path: str, source: Path) -> Path | None:
    """Look for the image in `source`, either in Kaggle's layout (images/076/…) or flat."""
    for candidate in (source / kaggle_path, source / kaggle_path.removeprefix("images/"), source / f"{article_id}.jpg"):
        if candidate.exists():
            return candidate
    return None


def fetch(from_dir: Path | None = None) -> None:
    """Fetch every image in outputs/images_needed.csv that is not yet in outputs/images/."""
    need = pd.read_csv(OUTPUTS / "images_needed.csv", dtype={"article_id": str})
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    target_of = lambda r: IMAGES_DIR / r.style_id / f"{r.article_id}.jpg"

    # images stored flat by an earlier version are moved into their style's folder instead of downloaded again
    for r in need.itertuples():
        flat = IMAGES_DIR / f"{r.article_id}.jpg"
        if flat.exists() and not target_of(r).exists():
            target_of(r).parent.mkdir(exist_ok=True)
            shutil.move(flat, target_of(r))

    missing = [r for r in need.itertuples() if not target_of(r).exists()]
    print(f"images: {len(need)} needed, {len(missing)} missing in {IMAGES_DIR}")
    if not missing:
        return

    api = connect() if from_dir is None else None
    failed = []
    for r in missing:
        target = target_of(r)
        target.parent.mkdir(exist_ok=True)
        try:
            if from_dir is not None:
                source = _find_local(r.article_id, r.kaggle_path, from_dir)
                if source is None:
                    raise FileNotFoundError(r.kaggle_path)
                shutil.copy(source, target)
            else:
                download(api, r.article_id, target)
        except Exception as error:                              # one missing image must not stop the rest
            failed.append(f"{r.article_id} ({type(error).__name__})")

    print(f"images: fetched {len(missing) - len(failed)}, failed {len(failed)}")
    if failed:
        print("not available (the app shows a placeholder for these):", ", ".join(failed[:20]),
              "…" if len(failed) > 20 else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-dir", type=Path, help="copy images from this folder instead of downloading")
    args = parser.parse_args()
    try:
        fetch(args.from_dir)
    except ImageSourceUnavailable as error:
        raise SystemExit(f"{error} Or use --from-dir with images downloaded by hand.")


if __name__ == "__main__":
    main()
