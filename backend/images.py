"""Local image store that fetches a missing article image the first time it is requested, then keeps it."""
import logging
import threading
from pathlib import Path

from data_science.image_source import ImageSourceUnavailable, connect, download

log = logging.getLogger(__name__)


class OnDemandImages:
    def __init__(self, images_dir: Path, enabled: bool = True):
        self.images_dir = images_dir
        self.enabled = enabled
        self._api = None
        self._source_unavailable = False      # e.g. no Kaggle login: stop trying until the server restarts
        self._failed: set[str] = set()        # articles the source couldn't deliver
        self._lock = threading.Lock()         # one download at a time, so an image is never fetched twice

    def path(self, style_id: str, article_id: str) -> Path:
        return self.images_dir / style_id / f"{article_id}.jpg"

    def get(self, style_id: str, article_id: str) -> Path | None:
        """The local image file, fetched first if it is missing; None if it isn't available."""
        target = self.path(style_id, article_id)
        if target.exists():
            return target
        if not self.enabled or self._source_unavailable or article_id in self._failed:
            return None
        with self._lock:
            if target.exists():                       # fetched by another request while this one waited
                return target
            try:
                if self._api is None:
                    self._api = connect()
                download(self._api, article_id, target)
                log.info("fetched image %s for %s", article_id, style_id)
                return target
            except ImageSourceUnavailable as error:
                self._source_unavailable = True
                log.warning("on-demand images disabled: %s", error)
            except Exception as error:                # one unavailable image must not break the request
                self._failed.add(article_id)
                log.warning("image %s not available: %s", article_id, error)
        return None
