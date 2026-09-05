"""Serving the pixels: full images and filmstrip thumbnails.

The only routes in the service that read a file the *user* put somewhere, so
this is where containment is checked. The rule is unchanged from the batch
API: a client names a project id and an image id, and the path comes out of the
database. Nothing here accepts a path.

Thumbnails are generated on demand and cached to disk beside the project.
Decoding a 4000x3000 JPEG per filmstrip cell is roughly 40ms; doing that for
sixty visible cells on every scroll is what makes an annotation tool feel slow
in a way no amount of front-end work can fix.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse

from ...annotate import images as images_mod
from ...annotate import projects as projects_mod
from ...annotate.store import Store, StoreError
from ...core import imaging
from ..deps import get_store, media_root, principal, require_key, resolve_media

router = APIRouter(prefix="/api", tags=["media"], dependencies=[Depends(require_key)])

#: Thumbnail long edge. One size, because two sizes is a cache with a policy.
THUMB_EDGE = 200

#: Browsers cache these hard: the bytes for a given image id never change --
#: an edit changes shapes, not pixels.
CACHE_CONTROL = "private, max-age=86400"

CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def _locate(store: Store, project_id: str, image_id: str, owner: str) -> tuple[Path, object]:
    """Resolve an image id to a real file, or fail with a status code."""
    try:
        project = projects_mod.get(store, project_id, owner)
        image = images_mod.get(store, project, image_id)
    except StoreError as exc:
        raise HTTPException(404, str(exc)) from None

    root = Path(project.root) if project.is_attached and project.root else media_root(project.id)
    path = resolve_media(root, image.rel_path)
    if not path.is_file():
        raise HTTPException(404, f"{image.rel_path} is no longer on disk")
    return path, image


@router.get("/projects/{project_id}/images/{image_id}/file")
def image_file(
    project_id: str,
    image_id: str,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> FileResponse:
    path, _ = _locate(store, project_id, image_id, owner)
    return FileResponse(
        path,
        media_type=CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"),
        headers={"Cache-Control": CACHE_CONTROL},
    )


@router.get("/projects/{project_id}/images/{image_id}/thumb")
def image_thumb(
    project_id: str,
    image_id: str,
    store: Store = Depends(get_store),
    owner: str = Depends(principal),
) -> Response:
    path, image = _locate(store, project_id, image_id, owner)

    # An image already smaller than a thumbnail is its own thumbnail. Decoding
    # and re-encoding it would cost a JPEG generation and, at these sizes,
    # routinely produce a *bigger* file than the original.
    width, height = int(image.width), int(image.height)
    if not (width and height):
        try:
            width, height = imaging.image_size(path)
        except Exception:
            width = height = 0
    if width and height and max(width, height) <= THUMB_EDGE:
        return FileResponse(
            path,
            media_type=CONTENT_TYPES.get(path.suffix.lower(), "image/jpeg"),
            headers={"Cache-Control": CACHE_CONTROL},
        )

    cache_dir = media_root(project_id) / ".thumbs"
    # Keyed on the source path and its mtime, so replacing an image on disk in
    # an attached project invalidates the thumbnail without a sweep.
    try:
        stamp = f"{path}:{path.stat().st_mtime_ns}"
    except OSError:
        raise HTTPException(404, "image unreadable") from None
    key = hashlib.sha256(stamp.encode("utf-8")).hexdigest()[:32]
    cached = cache_dir / f"{key}.jpg"

    if cached.is_file():
        return FileResponse(
            cached, media_type="image/jpeg", headers={"Cache-Control": CACHE_CONTROL}
        )

    try:
        pixels = imaging.load(path)
    except Exception:
        # A corrupt file is a normal thing to find in a dataset -- it is one of
        # the defects inspect.health reports. It must not be a 500.
        raise HTTPException(422, "that image could not be decoded") from None

    height, width = pixels.shape[:2]
    scale = min(THUMB_EDGE / max(1, width), THUMB_EDGE / max(1, height), 1.0)
    if scale < 1.0:
        transform = imaging.make_resize(
            width, height, max(1, round(width * scale)), max(1, round(height * scale)), "fit"
        )
        pixels = transform.image(pixels)

    data = imaging.encode(pixels, ".jpg", quality=78)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
    except OSError:
        # A cache that cannot be written is a slow tool, not a broken one.
        pass

    return Response(
        content=data, media_type="image/jpeg", headers={"Cache-Control": CACHE_CONTROL}
    )
