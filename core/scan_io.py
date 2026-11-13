"""Read full-resolution scans and identify scanner infrared samples."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile
from .hdr_input import HEIF_EXTENSIONS, read_heif_pixels

TIFF_EXTENSIONS = {".tif", ".tiff"}
IR_SUFFIXES = ("_ir", "_infrared")


@dataclass
class Scan:
    image: np.ndarray
    infrared: np.ndarray | None = None
    ir_source: str = ""
    ir_notice: str = ""
    native_pq: np.ndarray | None = None


def signature(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


def ir_sidecar(path):
    path = Path(path)
    names = {path.stem.casefold() + suffix for suffix in IR_SUFFIXES}
    return next((p for p in sorted(path.parent.iterdir())
                 if p.suffix.lower() in TIFF_EXTENSIONS
                 and p.stem.casefold() in names and p.is_file()), None)


def is_ir_sidecar(path, stems):
    stem = path.stem.casefold()
    return (path.suffix.lower() in TIFF_EXTENSIONS and any(
        stem.endswith(suffix) and stem[:-len(suffix)] in stems for suffix in IR_SUFFIXES
    ))


def _decode(path, flags):
    pixels = cv2.imdecode(np.fromfile(path, np.uint8), flags)
    if pixels is None:
        raise ValueError(f"无法解码 {Path(path).name}")
    return pixels


def _oriented(pixels, page):
    tag = page.tags.get("Orientation")
    orientation = int(tag.value) if tag else 1
    if orientation in (5, 6, 7, 8):
        pixels = pixels.swapaxes(0, 1)
    if orientation in (2, 3, 6, 7):
        pixels = pixels[:, ::-1]
    if orientation in (3, 4, 7, 8):
        pixels = pixels[::-1]
    return np.ascontiguousarray(pixels)


def _scanner_tags(page):
    return " ".join(str(tag.value) for name in ("Software", "ImageDescription", "PageName")
                    if (tag := page.tags.get(name)) is not None).casefold()


def _embedded_ir(path, *, force_fourth=False):
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        text = _scanner_tags(page)
        samples = page.samplesperpixel
        scanner = any(label in text for label in ("vuescan", "silverfast", "rgbi", "hdri"))
        unspecified = not page.extrasamples or all(int(sample) == 0 for sample in page.extrasamples)
        known_fourth = scanner or force_fourth or (unspecified and page.bitspersample == 16)
        if page.photometric == 2 and samples == 4 and known_fourth:
            try:
                data = page.asarray()
            except ValueError:
                # OpenCV's TIFF decoder covers LZW on installations where
                # tifffile has no optional codec package.
                native = _decode(path, cv2.IMREAD_UNCHANGED)
                orientation = page.tags.get("Orientation")
                if (native.ndim != 3 or native.shape[2] != 4
                        or native.dtype != page.dtype or page.bitspersample != 16
                        or (orientation and int(orientation.value) != 1)):
                    raise ValueError("该压缩 TIFF 的第四通道无法无损读取")
                return (np.ascontiguousarray(native[..., :3]),
                        np.ascontiguousarray(native[..., 3]), "TIFF RGBI")
            if page.axes == "SYX":
                data = np.moveaxis(data, 0, -1)
            data = _oriented(data, page)
            return (np.ascontiguousarray(data[..., 2::-1]),
                    np.ascontiguousarray(data[..., 3]), "TIFF RGBI")
        if force_fourth:
            raise ValueError("当前 TIFF 没有可用的第四通道。")
        for extra in list(tif.pages)[1:]:
            label = _scanner_tags(extra)
            if extra.samplesperpixel == 1 and (
                "infrared" in label or "isrd" in label or "红外" in label
            ):
                return None, _oriented(extra.asarray(), extra), "TIFF 红外页"
    return None, None, ""


def read_ir_plane(path, shape, *, embedded=False):
    if embedded:
        _, pixels, _ = _embedded_ir(path, force_fourth=True)
    else:
        pixels = _decode(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if pixels.shape != shape:
        raise ValueError("红外图的尺寸或方向与照片不一致，需要对齐后再导入。")
    return pixels


def read_scan(path, *, read_ir=True, keep_pq=False):
    path = Path(path)
    image = infrared = None
    source = notice = ""
    native_pq = None
    if path.suffix.lower() in HEIF_EXTENSIONS:
        image, native_pq = read_heif_pixels(path, keep_pq=keep_pq)
    elif path.suffix.lower() in TIFF_EXTENSIONS:
        try:
            image, infrared, source = _embedded_ir(path)
        except (tifffile.TiffFileError, ValueError, OSError, NotImplementedError, cv2.error) as error:
            notice = f"内嵌红外读取失败，已使用 RGB（{error}）"
    if image is None:
        image = _decode(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if image.dtype not in (np.uint8, np.uint16):
        image = _decode(path, cv2.IMREAD_COLOR)
    if not read_ir:
        return Scan(image, native_pq=native_pq)
    if read_ir and infrared is None:
        sidecar = ir_sidecar(path)
        if sidecar:
            try:
                infrared = read_ir_plane(sidecar, image.shape[:2])
                source, notice = sidecar.name, ""
            except (OSError, ValueError, cv2.error) as error:
                notice = str(error)
    if infrared is not None and infrared.shape != image.shape[:2]:
        return Scan(image, ir_notice="红外层和照片尺寸不一致，已使用 RGB", native_pq=native_pq)
    return Scan(image, infrared, source, notice, native_pq)


def display_pixels(image):
    """Convert a high bit-depth scan only for display or JPEG encoding."""
    if image.dtype == np.uint16:
        return (image >> 8).astype(np.uint8)
    return image
