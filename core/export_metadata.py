"""Encode repaired pixels while preserving compatible image metadata."""

from dataclasses import dataclass, field
from pathlib import Path
from struct import pack
from zlib import compress, crc32

import cv2
import numpy as np
import tifffile
from PIL import Image

from .scan_io import display_pixels

_TIFF_TEXT = (269, 270, 271, 272, 285, 305, 306, 315, 33432)


@dataclass
class PhotoMetadata:
    icc: bytes = b""
    exif: bytes = b""
    text: dict = field(default_factory=dict)
    dpi: tuple | None = None
    depth: int = 8


def read_metadata(path):
    try:
        with Image.open(path) as source:
            info = source.info
            icc = info.get("icc_profile") or b""
            # FilmClear decodes to RGB. An input gray/CMYK profile cannot
            # describe those three-channel output pixels.
            if icc[16:20] != b"RGB ":
                icc = b""
            tags = getattr(source, "tag_v2", {})
            bits = tags.get(258, 16 if source.mode.startswith("I;16") else 8)
            if isinstance(bits, tuple):
                bits = max(bits)
            # Pillow exposes 16-bit RGB PNG as RGB; IHDR retains the real depth.
            if source.format == "PNG":
                with open(path, "rb") as stream:
                    header = stream.read(25)
                if len(header) == 25:
                    bits = header[24]
            exif = source.getexif()
            text = {tag: str(exif[tag]) for tag in _TIFF_TEXT if tag in exif}
            # JPEG, PNG and WebP can safely store the normal EXIF IFD. TIFF's
            # root IFD also describes the pixel layout, so copy descriptive tags only.
            if source.format == "TIFF":
                payload = Image.Exif()
                for tag in _TIFF_TEXT:
                    if tag in text:
                        payload[tag] = text[tag]
            else:
                payload = exif
                for tag in (256, 257, 274, 513, 514):
                    payload.pop(tag, None)
            dpi = info.get("dpi")
            if dpi and (len(dpi) != 2 or not all(0 < float(value) <= 100000 for value in dpi)):
                dpi = None
            return PhotoMetadata(icc, payload.tobytes() if payload else b"", text, dpi, int(bits))
    except (OSError, ValueError, TypeError, SyntaxError, OverflowError):
        return PhotoMetadata()


def _png_chunk(kind, data):
    return pack(">I", len(data)) + kind + data + pack(">I", crc32(kind + data))


def write_repaired(output, image, suffix, metadata):
    if suffix in (".jpg", ".jpeg", ".webp"):
        pixels = Image.fromarray(cv2.cvtColor(display_pixels(image), cv2.COLOR_BGR2RGB))
        options = {}
        if metadata.icc:
            options["icc_profile"] = metadata.icc
        if metadata.exif:
            options["exif"] = metadata.exif
        if suffix == ".webp":
            pixels.save(output, format="WEBP", lossless=True, **options)
        else:
            if metadata.dpi:
                options["dpi"] = tuple(round(value) for value in metadata.dpi)
            pixels.save(output, format="JPEG", quality=95, subsampling=0, **options)
        return
    if suffix in (".tif", ".tiff"):
        tags = [(tag, "s", 0, value, True) for tag, value in metadata.text.items()
                if tag not in (270, 305, 306) and value.isascii()]
        if metadata.icc:
            tags.append((34675, "B", len(metadata.icc), metadata.icc, True))
        stamp = metadata.text.get(306, "")
        software = metadata.text.get(305, "")
        # Use the real path: older tifffile versions require a named stream.
        tifffile.imwrite(output, np.ascontiguousarray(image[..., ::-1]), photometric="rgb",
                         compression="deflate", metadata=None, extratags=tags,
                         description=metadata.text.get(270, "") if metadata.text.get(270, "").isascii() else None,
                         datetime=stamp if len(stamp) == 19 and stamp.isascii() else False,
                         software=software if software and software.isascii() else "FilmClear",
                         resolution=metadata.dpi, resolutionunit="INCH" if metadata.dpi else None)
        return
    pixels = display_pixels(image) if suffix == ".bmp" else image
    ok, buffer = cv2.imencode(suffix, pixels)
    if not ok:
        raise ValueError("图片编码失败")
    data = memoryview(buffer)
    if suffix == ".png" and (metadata.icc or metadata.exif or metadata.dpi):
        output.write(data[:33])  # PNG signature and IHDR.
        if metadata.icc:
            output.write(_png_chunk(b"iCCP", b"FilmClear ICC\0\0" + compress(metadata.icc)))
        if metadata.exif:
            output.write(_png_chunk(b"eXIf", metadata.exif.removeprefix(b"Exif\0\0")))
        if metadata.dpi:
            x, y = (round(value / 0.0254) for value in metadata.dpi)
            output.write(_png_chunk(b"pHYs", pack(">IIB", x, y, 1)))
        data = data[33:]
    output.write(data)


def describe_export(path, extension, mode):
    metadata = read_metadata(path)
    codec = Path(path).suffix.lower() if extension == "source" or mode == "original" else extension
    depth = metadata.depth if mode == "original" else 8
    if mode != "original" and metadata.depth == 16 and codec in (".tif", ".tiff", ".png"):
        depth = 16
    profile = "保留 ICC" if metadata.icc and codec != ".bmp" else "无可嵌入的 RGB ICC"
    if mode == "original":
        profile = "原文件及附带信息"
    return f"{codec.lstrip('.').upper()} · {depth} 位 · {profile}"
