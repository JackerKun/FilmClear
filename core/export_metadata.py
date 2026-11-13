"""Encode repaired pixels while preserving compatible image metadata."""

from dataclasses import dataclass, field
from pathlib import Path
from struct import pack, unpack
from zlib import compress, crc32

import cv2
import numpy as np
import tifffile
from PIL import Image

from .hdr_export import HDR_SUFFIXES, PQ_HEIC, ULTRA_HDR_JPEG, is_srgb_profile, srgb_profile, write_hdr, write_sdr_heic
from .hdr_input import HEIF_EXTENSIONS, is_pq_heif, is_ultra_hdr_jpeg, open_heif
from .scan_io import display_pixels

_TIFF_TEXT = (269, 270, 271, 272, 285, 305, 306, 315, 33432)


@dataclass
class PhotoMetadata:
    icc: bytes = b""
    exif: bytes = b""
    text: dict = field(default_factory=dict)
    dpi: tuple | None = None
    depth: int = 8
    non_srgb: bool = False
    hdr_format: str | None = None


def _heif_metadata(path):
    info = open_heif(path).info
    exif = Image.Exif()
    if info.get("exif"):
        exif.load(info["exif"])
    for tag in (256, 257, 274, 513, 514):
        exif.pop(tag, None)
    text = {tag: str(exif[tag]) for tag in _TIFF_TEXT if tag in exif}
    pq = is_pq_heif(info)
    icc = info.get("icc_profile") or b""
    if icc[16:20] != b"RGB ":
        icc = b""
    return PhotoMetadata(icc=srgb_profile() if pq else icc, exif=exif.tobytes() if exif else b"",
                         text=text, depth=info.get("bit_depth", 8), hdr_format=PQ_HEIC if pq else None)


def _png_hdr_metadata(path):
    # Pillow reads gAMA/cHRM but does not expose the newer cICP chunk.
    with open(path, "rb") as stream:
        header = stream.read(33)
        depth = header[24] if len(header) == 33 else 8
        while len(chunk := stream.read(8)) == 8:
            size, kind = unpack(">I4s", chunk)
            if kind in (b"IDAT", b"IEND"):
                break
            if kind == b"cICP":
                return depth, size != 4 or stream.read(4) != b"\x01\x0d\x00\x01"
            stream.seek(size + 4, 1)
    return depth, False


def read_metadata(path):
    try:
        if Path(path).suffix.lower() in HEIF_EXTENSIONS:
            return _heif_metadata(path)
        with Image.open(path) as source:
            info = source.info
            icc = info.get("icc_profile") or b""
            # FilmClear decodes to RGB. An input gray/CMYK profile cannot
            # describe those three-channel output pixels.
            non_srgb = bool(icc and icc[16:20] != b"RGB ")
            if non_srgb:
                icc = b""
            tags = getattr(source, "tag_v2", {})
            bits = tags.get(258, 16 if source.mode.startswith("I;16") else 8)
            if isinstance(bits, tuple):
                bits = max(bits)
            # Pillow exposes 16-bit RGB PNG as RGB; IHDR retains the real depth.
            if source.format == "PNG":
                bits, incompatible_cicp = _png_hdr_metadata(path)
                non_srgb |= incompatible_cicp
                if not icc and "srgb" not in info:
                    gamma = info.get("gamma")
                    chromaticity = info.get("chromaticity")
                    srgb_xy = (0.3127, 0.329, 0.64, 0.33, 0.30, 0.60, 0.15, 0.06)
                    non_srgb |= gamma is not None and abs(gamma - 0.45455) > 0.001
                    non_srgb |= bool(chromaticity and not np.allclose(chromaticity, srgb_xy, atol=0.001))
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
            # Pillow can return IFDRational values, which tifffile's writer
            # does not accept in its resolution argument.
            dpi = tuple(map(float, info.get("dpi") or ()))
            if len(dpi) != 2 or not all(0 < value <= 100000 for value in dpi):
                dpi = None
            hdr_format = ULTRA_HDR_JPEG if is_ultra_hdr_jpeg(source) else None
            return PhotoMetadata(icc, payload.tobytes() if payload else b"", text, dpi, int(bits),
                                 non_srgb, hdr_format)
    except (OSError, ValueError, TypeError, SyntaxError, OverflowError, IndexError):
        return PhotoMetadata()


def _png_chunk(kind, data):
    return pack(">I", len(data)) + kind + data + pack(">I", crc32(kind + data))


def write_repaired(output, image, suffix, metadata, *, geometry_changed=False, hdr_format=None,
                   gain_map=None, native_pq=None, changed=None):
    exif = metadata.exif
    if exif and geometry_changed and (hdr_format or suffix in HEIF_EXTENSIONS | {".jpg", ".jpeg", ".webp", ".png"}):
        tags = Image.Exif()
        tags.load(exif)
        if 34665 in tags:
            dimensions = tags.get_ifd(34665)
            for tag, value in ((40962, image.shape[1]), (40963, image.shape[0])):
                if tag in dimensions:
                    dimensions[tag] = value
        for tag, value in ((40962, image.shape[1]), (40963, image.shape[0])):
            if tag in tags:
                tags[tag] = value
        exif = tags.tobytes()
    if hdr_format in HDR_SUFFIXES:
        write_hdr(output, image, hdr_format, metadata, exif, gain_map, native_pq, changed)
        return
    if suffix in HEIF_EXTENSIONS:
        write_sdr_heic(output, image, exif, metadata.icc)
        return
    if suffix in (".jpg", ".jpeg", ".webp"):
        pixels = Image.fromarray(cv2.cvtColor(display_pixels(image), cv2.COLOR_BGR2RGB))
        options = {}
        if metadata.icc:
            options["icc_profile"] = metadata.icc
        if exif:
            options["exif"] = exif
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
    if suffix == ".png" and (metadata.icc or exif or metadata.dpi):
        output.write(data[:33])  # PNG signature and IHDR.
        if metadata.icc:
            output.write(_png_chunk(b"iCCP", b"FilmClear ICC\0\0" + compress(metadata.icc)))
        if exif:
            output.write(_png_chunk(b"eXIf", exif.removeprefix(b"Exif\0\0")))
        if metadata.dpi:
            x, y = (round(value / 0.0254) for value in metadata.dpi)
            output.write(_png_chunk(b"pHYs", pack(">IIB", x, y, 1)))
        data = data[33:]
    output.write(data)


def hdr_input_error(metadata):
    if metadata.hdr_format == ULTRA_HDR_JPEG and (metadata.non_srgb or not is_srgb_profile(metadata.icc)):
        return "当前支持带 sRGB 底图的 Ultra HDR JPEG。"
    if metadata.hdr_format:
        return ""
    if metadata.depth != 16:
        return "请选择 16 位 sRGB TIFF / PNG，或支持的 Ultra HDR JPEG / PQ HEIC。"
    if metadata.non_srgb or not is_srgb_profile(metadata.icc):
        return "请先将这张照片转换为 16 位 sRGB。"
    return ""


def describe_export(path, extension, mode, metadata=None):
    metadata = metadata if metadata is not None else read_metadata(path)
    hdr_format = metadata.hdr_format if extension == "source" else extension
    if mode != "original" and hdr_format in HDR_SUFFIXES:
        if error := hdr_input_error(metadata):
            return error
        codec = "HEIC · 10 位 PQ / BT.2020" if hdr_format == PQ_HEIC else "JPEG · SDR 底图 + HDR 增益图"
        if metadata.hdr_format:
            source = "PQ HEIC" if metadata.hdr_format == PQ_HEIC else "Ultra HDR JPEG"
            return f"{codec} · 使用原片的 {source} 高光数据"
        profile = "sRGB" if metadata.icc else "无 ICC，按 sRGB 处理"
        return f"{codec} · {profile}"
    codec = Path(path).suffix.lower() if extension == "source" or mode == "original" else extension
    depth = metadata.depth if mode == "original" else 8
    if mode != "original" and metadata.depth == 16 and codec in (".tif", ".tiff", ".png"):
        depth = 16
    if mode != "original" and codec in HEIF_EXTENSIONS:
        depth = 10 if metadata.depth > 8 else 8
    profile = "保留 ICC" if metadata.icc and codec != ".bmp" else "无可嵌入的 RGB ICC"
    if mode == "original":
        profile = "原文件及附带信息"
    return f"{codec.lstrip('.').upper()} · {depth} 位 · {profile}"
