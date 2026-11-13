"""Read HEIF pixels and the gain map of an SDR-base Ultra HDR JPEG."""

from io import BytesIO
from pathlib import Path
from struct import error as StructError, unpack_from
from xml.etree import ElementTree

import cv2
import numpy as np
from PIL import Image

from .hdr_export import GainMap, _ISO, _XMP, pq_to_sdr

HEIF_EXTENSIONS = {".heic", ".heif"}

_GAIN_NS = "{http://ns.adobe.com/hdr-gain-map/1.0/}"
_ITEM_NS = "{http://ns.google.com/photos/1.0/container/item/}"
_GAIN_FIELDS = ("GainMapMin", "GainMapMax", "Gamma", "OffsetSDR", "OffsetHDR",
                "HDRCapacityMin", "HDRCapacityMax")
_TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90,
}


def open_heif(path):
    try:
        from pillow_heif import open_heif as decode

        # The container's rotation/mirror properties are applied by libheif.
        return decode(path, convert_hdr_to_8bit=False)
    except (ImportError, OSError, RuntimeError, ValueError, EOFError, SyntaxError) as error:
        raise ValueError(f"无法读取 HEIC / HEIF · {Path(path).name} · {error}") from error


def is_pq_heif(info):
    profile = info.get("nclx_profile", {})
    return info.get("bit_depth", 8) >= 10 and (
        profile.get("color_primaries"), profile.get("transfer_characteristics")) == (9, 16)


def read_heif_pixels(path, *, keep_pq=False):
    image = open_heif(path)
    try:
        rgb = np.asarray(image)
    except (OSError, RuntimeError, ValueError, EOFError, SyntaxError, IndexError) as error:
        raise ValueError(f"无法解码 HEIC / HEIF · {Path(path).name} · {error}") from error
    if rgb.ndim == 2 or rgb.shape[2] < 3:
        gray = rgb if rgb.ndim == 2 else rgb[..., 0]
        rgb = np.repeat(gray[..., None], 3, axis=2)
    else:
        rgb = rgb[..., :3]
    if is_pq_heif(image.info):
        base, _ = pq_to_sdr(rgb)
        return base, rgb.copy() if keep_pq else None
    profile = image.info.get("nclx_profile", {})
    if profile.get("transfer_characteristics") in (16, 18):
        raise ValueError("当前 HEIC HDR 色彩格式暂不支持。可先转换为 BT.2020 PQ HEIC。")
    return np.ascontiguousarray(rgb[..., ::-1]), None


def _packet(image, marker, prefix):
    return next((data[len(prefix):] for name, data in image.applist
                 if name == marker and data.startswith(prefix)), None)


def is_ultra_hdr_jpeg(image):
    if image.format not in ("JPEG", "MPO"):
        return False
    xmp = _packet(image, "APP1", _XMP) or b""
    return _packet(image, "APP2", _ISO) is not None or (
        b"http://ns.adobe.com/hdr-gain-map/1.0/" in xmp and b"GainMap" in xmp
    )


def _iso_parameters(data):
    version, _, flags = unpack_from(">HHB", data)
    if version != 0 or flags & 0x84 or not flags & 0x40:
        raise ValueError("仅支持使用底图色彩空间的单通道 Ultra HDR 增益图。")
    common = bool(flags & 8)
    denominator = unpack_from(">I", data, 5)[0] if common else 1
    cursor = 9 if common else 5
    values = []
    for index in range(7):
        numerator = unpack_from(">I" if index in (0, 1, 4) else ">i", data, cursor)[0]
        cursor += 4
        if not common:
            denominator = unpack_from(">I", data, cursor)[0]
            cursor += 4
        values.append(numerator / denominator)
    return (*values[2:], *values[:2])


def _xmp_parameters(data):
    fields = {}
    for node in ElementTree.fromstring(data).iter():
        fields.update(node.attrib)
        if node.tag.startswith(_GAIN_NS) and node.text:
            fields[node.tag] = node.text
    if fields.get(_GAIN_NS + "BaseRenditionIsHDR", "False").lower() != "false":
        raise ValueError("仅支持带 SDR 底图的 Ultra HDR JPEG。")
    defaults = (0, None, 1, 1 / 64, 1 / 64, 0, None)
    return tuple(float(fields.get(_GAIN_NS + key, default)) for key, default in zip(_GAIN_FIELDS, defaults))


def read_gain_map(path, shape):
    try:
        with open(path, "rb") as stream, Image.open(stream) as base:
            if not is_ultra_hdr_jpeg(base):
                return None
            orientation = base.getexif().get(274)
            mp = base._getmp()
            entries = mp.get(45058, []) if mp else []
            if len(entries) != 2:
                raise ValueError("Ultra HDR JPEG 缺少可用的增益图索引。")
            entry = entries[1]
            start = base.info["mpoffset"] + entry["DataOffset"]
            length = entry["Size"]
            stream.seek(0, 2)
            if start < 2 or length < 4 or start + length > stream.tell():
                raise ValueError("Ultra HDR 增益图索引无效。")
            stream.seek(start)
            with Image.open(BytesIO(stream.read(length)), formats=("JPEG",)) as auxiliary:
                iso = _packet(auxiliary, "APP2", _ISO)
                if iso is not None:
                    values = _iso_parameters(iso)
                else:
                    values = _xmp_parameters(_packet(auxiliary, "APP1", _XMP) or b"")
                if not np.all(np.isfinite(values)) or max(map(abs, values)) > 64 or values[2] <= 0:
                    raise ValueError("Ultra HDR 增益图参数无效。")
                if auxiliary.mode != "L":
                    raise ValueError("当前仅支持单通道 Ultra HDR 增益图。")
                if orientation in _TRANSPOSE:
                    auxiliary = auxiliary.transpose(_TRANSPOSE[orientation])
                pixels = np.asarray(auxiliary)
        if pixels.shape != shape[:2]:
            pixels = cv2.resize(pixels, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        return GainMap(pixels, *values)
    except (OSError, TypeError, KeyError, IndexError, StructError, SyntaxError, ZeroDivisionError) as error:
        raise ValueError(f"无法读取 Ultra HDR 增益图 · {error}") from error
