"""Render positive 16-bit sRGB scans as Ultra HDR JPEG or PQ HEIC."""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from io import BytesIO
from struct import pack, unpack_from

import numpy as np
from PIL import Image, ImageCms

ULTRA_HDR_JPEG = "ultra-hdr-jpeg"
PQ_HEIC = "pq-heic"
HDR_SUFFIXES = {ULTRA_HDR_JPEG: ".jpg", PQ_HEIC: ".heic"}

_SRGB_LUMA = np.float32([0.2126, 0.7152, 0.0722])
_TO_BT2020 = np.float32([
    [0.627404, 0.329282, 0.043314],
    [0.069097, 0.919540, 0.011362],
    [0.016391, 0.088013, 0.895595],
])
_TO_SRGB = np.linalg.inv(_TO_BT2020)
_SDR_WHITE_NITS = 203
_HIGHLIGHT_START = 0.45
_PQ_M1, _PQ_M2 = 2610 / 16384, 2523 / 32
_PQ_C1, _PQ_C2, _PQ_C3 = 3424 / 4096, 2413 / 128, 2392 / 128

_XMP = b"http://ns.adobe.com/xap/1.0/\0"
_ISO = b"urn:iso:std:iso:ts:21496:-1\0"


@dataclass
class GainMap:
    pixels: np.ndarray
    minimum: float = 0
    maximum: float = 1
    gamma: float = 1
    offset_sdr: float = 0
    offset_hdr: float = 0
    capacity_min: float = 0
    capacity_max: float = 1

    def apply(self, linear, band):
        codes = self.pixels[band].astype(np.float32) / 255
        stops = self.minimum + (self.maximum - self.minimum) * codes ** (1 / self.gamma)
        return np.maximum((linear + self.offset_sdr) * np.exp2(stops)[..., None] - self.offset_hdr, 0)


@lru_cache(maxsize=1)
def srgb_profile():
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


@lru_cache(maxsize=8)
def is_srgb_profile(icc):
    """Check the actual transform so profile names do not determine color handling."""
    if not icc or icc == srgb_profile():
        return True
    levels = (0, 8, 24, 48, 80, 112, 152, 200, 255)
    rgb = np.array([(r, g, b) for r in levels for g in levels for b in levels],
                   dtype=np.uint8).reshape(1, -1, 3)
    try:
        converted = ImageCms.profileToProfile(
            Image.fromarray(rgb), BytesIO(icc), BytesIO(srgb_profile()),
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode="RGB",
        )
    except (ImageCms.PyCMSError, OSError, ValueError):
        return False
    return bool(np.max(np.abs(np.asarray(converted).astype(np.int16) - rgb)) <= 1)


def _srgb_to_linear(pixels):
    return np.where(pixels <= 0.04045, pixels / 12.92, ((pixels + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(linear):
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


def _log_gain(luminance):
    # Smoothstep leaves shadows and midtones unchanged and reaches +1 EV at white.
    position = np.clip((luminance - _HIGHLIGHT_START) / (1 - _HIGHLIGHT_START), 0, 1)
    return position * position * (3 - 2 * position)


def _linear_rgb(bgr):
    return _srgb_to_linear(bgr[..., ::-1].astype(np.float32) / np.iinfo(bgr.dtype).max)


def _bands(height):
    # Keep temporary float RGB buffers small even for medium-format scans.
    for start in range(0, height, 128):
        yield slice(start, start + 128)


def _jpeg(pixels, **options):
    with BytesIO() as stream:
        Image.fromarray(pixels).save(stream, format="JPEG", **options)
        return stream.getvalue()


def _app(marker, data):
    return pack(">BBH", 0xFF, marker, len(data) + 2) + data


def _xmp(description):
    packet = ('<x:xmpmeta xmlns:x="adobe:ns:meta/">'
              '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
              '<rdf:Description rdf:about="" xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/"'
              f' hdrgm:Version="1.0" {description}</rdf:RDF></x:xmpmeta>')
    return _app(0xE1, _XMP + packet.encode("ascii"))


def _container_xmp(length):
    return _xmp(f'''
        xmlns:Container="http://ns.google.com/photos/1.0/container/"
        xmlns:Item="http://ns.google.com/photos/1.0/container/item/">
        <Container:Directory><rdf:Seq>
          <rdf:li rdf:parseType="Resource">
            <Container:Item Item:Semantic="Primary" Item:Mime="image/jpeg"/>
          </rdf:li>
          <rdf:li rdf:parseType="Resource">
            <Container:Item Item:Semantic="GainMap" Item:Mime="image/jpeg" Item:Length="{length}"/>
          </rdf:li>
        </rdf:Seq></Container:Directory></rdf:Description>''')


def _mp_index(primary_size, gain_size, gain_offset):
    # MP entries are 16 bytes each. Offsets use the MP endian field as origin.
    directory_size = 2 + 3 * 12 + 4
    return _app(0xE2, b"MPF\0MM\0*" + pack(">IH", 8, 3)
                + pack(">HHI4s", 0xB000, 7, 4, b"0100")
                + pack(">HHII", 0xB001, 4, 1, 2)
                + pack(">HHIII", 0xB002, 7, 32, 8 + directory_size, 0)
                + pack(">IIIHH", 0x030000, primary_size, 0, 0, 0)
                + pack(">IIIHH", 0, gain_size, gain_offset, 0, 0))


def _after_jfif(jpeg):
    return 4 + unpack_from(">H", jpeg, 4)[0] if jpeg[2:4] == b"\xff\xe0" else 2


def _write_ultra_hdr(output, image, exif, dpi, gain_map):
    height, width = image.shape[:2]
    base = image[..., ::-1]
    if image.dtype == np.uint16:
        base = ((base.astype(np.uint32) + 128) // 257).astype(np.uint8)
    if gain_map is None:
        gain = np.empty((height, width), np.uint8)
        for band in _bands(height):
            gain[band] = np.rint(_log_gain(_linear_rgb(image[band]) @ _SRGB_LUMA) * 255).astype(np.uint8)
        gain_map = GainMap(gain)

    options = {"icc_profile": srgb_profile()}
    if exif:
        options["exif"] = exif
    if dpi:
        options["dpi"] = tuple(round(value) for value in dpi)
    primary = _jpeg(base, quality=95, subsampling=0, **options)
    secondary = _jpeg(gain_map.pixels, quality=95)

    # ISO 21496-1 uses the same scalar metadata as XMP, in the base color space.
    version = pack(">HH", 0, 0)
    fractions = bytearray(b"\x40")
    for value in (gain_map.capacity_min, gain_map.capacity_max, gain_map.minimum, gain_map.maximum,
                  gain_map.gamma, gain_map.offset_sdr, gain_map.offset_hdr):
        fraction = Fraction(float(value)).limit_denominator(1000000)
        fractions.extend(pack(">iI", fraction.numerator, fraction.denominator))
    secondary_metadata = _xmp(
        f'hdrgm:GainMapMin="{gain_map.minimum:.9g}" hdrgm:GainMapMax="{gain_map.maximum:.9g}" '
        f'hdrgm:Gamma="{gain_map.gamma:.9g}" hdrgm:OffsetSDR="{gain_map.offset_sdr:.9g}" '
        f'hdrgm:OffsetHDR="{gain_map.offset_hdr:.9g}" hdrgm:HDRCapacityMin="{gain_map.capacity_min:.9g}" '
        f'hdrgm:HDRCapacityMax="{gain_map.capacity_max:.9g}" hdrgm:BaseRenditionIsHDR="False"/>'
    ) + _app(0xE2, _ISO + version + fractions)
    gain_size = len(secondary) + len(secondary_metadata)
    primary_metadata = _container_xmp(gain_size) + _app(0xE2, _ISO + version)
    primary_start = _after_jfif(primary)
    mp_start = primary_start + len(primary_metadata)
    primary_size = len(primary) + len(primary_metadata) + len(_mp_index(0, 0, 0))
    index = _mp_index(primary_size, gain_size, primary_size - (mp_start + 8))
    secondary_start = _after_jfif(secondary)
    for part in (primary[:primary_start], primary_metadata, index, memoryview(primary)[primary_start:],
                 secondary[:secondary_start], secondary_metadata, memoryview(secondary)[secondary_start:]):
        output.write(part)


def _encode_heic(output, pixels, mode, exif, **options):
    height, width = pixels.shape[:2]
    try:
        from pillow_heif import encode

        encode(mode, (width, height), memoryview(np.ascontiguousarray(pixels)).cast("B"), output,
               quality=92, chroma="420", exif=exif or None, **options)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"HEIC 编码失败：{error}") from error


def write_sdr_heic(output, image, exif, icc):
    rgb = image[..., ::-1]
    mode = "RGB;16" if image.dtype == np.uint16 else "RGB"
    _encode_heic(output, rgb, mode, exif, icc_profile=icc or srgb_profile(), bit_depth=10,
                 color_primaries=2, transfer_characteristics=2, matrix_coefficients=6,
                 full_range_flag=1, save_nclx_profile=True)


def _pq_encode(linear):
    power = np.clip(linear * (_SDR_WHITE_NITS / 10000), 0, 1) ** _PQ_M1
    encoded = ((_PQ_C1 + _PQ_C2 * power) / (1 + _PQ_C3 * power)) ** _PQ_M2
    return np.rint(encoded * 65535).astype(np.uint16)


def _write_pq_heic(output, image, exif, gain_map, native_pq, changed):
    pq = np.empty(image.shape, dtype="<u2") if native_pq is None else native_pq
    if native_pq is None or changed is not None:
        for band in _bands(image.shape[0]):
            if changed is not None and not np.any(changed[band]):
                continue
            linear = _linear_rgb(image[band])
            if gain_map is None:
                linear *= np.exp2(_log_gain(linear @ _SRGB_LUMA))[..., None]
            else:
                linear = gain_map.apply(linear, band)
            encoded = _pq_encode(linear @ _TO_BT2020.T)
            if changed is None:
                pq[band] = encoded
            else:
                np.copyto(pq[band], encoded, where=changed[band, :, None] != 0)
    _encode_heic(output, pq, "RGB;16", exif, bit_depth=10,
                 color_primaries=9, transfer_characteristics=16, matrix_coefficients=9,
                 full_range_flag=1, save_nclx_profile=True)


def write_hdr(output, image, format_id, metadata, exif, gain_map=None, native_pq=None, changed=None):
    if gain_map is None and native_pq is None and (image.dtype != np.uint16 or metadata.depth != 16):
        raise ValueError("HDR 导出需要 16 位 sRGB TIFF / PNG，或支持的 Ultra HDR JPEG / PQ HEIC。")
    if gain_map is None and native_pq is None and (metadata.non_srgb or not is_srgb_profile(metadata.icc)):
        raise ValueError("HDR 导出需要 sRGB 正片扫描。请先转换为 16 位 sRGB TIFF 或 PNG。")
    if format_id == ULTRA_HDR_JPEG:
        _write_ultra_hdr(output, image, exif, metadata.dpi, gain_map)
    else:
        _write_pq_heic(output, image, exif, gain_map, native_pq, changed)


def pq_to_sdr(encoded, *, keep_gain=False):
    """Use the inverse highlight curve for editing and retain a gain for conversion."""
    result = np.empty(encoded.shape, np.uint16)
    log_gain = np.empty(encoded.shape[:2], np.float32) if keep_gain else None
    sdr_luma = np.linspace(0, 1, 4097, dtype=np.float32)
    hdr_luma = sdr_luma * np.exp2(_log_gain(sdr_luma))
    for band in _bands(encoded.shape[0]):
        power = (encoded[band].astype(np.float32) / 65535) ** (1 / _PQ_M2)
        relative = (np.maximum(power - _PQ_C1, 0) / (_PQ_C2 - _PQ_C3 * power)) ** (1 / _PQ_M1)
        linear = (relative @ _TO_SRGB.T) * (10000 / _SDR_WHITE_NITS)
        luma = linear @ _SRGB_LUMA
        recovered = np.interp(luma, hdr_luma, sdr_luma).astype(np.float32)
        gain = np.maximum(np.exp2(_log_gain(recovered)), luma)
        # Fit saturated colors into the SDR base while retaining their brightness in the gain map.
        gain = np.maximum(gain, linear.max(axis=-1))
        result[band] = np.rint(_linear_to_srgb(linear[..., ::-1] / gain[..., None]) * 65535).astype(np.uint16)
        if keep_gain:
            log_gain[band] = np.log2(gain)
    if log_gain is None:
        return result, None
    maximum = max(1, float(np.ceil(log_gain.max() * 4) / 4))
    return result, GainMap(np.rint(log_gain * (255 / maximum)).astype(np.uint8),
                           maximum=maximum, capacity_max=maximum)
