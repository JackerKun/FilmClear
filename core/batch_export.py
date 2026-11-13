"""Render export entries from original pixels, one image at a time."""

from dataclasses import dataclass
from pathlib import Path
import os
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from .dust_remover import DustRemover
from .export_metadata import hdr_input_error, read_metadata, write_repaired
from .frame_crop import apply_crop
from .hdr_export import HDR_SUFFIXES, ULTRA_HDR_JPEG, pq_to_sdr
from .hdr_input import read_gain_map
from .scan_io import read_scan, signature

SOURCE_FORMAT = "source"

@dataclass
class ExportEntry:
    path: Path
    stamp: tuple | None
    patches: tuple
    selected: tuple
    revision: int = 0
    candidates: tuple = ()
    crop: dict | None = None


def replay_patches(original, patches):
    if not patches:
        return original
    image = original.copy()
    flat = image.reshape(-1, image.shape[2])
    for _, positions, values in patches:
        flat[positions] = values
    return image


def _render(entry, metadata, hdr_format, mode, repair_params):
    source = entry.path
    scan = read_scan(source, read_ir=False, keep_pq=bool(hdr_format))
    if entry.stamp is not None and signature(source) != entry.stamp:
        raise ValueError("读取期间原图发生变化，请重新扫描。")
    image = replay_patches(scan.image, entry.patches)
    remover = DustRemover(**repair_params)
    if mode == "repair" and entry.selected:
        image = remover.remove(image, entry.selected, selected_only=False)
    changed = None
    gain_map = None
    pq = scan.native_pq
    if hdr_format and metadata.hdr_format == ULTRA_HDR_JPEG:
        gain_map = read_gain_map(source, scan.image.shape)
    if (gain_map is not None or pq is not None) and image is not scan.image:
        changed = np.any(image != scan.image, axis=2).astype(np.uint8) * 255
        if not np.any(changed):
            changed = None
    if pq is not None and (hdr_format == ULTRA_HDR_JPEG or changed is not None):
        _, gain_map = pq_to_sdr(pq, keep_gain=True)
        if hdr_format == ULTRA_HDR_JPEG:
            pq = None
    if gain_map is not None:
        if changed is not None:
            gain_map.pixels = remover.apply_mask(gain_map.pixels, changed)
        gain_map.pixels = apply_crop(gain_map.pixels, entry.crop)
    if pq is not None:
        pq = apply_crop(pq, entry.crop)
        if changed is not None:
            changed = apply_crop(changed, entry.crop)
    return apply_crop(image, entry.crop), gain_map, pq, changed


def _write_entry(entry, directory, extension, mode, repair_params):
    source = entry.path
    if mode != "original" and entry.stamp is not None and signature(source) != entry.stamp:
        raise ValueError("原图已发生变化，需要重新扫描后再应用这些修复。")
    metadata = read_metadata(source) if mode != "original" else None
    hdr_format = None
    if metadata is not None:
        if extension == SOURCE_FORMAT:
            hdr_format = metadata.hdr_format
        elif extension in HDR_SUFFIXES:
            hdr_format = extension
    suffix = source.suffix if mode == "original" or extension == SOURCE_FORMAT else HDR_SUFFIXES.get(extension, extension)
    label = "original" if mode == "original" else "hdr" if hdr_format else "clean"
    base = f"{source.stem}_{label}"
    with NamedTemporaryFile(dir=directory, prefix=".filmclear-", delete=False) as output:
        temporary = Path(output.name)
        try:
            untouched_hdr = (metadata is not None and hdr_format and hdr_format == metadata.hdr_format
                             and not entry.patches and entry.crop is None
                             and (mode != "repair" or not entry.selected))
            if mode == "original" or untouched_hdr:
                with source.open("rb") as original:
                    copyfileobj(original, output)
                if mode != "original" and entry.stamp is not None and signature(source) != entry.stamp:
                    raise ValueError("读取期间原图发生变化，请重新扫描。")
            else:
                if hdr_format and (error := hdr_input_error(metadata)):
                    raise ValueError(error)
                image, gain_map, pq, changed = _render(entry, metadata, hdr_format, mode, repair_params)
                write_repaired(output, image, suffix.lower(), metadata, geometry_changed=entry.crop is not None,
                               hdr_format=hdr_format, gain_map=gain_map, native_pq=pq, changed=changed)
            output.close()
            return publish_file(temporary, directory, base, suffix)
        finally:
            output.close()
            temporary.unlink(missing_ok=True)


def publish_file(temporary, directory, base, suffix):
    # Publish a complete file without overwriting an existing export,
    # including another export made while this job was running.
    index = 1
    while True:
        extra = "" if index == 1 else f"_{index}"
        target = directory / f"{base}{extra}{suffix}"
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            index += 1
            continue
        os.close(fd)
        try:
            temporary.replace(target)
        except OSError:
            target.unlink(missing_ok=True)
            raise
        return target


def export_batch(entries, directory, extension, mode, repair_params, progress, canceled):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    saved, errors = [], []
    for index, entry in enumerate(entries):
        if canceled():
            break
        progress(f"正在导出 {entry.path.name}", index, len(entries))
        try:
            target = _write_entry(entry, directory, extension, mode, repair_params)
            saved.append((entry.path, entry.revision, target))
        except (OSError, ValueError, cv2.error) as error:
            errors.append((entry.path, str(error)))
        progress(f"已处理 {index + 1} / {len(entries)} 张", index + 1, len(entries))
    return saved, errors
