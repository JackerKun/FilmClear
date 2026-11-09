"""Render export entries from original pixels, one image at a time."""

from dataclasses import dataclass
from pathlib import Path
import os
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

import cv2

from .dust_remover import DustRemover
from .export_metadata import read_metadata, write_repaired
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


def replay_patches(original, patches):
    if not patches:
        return original
    image = original.copy()
    flat = image.reshape(-1, image.shape[2])
    for _, positions, values in patches:
        flat[positions] = values
    return image


def _write_entry(entry, directory, extension, mode, repair_params):
    source = entry.path
    if mode != "original" and entry.stamp is not None and signature(source) != entry.stamp:
        raise ValueError("原图已发生变化，需要重新扫描后再应用这些修复。")
    suffix = source.suffix if mode == "original" or extension == SOURCE_FORMAT else extension
    base = f"{source.stem}_{'original' if mode == 'original' else 'clean'}"
    with NamedTemporaryFile(dir=directory, prefix=".filmclear-", delete=False) as output:
        temporary = Path(output.name)
        try:
            if mode == "original":
                with source.open("rb") as original:
                    copyfileobj(original, output)
            else:
                scan = read_scan(source, read_ir=False)
                if entry.stamp is not None and signature(source) != entry.stamp:
                    raise ValueError("读取期间原图发生变化，请重新扫描。")
                image = replay_patches(scan.image, entry.patches)
                if mode == "repair" and entry.selected:
                    image = DustRemover(**repair_params).remove(image, entry.selected, selected_only=False)
                write_repaired(output, image, suffix.lower(), read_metadata(source))
            output.close()
            return publish_file(temporary, directory, base, suffix)
        finally:
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
