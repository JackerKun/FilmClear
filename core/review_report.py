"""Create a portable, interactive contact sheet from actual exports."""

from base64 import b64encode
from hashlib import sha256
from html import escape
from io import BytesIO
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image

from .batch_export import publish_file
from .export_metadata import read_metadata
from .scan_io import display_pixels, read_scan, signature

_ASSETS = Path(__file__).resolve().parents[1] / "assets"

_TOOLBAR = """
<div class="toolbar" hidden role="group" aria-label="对比工具">
  <button type="button" data-layout="side" aria-pressed="true">左右并排</button>
  <button type="button" data-layout="stack" aria-pressed="false">上下排列</button>
  <button type="button" data-layout="swipe" aria-pressed="false">滑动分割</button>
  <label class="split-control" hidden>分界 <input type="range" min="0" max="100" value="50"></label>
  <label class="mark-toggle"><input type="checkbox" checked>显示标记</label>
  <button type="button" data-action="out" aria-label="缩小">−</button>
  <output aria-label="相对适应窗口的倍率">1.0×</output>
  <button type="button" data-action="in" aria-label="放大">+</button>
  <button type="button" data-action="fit">适应</button>
  <button type="button" data-action="whole">返回全图</button>
  <button type="button" data-action="previous">上一处</button>
  <button type="button" data-action="next">下一处</button>
</div>
"""


def _head(script_hash):
    style = (_ASSETS / "review-report.css").read_text(encoding="utf-8")
    return f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'sha256-{script_hash}'">
<title>FilmClear 修片对照</title>
<style>{style}</style>
<h1>FilmClear 修片对照</h1>
<p>点击缺陷轮廓或缺陷列表查看局部。放大后可拖动，按住 Ctrl / ⌘ 滚动也可缩放。青色为已修复，琥珀色为待处理。</p>
"""


def _preview(pixels, icc, maximum):
    height, width = pixels.shape[:2]
    factor = min(1, maximum / max(height, width))
    if factor < 1:
        pixels = cv2.resize(pixels, (max(1, round(width * factor)), max(1, round(height * factor))),
                            interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(display_pixels(pixels), cv2.COLOR_BGR2RGB)
    stream = BytesIO()
    Image.fromarray(rgb).save(stream, format="JPEG", quality=88, icc_profile=icc or None)
    return "data:image/jpeg;base64," + b64encode(stream.getvalue()).decode("ascii")


def _outline(contours, box):
    segments = []
    for contour in contours:
        points = cv2.approxPolyDP(contour, 0.65, True).reshape(-1, 2)
        if len(points) >= 3:
            segments.append("M" + "L".join(f"{x} {y}" for x, y in points) + "Z")
    if segments:
        return " ".join(segments)
    x, y, w, h = box
    return f"M{x} {y}h{w}v{h}h{-w}Z"


def _patched_regions(patches, shape):
    if not patches:
        return []
    mask = np.zeros(shape, dtype=np.uint8)
    for _, positions, _ in patches:
        mask.ravel()[positions] = 255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    groups = {}
    for index, contour in enumerate(contours):
        parent = int(hierarchy[0, index, 3])
        if parent == -1:
            groups.setdefault(index, []).insert(0, contour)
        else:
            groups.setdefault(parent, []).append(contour)
    return [(cv2.boundingRect(group[0]), group) for group in groups.values()]


def _marks(entry, mode, shape):
    height, width = shape
    marks = []

    def add(box, contours, label, status):
        x, y, w, h = map(int, box)
        if min(w, h) <= 0 or x >= width or y >= height or x + w <= 0 or y + h <= 0:
            return
        margin = max(28, min(72, max(w, h) // 2))
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(width, x + w + margin), min(height, y + h + margin)
        marks.append({"label": label, "status": status, "box": [x, y, w, h],
                      "path": _outline(contours, (x, y, w, h)),
                      "crop": [x1, y1, x2 - x1, y2 - y1], "images": []})

    regions = sorted(_patched_regions(entry.patches, shape), key=lambda region: region[0][1::-1])
    for index, (box, contours) in enumerate(regions, 1):
        add(box, contours, f"已修复区域 {index:02d}", "applied")
    selected_ids = {item.id for item in entry.selected} if mode == "repair" else set()
    candidates = {item.id: item for item in (*entry.candidates, *entry.selected)}
    for item in candidates.values():
        applied = item.id in selected_ids
        label = f"{'已修复' if applied else '待处理'} #{item.id:03d} · {item.type_name}"
        contours = (item.contour, *item.holes) if item.contour is not None else ()
        add(item.bbox, contours, label, "applied" if applied else "pending")
    return marks


def _previews(path, *, marks=None, entry=None, mode=None, expected_shape=None):
    pixels = read_scan(path, read_ir=False).image
    shape = pixels.shape[:2]
    if expected_shape is not None and shape != expected_shape:
        raise ValueError("原片与导出结果尺寸不一致")
    if marks is None:
        marks = _marks(entry, mode, shape)
    icc = read_metadata(path).icc
    for mark in marks:
        x, y, width, height = mark["crop"]
        mark["images"].append(_preview(pixels[y:y + height, x:x + width], icc, 460))
    return _preview(pixels, icc, 1360), marks, shape


def _write_viewer(stream, before, after, marks):
    stream.write(_TOOLBAR)
    stream.write('<div class="viewer">\n')
    for side, image, label in (("before", before, "原片"), ("after", after, "导出结果")):
        stream.write(f'<div class="viewport {side}" tabindex="0" aria-label="{label}画面">'
                     f'<div class="stage"><img alt="{label}" src="{image}" loading="lazy" draggable="false">'
                     '<svg aria-hidden="true"></svg></div>'
                     f'<span class="badge">{label}</span></div>\n')
    stream.write('<div class="divider" aria-hidden="true"></div></div>\n')
    repaired = sum(mark["status"] == "applied" for mark in marks)
    if marks:
        stream.write(f'<details class="defects" open><summary>缺陷位置 · 已修复 {repaired} · 待处理 {len(marks) - repaired}'
                     '</summary><div class="defect-list"></div></details>\n')
    else:
        stream.write('<p>此照片没有可导出的缺陷标记。</p><div class="defect-list" hidden></div>\n')
    stream.write('<p class="scope" aria-live="polite">整张预览</p>\n')
    stream.write(f"<template>{escape(json.dumps(marks, ensure_ascii=False, separators=(',', ':')))}</template>\n")


def create_review_report(entries, saved, directory, mode, progress, canceled):
    by_path = {entry.path: entry for entry in entries}
    directory = Path(directory)
    script = (_ASSETS / "review-report.js").read_text(encoding="utf-8")
    script_hash = b64encode(sha256(script.encode()).digest()).decode("ascii")
    with NamedTemporaryFile(dir=directory, mode="w", encoding="utf-8",
                            prefix=".filmclear-review-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(_head(script_hash))
            for index, (source, _, target) in enumerate(saved):
                if canceled():
                    stream.write("<p>生成已停止。此页包含已完成的对照。</p>\n")
                    break
                progress(f"正在生成对照 · {source.name}", index, len(saved))
                entry = by_path[source]
                try:
                    if entry.stamp is not None and signature(source) != entry.stamp:
                        raise ValueError("原片已发生变化")
                    before, marks, shape = _previews(source, entry=entry, mode=mode)
                    if entry.stamp is not None and signature(source) != entry.stamp:
                        raise ValueError("原片在读取时发生变化")
                    after, _, _ = _previews(target, marks=marks, expected_shape=shape)
                    dimensions = f' data-width="{shape[1]}" data-height="{shape[0]}"'
                except (OSError, ValueError, cv2.error) as error:
                    stream.write(f"<article><h2>{escape(source.name)}</h2>"
                                 f"<p>无法生成这一组预览 · {escape(str(error))}</p></article>\n")
                    continue
                stream.write(f'<article{dimensions}><header class="photo-head"><h2>{escape(source.name)}</h2>'
                             f'<a href="{quote(target.name, safe="")}">{escape(target.name)}</a></header>\n')
                _write_viewer(stream, before, after, marks)
                stream.write("</article>\n")
            stream.write(f"<script>{script}</script></html>\n")
            stream.close()
            if not canceled():
                progress("已生成修片对照", len(saved), len(saved))
            return publish_file(temporary, directory, "FilmClear-修片对照", ".html")
        finally:
            temporary.unlink(missing_ok=True)
