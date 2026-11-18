"""Local learning and review of RGB line candidates."""

from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from .dust_item import DustItem


LINE_TYPES = ("fiber", "scratch")
MODEL_NAME = "fiber-review.json"
_FORMAT = "filmclear-fiber-review-v1"
_DISCOVERY_FORMAT = "filmclear-fiber-review-v2"
_GROUP_FORMAT = "filmclear-fiber-review-group-v1"
_FILE_LIMIT = 2 * 1024 * 1024
_FEATURE_COUNT = 14
_DEFER_THRESHOLD = 0.20
# Confidence and the detector's fiber/scratch name can change between a local
# trace and a full scan. Match the geometry and original-pixel contrast.
_MATCH_AXES = (0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13)
_MATCH_TOLERANCE = np.array((0.44, 0.48, 0.36, 0.28, 0.25, 0.22,
                             0.37, 0.28, 0.32, 0.22, 0.05, 0.24))
_MAX_TEMPLATES = 32
_BUNDLED_MODEL = Path(__file__).resolve().parents[1] / "models" / MODEL_NAME


@dataclass(frozen=True)
class FiberExample:
    photo: str
    roll: str
    label: int
    checked: bool
    features: tuple[float, ...]
    traced: bool = False
    outline: tuple = ()


@dataclass(frozen=True)
class FiberModel:
    center: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    metrics: dict
    templates: tuple[tuple[float, ...], ...] = ()
    outlines: tuple = ()

    def _scores(self, features):
        normalized = np.clip((np.asarray(features) - self.center) / self.scale, -8, 8)
        logits = (normalized * self.weights[:-1]).sum(axis=-1) + self.weights[-1]
        return 1 / (1 + np.exp(-np.clip(logits, -60, 60)))

    def score(self, features):
        return float(self._scores(features))

    def defers(self, features):
        return self.score(features) <= _DEFER_THRESHOLD

    def recognizes(self, features):
        return bool(self.templates and _matches(self.templates, features).any()
                    and not self.defers(features))

    def encode(self):
        return json.dumps(self._data(), indent=2, sort_keys=True, allow_nan=False)

    def _data(self):
        data = dict(format=_FORMAT, center=self.center, scale=self.scale,
                    weights=self.weights, metrics=self.metrics)
        if self.templates:
            data.update(format=_DISCOVERY_FORMAT, templates=self.templates)
        if self.outlines:
            data["outlines"] = self.outlines
        return data

    @classmethod
    def decode(cls, content):
        if len(content) > _FILE_LIMIT:
            raise ValueError("模型文件过大")
        try:
            data = json.loads(content)
            if data["format"] == _GROUP_FORMAT:
                members = data["members"]
                if not isinstance(members, list) or len(members) < 2:
                    raise ValueError
                return combine_fiber_models(cls._from_data(member) for member in members)
            return cls._from_data(data)
        except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as error:
            raise ValueError("此文件不是兼容的 FilmClear 毛絮模型") from error

    @classmethod
    def _from_data(cls, data):
        if data["format"] not in (_FORMAT, _DISCOVERY_FORMAT):
            raise ValueError
        values = []
        for key, size in (("center", _FEATURE_COUNT), ("scale", _FEATURE_COUNT),
                          ("weights", _FEATURE_COUNT + 1)):
            if not isinstance(data[key], list) or len(data[key]) != size:
                raise ValueError
            array = tuple(float(v) for v in data[key])
            if not np.isfinite(array).all() or np.max(np.abs(array)) > 1e6:
                raise ValueError
            values.append(array)
        if min(values[1]) < 1e-6:
            raise ValueError
        metrics = data["metrics"]
        unit = metrics["validation"]
        if unit not in ("roll", "photo", "none"):
            raise ValueError
        summary = {"validation": unit}
        for key in ("samples", "defects", "scene", "evaluated", "scene_deferred", "defects_deferred",
                    "baseline_scene", "baseline_defects"):
            number = metrics[key]
            if type(number) is not int or number < 0:
                raise ValueError
            summary[key] = number
        templates = ()
        if data["format"] == _DISCOVERY_FORMAT:
            rows = data["templates"]
            if not isinstance(rows, list) or not 1 <= len(rows) <= _MAX_TEMPLATES:
                raise ValueError
            if any(not isinstance(row, list) or len(row) != _FEATURE_COUNT for row in rows):
                raise ValueError
            templates = tuple(tuple(float(value) for value in row) for row in rows)
            if not np.isfinite(templates).all() or np.max(np.abs(templates)) > 1e6:
                raise ValueError
        outlines = data.get("outlines", [])
        if not isinstance(outlines, list):
            raise ValueError
        if outlines:
            if len(outlines) != len(templates):
                raise ValueError
            for polygon in outlines:
                if not isinstance(polygon, list) or (polygon and not 3 <= len(polygon) <= 28):
                    raise ValueError
                if any(not isinstance(point, list) or len(point) != 2 or
                       any(type(value) is not int or not 0 <= value <= 100 for value in point)
                       for point in polygon):
                    raise ValueError
        return cls(*values, summary, templates, tuple(tuple(map(tuple, path)) for path in outlines))


@dataclass(frozen=True)
class FiberModelGroup:
    members: tuple[FiberModel, ...]

    @cached_property
    def templates(self):
        return tuple(dict.fromkeys(template for model in self.members for template in model.templates))

    def score(self, features):
        return max(model.score(features) for model in self.members)

    def recognizes(self, features):
        return any(model.recognizes(features) for model in self.members)

    def defers(self, features):
        classifiers = [model for model in self.members if model.metrics["scene"]]
        return (bool(classifiers) and all(model.defers(features) for model in classifiers)
                and not self.recognizes(features))

    def encode(self):
        result = json.dumps(dict(format=_GROUP_FORMAT, members=[model._data() for model in self.members]),
                            separators=(",", ":"), sort_keys=True, allow_nan=False)
        if len(result) > _FILE_LIMIT:
            raise ValueError("合并模型文件过大")
        return result


def combine_fiber_models(models):
    members = {}
    for model in models:
        for member in model.members if isinstance(model, FiberModelGroup) else (model,):
            members.setdefault(member.encode(), member)
    if not members:
        return None
    if len(members) == 1:
        return next(iter(members.values()))
    return FiberModelGroup(tuple(members[key] for key in sorted(members)))


def read_fiber_model(path):
    with Path(path).open("rb") as stream:
        return FiberModel.decode(stream.read(_FILE_LIMIT + 1))


def write_fiber_model(path, model):
    path = Path(path)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(model.encode() + "\n")
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def photo_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_key(item):
    digest = hashlib.blake2b(digest_size=16)
    for value in (item.contour, *item.holes):
        points = np.asarray(value, dtype="<i4").reshape(-1, 2)
        digest.update(len(points).to_bytes(8, "little"))
        digest.update(points.tobytes())
    return digest.hexdigest()


def _preview_outline(contour):
    points = np.asarray(contour, dtype=np.int32).reshape(-1, 2)
    span = max(1, int(np.ptp(points, axis=0).max()))
    scaled = np.rint((points - points.min(axis=0)) * (100 / span)).astype(np.int32)
    outline = cv2.approxPolyDP(scaled, 1.5, True).reshape(-1, 2)
    if len(outline) < 3:
        return ()
    if len(outline) > 28:
        outline = outline[np.linspace(0, len(outline), 28, endpoint=False, dtype=int)]
    return tuple(tuple(map(int, point)) for point in outline)


def fiber_features(image, item):
    """Describe trace geometry and contrast relative to its immediate surroundings."""
    x, y, w, h = item.bbox
    points = item.contour.reshape(-1, 2)
    a, b = cv2.minAreaRect(item.contour)[1]
    length = max(a, b, 1)
    thickness = max(1, item.area / length)
    perimeter = cv2.arcLength(item.contour, True)
    hull_area = cv2.contourArea(cv2.convexHull(item.contour))
    margin = max(16, min(120, round(5 * thickness)))
    left, top = max(0, x - margin), max(0, y - margin)
    crop = image[top:y + h + margin, left:x + w + margin]
    scale = max(1, np.sqrt(crop.shape[0] * crop.shape[1] / 360_000))
    if scale > 1:
        size = (max(1, round(crop.shape[1] / scale)), max(1, round(crop.shape[0] / scale)))
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = gray.astype(np.float32) * (1 / 257 if gray.dtype == np.uint16 else 1)
    mask = np.zeros(gray.shape, np.uint8)
    contours = [np.rint((contour - (left, top)) / scale).astype(np.int32)
                for contour in (points, *item.holes)]
    cv2.drawContours(mask, contours, -1, 1, cv2.FILLED)
    radius = max(2, min(20, round(2 * thickness / scale)))
    ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)) > mask
    foreground, surroundings = gray[mask != 0], gray[ring]
    if not foreground.size:
        foreground = np.array([float(gray.mean())])
    if not surroundings.size:
        surroundings = foreground
    low, background, high = np.percentile(surroundings, (10, 50, 90))
    median = float(np.median(foreground))
    local_detail = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.6))[ring]
    scale_factor = 3000 / min(image.shape[:2])
    return tuple(float(value) for value in (
        np.log1p(length * scale_factor), np.log1p(thickness * scale_factor),
        np.log1p((length + 1) / (min(a, b) + 1)),
        np.log1p(perimeter / length), np.log1p(item.area / max(1, hull_area)),
        item.area / (w * h), min(w, h) / max(w, h),
        item.confidence, float(item.dust_type == "scratch"),
        (median - background) / 255, (high - low) / 255,
        float(np.std(foreground)) / 255,
        (float(np.median(local_detail)) if local_detail.size else 0) / 255,
        (float(np.mean(local_detail >= 8)) if local_detail.size else 0),
    ))


def apply_fiber_model(image, items, model):
    for item in items:
        if (item.dust_type in LINE_TYPES and item.contour is not None and not item.requires_review
                and model.defers(fiber_features(image, item))):
            item.scene_line = True
            item.selected = False


def _matches(templates, features):
    differences = np.abs(np.asarray(templates)[..., _MATCH_AXES] -
                         np.asarray(features)[..., _MATCH_AXES]) / _MATCH_TOLERANCE
    return (differences.max(axis=-1) <= 1.0) & (np.mean(differences ** 2, axis=-1) <= 0.35)


def _fit(features, labels):
    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 0.015)
    design = np.column_stack((np.clip((features - center) / scale, -8, 8), np.ones(len(labels))))
    weights = np.zeros(design.shape[1], dtype=float)
    penalty = np.full(len(weights), 1.5 / len(labels))
    penalty[-1] = 0.1 / len(labels)
    # The trace bounds the curvature of the logistic loss. An elementwise
    # gradient step also avoids calling LAPACK from the Qt worker thread.
    step = 1 / (0.25 * np.mean((design ** 2).sum(axis=1)) + penalty.max())
    for _ in range(500):
        logits = (design * weights).sum(axis=1)
        probability = 1 / (1 + np.exp(-np.clip(logits, -60, 60)))
        gradient = (design * (probability - labels)[:, None]).mean(axis=0) + penalty * weights
        delta = step * gradient
        weights -= delta
        if np.max(np.abs(delta)) < 1e-6:
            break
    return FiberModel(tuple(center), tuple(scale), tuple(weights), {})


def train_fiber_model(examples, progress=lambda *_: None):
    labels = np.array([row.label for row in examples])
    traced = [row for row in examples if row.traced and row.label == 1]
    if set(labels) != {0, 1} and not traced:
        raise ValueError("请先保存描画或框选找到的真实缺陷，或分别标注真实缺陷和画面线条。")
    features = np.array([row.features for row in examples])
    scene = features[labels == 0]
    templates = []
    outlines = []
    for row in traced:
        sample = row.features
        if scene.size and _matches(scene, sample).any():
            continue
        if not templates or not _matches(templates, sample).any():
            templates.append(sample)
            outlines.append(row.outline)
        if len(templates) == _MAX_TEMPLATES:
            break
    checked = np.array([row.checked for row in examples])
    metrics = dict(samples=len(labels), defects=int(labels.sum()), scene=int((labels == 0).sum()),
                   validation="none", evaluated=0, scene_deferred=0, defects_deferred=0,
                   baseline_scene=int(np.count_nonzero(checked & (labels == 0))),
                   baseline_defects=int(np.count_nonzero(checked & (labels == 1))))
    for unit in ("roll", "photo") if scene.size else ():
        groups = np.array([getattr(row, unit) for row in examples])
        unique = np.unique(groups)
        if len(unique) < 2 or any(len(set(labels[groups != group])) < 2 for group in unique):
            continue
        # Leave out entire groups. Keep the work bounded for large collections.
        folds = [unique[index::5] for index in range(5)] if len(unique) > 5 else [[group] for group in unique]
        if any(len(set(labels[~np.isin(groups, fold)])) < 2 for fold in folds):
            folds = [[group] for group in unique]
        held_out_scores = np.empty(len(labels))
        for index, fold in enumerate(folds, 1):
            held_out = np.isin(groups, fold)
            model = _fit(features[~held_out], labels[~held_out])
            held_out_scores[held_out] = model._scores(features[held_out])
            progress(index, len(folds))
        deferred = checked & (held_out_scores <= _DEFER_THRESHOLD)
        metrics.update(validation=unit, evaluated=len(labels),
                       scene_deferred=int(np.count_nonzero(deferred & (labels == 0))),
                       defects_deferred=int(np.count_nonzero(deferred & (labels == 1))))
        break
    previews = tuple(outlines) if any(outlines) else ()
    if scene.size:
        fit = _fit(features, labels)
        return FiberModel(fit.center, fit.scale, fit.weights, metrics, tuple(templates), previews)
    return FiberModel((0.,) * _FEATURE_COUNT, (1.,) * _FEATURE_COUNT, (0.,) * (_FEATURE_COUNT + 1),
                      metrics, tuple(templates), previews)


@dataclass(frozen=True)
class FiberModelEntry:
    id: int
    name: str
    source: str
    enabled: bool
    model: FiberModel | FiberModelGroup


class FiberLearningStore:
    def __init__(self, config_path):
        self.path = Path(config_path).with_name("filmclear_fiber_learning.sqlite3")

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path)
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS photos (photo TEXT PRIMARY KEY, roll TEXT NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS examples (
                photo TEXT NOT NULL, candidate TEXT NOT NULL, label INTEGER NOT NULL,
                checked INTEGER NOT NULL, features TEXT NOT NULL, PRIMARY KEY(photo, candidate))""")
            db.execute("CREATE TABLE IF NOT EXISTS models (slot TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS fiber_library (
                id INTEGER PRIMARY KEY, name TEXT NOT NULL, source TEXT NOT NULL,
                enabled INTEGER NOT NULL, payload TEXT NOT NULL UNIQUE)""")
            db.execute("CREATE TABLE IF NOT EXISTS fiber_library_settings (key TEXT PRIMARY KEY)")
            db.execute("""CREATE TABLE IF NOT EXISTS traces (
                photo TEXT NOT NULL, candidate TEXT NOT NULL, geometry TEXT NOT NULL,
                PRIMARY KEY(photo, candidate))""")
            db.execute("""CREATE TABLE IF NOT EXISTS annotation_previews (
                photo TEXT NOT NULL, candidate TEXT NOT NULL, outline TEXT NOT NULL,
                PRIMARY KEY(photo, candidate))""")
        return db

    def photo_labels(self, photo):
        if not self.path.is_file():
            return None, {}
        with closing(self._connect()) as db:
            roll = db.execute("SELECT roll FROM photos WHERE photo = ?", (photo,)).fetchone()
            labels = dict(db.execute("SELECT candidate, label FROM examples WHERE photo = ?", (photo,)))
        return roll[0] if roll else None, labels

    def label(self, photo, roll, key, label, checked, features):
        self.label_many(photo, roll, label, [(key, checked, features, None)])

    def label_many(self, photo, roll, label, samples, *, previews=None):
        with closing(self._connect()) as db, db:
            db.execute("INSERT OR REPLACE INTO photos VALUES (?, ?)", (photo, roll))
            for key, checked, features, trace in samples:
                db.execute("INSERT OR REPLACE INTO examples VALUES (?, ?, ?, ?, ?)",
                           (photo, key, label, checked, json.dumps(features, allow_nan=False)))
                if trace is not None:
                    geometry = dict(x=trace.x, y=trace.y, width=trace.width, height=trace.height,
                                    area=trace.area, center_x=trace.center_x, center_y=trace.center_y,
                                    dust_type=trace.dust_type, confidence=trace.confidence,
                                    selected=trace.selected, face_protected=trace.face_protected,
                                    scene_line=trace.scene_line,
                                    contour=trace.contour.tolist(), holes=[hole.tolist() for hole in trace.holes])
                    db.execute("INSERT OR REPLACE INTO traces VALUES (?, ?, ?)",
                               (photo, key, json.dumps(geometry, allow_nan=False)))
                elif previews and key in previews:
                    outline = _preview_outline(previews[key])
                    db.execute("INSERT OR REPLACE INTO annotation_previews VALUES (?, ?, ?)",
                               (photo, key, json.dumps(outline)))

    def photo_traces(self, photo):
        if not self.path.is_file():
            return []
        with closing(self._connect()) as db:
            rows = db.execute("""SELECT candidate, features, geometry FROM examples
                               JOIN traces USING (photo, candidate) WHERE photo = ?""", (photo,))
            traces = []
            for key, features, geometry in rows:
                data = json.loads(geometry)
                data["contour"] = np.asarray(data["contour"], dtype=np.int32)
                data["holes"] = tuple(np.asarray(hole, dtype=np.int32) for hole in data["holes"])
                traces.append((DustItem(0, **data), key, tuple(json.loads(features))))
            return traces

    def set_roll(self, photo, roll):
        with closing(self._connect()) as db, db:
            db.execute("INSERT OR REPLACE INTO photos VALUES (?, ?)", (photo, roll))

    def remove_label(self, photo, key):
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM examples WHERE photo = ? AND candidate = ?", (photo, key))
            db.execute("DELETE FROM traces WHERE photo = ? AND candidate = ?", (photo, key))
            db.execute("DELETE FROM annotation_previews WHERE photo = ? AND candidate = ?", (photo, key))

    def examples(self):
        if not self.path.is_file():
            return []
        with closing(self._connect()) as db:
            rows = db.execute("""SELECT photo, roll, label, checked, features, geometry, outline
                               FROM examples JOIN photos USING (photo)
                               LEFT JOIN traces USING (photo, candidate)
                               LEFT JOIN annotation_previews USING (photo, candidate)
                               ORDER BY examples.rowid""")
            examples = []
            for photo, roll, label, checked, features, geometry, outline in rows:
                if geometry is not None:
                    points = _preview_outline(json.loads(geometry)["contour"])
                else:
                    points = tuple(map(tuple, json.loads(outline))) if outline else ()
                examples.append(FiberExample(photo, roll, label, bool(checked), tuple(json.loads(features)),
                                             geometry is not None, points))
            return examples

    def counts(self):
        if not self.path.is_file():
            return 0, 0, 0
        with closing(self._connect()) as db:
            counts = Counter(dict(db.execute("SELECT label, COUNT(*) FROM examples GROUP BY label")))
            rolls, = db.execute("SELECT COUNT(DISTINCT roll) FROM examples JOIN photos USING (photo)").fetchone()
        return counts[1], counts[0], rolls

    def _prepare_library(self, db, legacy_enabled):
        db.execute("BEGIN IMMEDIATE")
        migrating = db.execute("PRAGMA user_version").fetchone()[0] == 0
        if migrating:
            names = {"trained": "本机训练", "previous": "上一个模型", "active": "此前启用的模型"}
            for slot, payload in db.execute("""
                    SELECT slot, payload FROM models
                    ORDER BY CASE slot WHEN 'trained' THEN 0 WHEN 'previous' THEN 1 ELSE 2 END"""):
                if slot in names:
                    entry = self._add_model(db, names[slot], "", FiberModel.decode(payload))
                    if slot == "active" and legacy_enabled:
                        db.execute("UPDATE fiber_library SET enabled = 1 WHERE id = ?", (entry,))
            db.execute("PRAGMA user_version = 1")
        if _BUNDLED_MODEL.is_file() and not db.execute(
                "SELECT 1 FROM fiber_library_settings WHERE key = 'bundled_registered'"
        ).fetchone():
            has_model = db.execute("SELECT 1 FROM fiber_library LIMIT 1").fetchone()
            if not db.execute("SELECT 1 FROM fiber_library WHERE source = ?",
                              (str(_BUNDLED_MODEL),)).fetchone():
                entry = self._add_model(db, "随附模型", str(_BUNDLED_MODEL), read_fiber_model(_BUNDLED_MODEL))
                if migrating and legacy_enabled and not has_model:
                    db.execute("UPDATE fiber_library SET enabled = 1 WHERE id = ?", (entry,))
            db.execute("INSERT INTO fiber_library_settings VALUES ('bundled_registered')")

    @staticmethod
    def _add_model(db, name, source, model):
        payload = model.encode()
        db.execute("""INSERT INTO fiber_library (name, source, enabled, payload) VALUES (?, ?, 0, ?)
                      ON CONFLICT(payload) DO UPDATE SET name = excluded.name, source = excluded.source
                      WHERE fiber_library.source = '' AND excluded.source <> ''""",
                   (name, source, payload))
        return db.execute("SELECT id FROM fiber_library WHERE payload = ?", (payload,)).fetchone()[0]

    def library(self, *, legacy_enabled=True):
        if not self.path.is_file() and not _BUNDLED_MODEL.is_file():
            return []
        with closing(self._connect()) as db, db:
            self._prepare_library(db, legacy_enabled)
            return [FiberModelEntry(identifier, name, source, bool(enabled), FiberModel.decode(payload))
                    for identifier, name, source, enabled, payload in db.execute(
                        "SELECT id, name, source, enabled, payload FROM fiber_library ORDER BY id DESC")]

    def add_models(self, models):
        with closing(self._connect()) as db, db:
            self._prepare_library(db, True)
            return [self._add_model(db, name, source, model) for name, source, model in models]

    def set_enabled(self, ids, enabled):
        with closing(self._connect()) as db, db:
            self._prepare_library(db, True)
            db.executemany("UPDATE fiber_library SET enabled = ? WHERE id = ?",
                           ((enabled, identifier) for identifier in ids))

    def remove_models(self, ids):
        with closing(self._connect()) as db, db:
            self._prepare_library(db, True)
            db.executemany("DELETE FROM fiber_library WHERE id = ?", ((identifier,) for identifier in ids))

    def active_model(self, *, legacy_enabled=True):
        return combine_fiber_models(entry.model for entry in self.library(legacy_enabled=legacy_enabled)
                                   if entry.enabled)

    def save_trained(self, model):
        name = datetime.now().strftime("本机训练 · %Y-%m-%d %H:%M:%S")
        with closing(self._connect()) as db, db:
            self._prepare_library(db, True)
            entry = self._add_model(db, name, "", model)
            db.execute("""INSERT OR REPLACE INTO models (slot, payload)
                          SELECT 'trained', payload FROM fiber_library WHERE id = ?""", (entry,))
            return entry

    def latest_trained(self):
        if self.path.is_file():
            with closing(self._connect()) as db:
                row = db.execute("SELECT payload FROM models WHERE slot = 'trained'").fetchone()
                if row:
                    return FiberModel.decode(row[0])
        return None
