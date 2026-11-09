"""Local, transactional workspace snapshots with lossless sparse pixel patches."""

from contextlib import closing
from dataclasses import fields
from io import BytesIO
import json
from pathlib import Path
import sqlite3

import numpy as np

from .dust_item import DustItem


def _encode(snapshot):
    arrays = {}
    array_keys = {}

    def encode(value):
        if isinstance(value, DustItem):
            return {"dust": {field.name: encode(getattr(value, field.name)) for field in fields(value)}}
        if isinstance(value, np.ndarray):
            key = array_keys.setdefault(id(value), f"a{len(arrays)}")
            arrays[key] = value
            return {"array": key}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        return value

    content = json.dumps(encode(snapshot), ensure_ascii=False).encode()
    with BytesIO() as stream:
        np.savez_compressed(stream, manifest=np.frombuffer(content, np.uint8), **arrays)
        return stream.getvalue()


def _decode(data):
    with np.load(BytesIO(data), allow_pickle=False) as arrays:
        decoded = {}

        def decode(value):
            if isinstance(value, dict):
                if "array" in value:
                    key = value["array"]
                    if key not in decoded:
                        decoded[key] = arrays[key]
                    return decoded[key]
                if "dust" in value:
                    record = {key: decode(item) for key, item in value["dust"].items()}
                    record["holes"] = tuple(record.get("holes", ()))
                    return DustItem(**record)
                return {key: decode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [decode(item) for item in value]
            return value

        return decode(json.loads(arrays["manifest"].tobytes()))


class WorkspaceStore:
    def __init__(self, config_path):
        self.path = Path(config_path).with_name("filmclear_workspace.sqlite3")

    def load(self):
        if not self.path.is_file():
            return {}, {}
        with closing(sqlite3.connect(self.path)) as db:
            meta = db.execute("SELECT data FROM workspace WHERE id = 1").fetchone()
            if meta is None:
                return {}, {}
            manifest = json.loads(meta[0])
            if manifest.get("version") != 1:
                raise ValueError("工作区版本不受支持")
            snapshots = {path: _decode(data) for path, data in db.execute("SELECT path, data FROM photo")}
            return manifest, snapshots

    def save(self, paths, active, snapshots, views):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS workspace (id INTEGER PRIMARY KEY, data TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS photo (path TEXT PRIMARY KEY, data BLOB NOT NULL)")
            db.executemany("INSERT OR REPLACE INTO photo VALUES (?, ?)",
                           ((path, _encode(state)) for path, state in snapshots.items()))
            db.execute("INSERT OR REPLACE INTO workspace VALUES (1, ?)", (
                json.dumps({"version": 1, "paths": paths, "active": active, "views": views}, ensure_ascii=False),
            ))
