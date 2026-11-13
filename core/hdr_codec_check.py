"""Exercise HDR imports and exports inside the packaged application."""

from traceback import format_exc

from .batch_export import SOURCE_FORMAT, ExportEntry, export_batch
from .export_metadata import read_metadata
from .hdr_export import PQ_HEIC, ULTRA_HDR_JPEG
from .hdr_input import read_gain_map
from .scan_io import read_scan


def check_hdr_codecs(demo, log):
    try:
        tiff = demo.parent / "HDR-window-16bit.tif"
        saved, errors = export_batch([ExportEntry(tiff, None, (), ())], log.parent,
                                     SOURCE_FORMAT, "edited", {}, lambda *_: None, lambda: False)
        if errors:
            raise RuntimeError(f"{tiff.name} → TIFF: {errors}")
        original, exported = read_metadata(tiff), read_metadata(saved[0][2])
        if exported.depth != 16 or exported.dpi != (2400, 2400) or exported.icc != original.icc:
            raise RuntimeError("TIFF export did not retain bit depth, DPI and ICC profile")
        examples = {
            ULTRA_HDR_JPEG: demo / "HDR-window-UltraHDR.jpg",
            PQ_HEIC: demo / "HDR-window-PQ.heic",
        }
        for source in examples.values():
            scan = read_scan(source, read_ir=False)
            height, width = scan.image.shape[:2]
            crop = {"rect": [2, 2, width - 2, height - 2], "rotation": 0}
            for format_id in examples:
                saved, errors = export_batch([ExportEntry(source, None, (), (), crop=crop)],
                                             log.parent, format_id, "edited", {},
                                             lambda *_: None, lambda: False)
                if errors:
                    raise RuntimeError(f"{source.name} → {format_id}: {errors}")
                target = saved[0][2]
                rendered = read_scan(target, read_ir=False, keep_pq=True)
                if rendered.image.shape[:2] != (height - 4, width - 4):
                    raise RuntimeError(f"Unexpected crop: {target.name}")
                if read_metadata(target).hdr_format != format_id:
                    raise RuntimeError(f"Missing HDR metadata: {target.name}")
                if format_id == PQ_HEIC and rendered.native_pq is None:
                    raise RuntimeError(f"Missing PQ pixels: {target.name}")
                if format_id == ULTRA_HDR_JPEG and read_gain_map(target, rendered.image.shape) is None:
                    raise RuntimeError(f"Missing gain map: {target.name}")
    except Exception:
        log.write_text(format_exc(), encoding="utf-8")
        return 1
    log.write_text("16-bit TIFF export and HEIC / Ultra HDR JPEG round trips passed.", encoding="utf-8")
    return 0
