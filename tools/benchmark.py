"""Repeatable film-defect checks using local scans and controlled damage."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from core.dust_remover import DustRemover
from core.scan_detection import detect_brush, detect_scan
from core.scan_io import read_scan

PARAMS = {"threshold": 20, "min_area": 3, "min_confidence": 0.34}
LAMA_MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"
CHECKS = (
    ("2026-10-28-0001.jpg", ((1879, 6635),), ([(1854, 6618), (1884, 6618)], 28)),
    ("2026-10-29-0007.jpg", ((2520, 430),), None),
    ("2026-10-29-0016.jpg", (), None),
)


def hits(items, points):
    return sum(any(cv2.pointPolygonTest(item.contour, point, True) >= -8
                   for item in items if item.contour is not None) for point in points)


def detection_report(directory, reviews, repeats):
    reports = []
    for name, expected, brush in CHECKS:
        path = directory / name
        if not path.is_file():
            continue
        scan = read_scan(path)
        for mode in ("全图", "描画补查") if brush else ("全图",):
            timings = []
            for _ in range(repeats):
                started = perf_counter()
                if mode == "全图":
                    result = detect_scan(scan.image, PARAMS, infrared=scan.infrared)
                else:
                    result = detect_brush(scan.image, PARAMS, *brush, infrared=scan.infrared)
                timings.append(perf_counter() - started)
            items = result.items
            controls = None
            if name.endswith("0016.jpg"):
                controls = sum(item.center_y <= 650 or item.center_y >= 3600 for item in items)
            rejected = reviews.get(name, {}).get(mode, {}).get("rejected_points")
            reports.append({
                "photo": name, "mode": mode, "source": result.source, "candidates": len(items),
                "marked_defects": len(expected), "marked_missed": len(expected) - hits(items, expected),
                "perforation_band_candidates": controls,
                "review_rejections_matched": hits(items, [tuple(point) for point in rejected])
                if rejected is not None else None,
                "p50_seconds": round(float(np.percentile(timings, 50)), 3),
                "p95_seconds": round(float(np.percentile(timings, 95)), 3),
            })
    return reports


def load_lama(path):
    with path.open("rb") as stream:
        digest = hashlib.md5()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != LAMA_MD5:
        raise ValueError("LaMa 权重与 IOPaint 发布的 big-lama.pt 校验值不符")
    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = torch.jit.load(str(path), map_location="cpu").eval()

    def apply(image, mask):
        rgb = np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1), dtype=np.float32) / 255
        selection = np.float32(mask[None, None] != 0)
        with torch.inference_mode():
            output = model(torch.from_numpy(rgb[None]), torch.from_numpy(selection))
        restored = np.clip(np.rint(output[0].numpy().transpose(1, 2, 0)[..., ::-1] * 255),
                           0, 255).astype(np.uint8)
        return np.where(mask[..., None] != 0, restored, image)

    return apply


def repair_report(directory, lama, previews):
    methods = {name: DustRemover(name).apply_mask for name in ("ns", "grain")}
    if lama:
        methods["LaMa"] = load_lama(lama)
    results = []
    for name, (x, y), case in (
        ("2026-10-28-0001.jpg", (1670, 6440), "暗部纹理"),
        ("2026-10-29-0007.jpg", (2480, 380), "发际线与肤色"),
    ):
        path = directory / name
        if not path.is_file():
            continue
        image = read_scan(path, read_ir=False).image
        clean = np.ascontiguousarray(image[y:y + 256, x:x + 256])
        if clean.shape != (256, 256, 3) or clean.dtype != np.uint8:
            continue
        mask = np.zeros(clean.shape[:2], np.uint8)
        polygon = np.int32([(81, 103), (118, 91), (137, 106), (159, 98), (168, 144),
                            (152, 163), (116, 165), (85, 151)])
        cv2.fillPoly(mask, [polygon], 255)
        area = mask != 0
        damaged = np.where(area[..., None], np.uint8(233), clean)
        images = [clean, damaged]
        for method, apply in methods.items():
            started = perf_counter()
            repaired = apply(damaged, mask)
            duration = perf_counter() - started
            error = np.mean((repaired[area].astype(np.float32) - clean[area]) ** 2)
            results.append({"photo": name, "case": case, "method": method,
                            "mask_psnr_db": round(10 * np.log10(255 ** 2 / max(error, 1e-8)), 2),
                            "seconds": round(duration, 3),
                            "outside_unchanged": bool(np.array_equal(repaired[~area], clean[~area]))})
            images.append(repaired)
        if previews:
            previews.mkdir(parents=True, exist_ok=True)
            labeled = []
            for label, pixels in zip(("Reference", "Injected", *methods), images):
                frame = cv2.copyMakeBorder(pixels, 24, 0, 0, 0, cv2.BORDER_CONSTANT)
                cv2.putText(frame, label, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                labeled.append(frame)
            cv2.imwrite(str(previews / f"{Path(name).stem}-repair.png"), np.hstack(labeled))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, default=Path("demo"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--review", type=Path, help="JSON: file -> mode -> rejected_points")
    parser.add_argument("--lama-model", type=Path, help="Optional verified IOPaint big-lama.pt")
    parser.add_argument("--repair-previews", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    cv2.setNumThreads(4)
    reviews = json.loads(args.review.read_text()) if args.review else {}
    report = {
        "detection": detection_report(args.demo, reviews, args.repeats),
        "inpainting": repair_report(args.demo, args.lama_model, args.repair_previews),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
