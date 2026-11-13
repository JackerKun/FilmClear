from io import BytesIO
from pathlib import Path
from struct import unpack_from
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
import pillow_heif
from PIL import Image, PngImagePlugin
import tifffile

from core.batch_export import SOURCE_FORMAT, ExportEntry, export_batch
from core.hdr_export import PQ_HEIC, ULTRA_HDR_JPEG, srgb_profile
from core.hdr_input import read_gain_map
from core.review_report import create_review_report
from core.scan_io import display_pixels, read_scan


class HdrExportTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / "测试扫描.tif"
        self.ramp = np.broadcast_to(np.linspace(0, 65535, 257, dtype=np.uint16)[None, :, None],
                                    (73, 257, 3)).copy()
        profile = srgb_profile()
        tifffile.imwrite(self.source, self.ramp, photometric="rgb", software="Film Scanner",
                         extratags=[(34675, "B", len(profile), profile, True)])
        self.entry = ExportEntry(self.source, None, (), ())

    def export(self, entries, format_id, mode="edited"):
        return export_batch(entries, self.directory, format_id, mode, {}, lambda *_: None, lambda: False)

    def test_ultra_hdr_jpeg_has_sdr_base_and_a_locatable_gain_map(self):
        saved, errors = self.export([self.entry], ULTRA_HDR_JPEG)
        self.assertEqual(errors, [])
        target = saved[0][2]
        self.assertEqual(target.name, "测试扫描_hdr.jpg")
        payload = target.read_bytes()
        mp_start = payload.index(b"MPF\0") + 4
        with Image.open(target) as base:
            self.assertEqual(base.size, (257, 73))
            self.assertEqual(base.info["icc_profile"], srgb_profile())
            self.assertEqual(base.getexif()[305], "Film Scanner")
            self.assertIn(b'Item:Semantic="GainMap"', base.info["xmp"])
            self.assertAlmostEqual(int(np.asarray(base)[35, 128, 0]), 128, delta=2)
        primary_size = unpack_from(">I", payload, mp_start + 54)[0]
        gain_size, gain_offset = unpack_from(">II", payload, mp_start + 70)
        self.assertEqual(primary_size, mp_start + gain_offset)
        self.assertEqual(len(payload), primary_size + gain_size)
        self.assertEqual(payload.count(b"urn:iso:std:iso:ts:21496:-1\0"), 2)
        with Image.open(BytesIO(payload[primary_size:])) as gain:
            self.assertEqual(gain.mode, "L")
            self.assertEqual(gain.size, (257, 73))
            self.assertIn(b'hdrgm:GainMapMax="1"', gain.info["xmp"])
            self.assertEqual(int(np.asarray(gain)[35, 128]), 0)
            self.assertGreater(int(np.asarray(gain)[35, 255]), 248)

    def test_pq_heic_uses_10_bit_pq_and_sdr_report(self):
        saved, errors = self.export([self.entry], PQ_HEIC)
        self.assertEqual(errors, [])
        target = saved[0][2]
        heic = pillow_heif.open_heif(target, convert_hdr_to_8bit=False)
        self.assertEqual(heic.size, (257, 73))
        self.assertEqual(heic.info["bit_depth"], 10)
        nclx = heic.info["nclx_profile"]
        self.assertEqual(tuple(nclx[field] for field in
                               ("color_primaries", "transfer_characteristics", "matrix_coefficients")), (9, 16, 9))
        # ST 2084 encodes 203 and 406 nits at roughly 0.58 and 0.65.
        pq = np.asarray(heic)
        self.assertAlmostEqual(pq[35, 256, 0] / 65535, 0.654, delta=0.01)
        preview = display_pixels(read_scan(target, read_ir=False).image)
        np.testing.assert_allclose(preview, (self.ramp / 257).round(), atol=3)
        report = create_review_report([self.entry], saved, self.directory, "edited",
                                      lambda *_: None, lambda: False)
        self.assertEqual(report.read_text(encoding="utf-8").count("data:image/jpeg;base64,"), 2)

    def test_hdr_batch_reports_incompatible_inputs_and_continues(self):
        jpg = self.directory / "eight-bit.jpg"
        Image.new("RGB", (32, 32)).save(jpg)
        png = self.directory / "linear.png"
        png_info = PngImagePlugin.PngInfo()
        png_info.add(b"gAMA", (100000).to_bytes(4, "big"))
        Image.fromarray(self.ramp[..., 0]).save(png, pnginfo=png_info)
        pq_png = self.directory / "pq.png"
        pq_info = PngImagePlugin.PngInfo()
        pq_info.add(b"cICP", bytes((9, 16, 0, 1)))
        Image.fromarray(self.ramp[..., 0]).save(pq_png, pnginfo=pq_info)
        entries = [ExportEntry(path, None, (), ()) for path in (jpg, png, pq_png, self.source)]
        saved, errors = self.export(entries, ULTRA_HDR_JPEG)
        self.assertEqual([item[0] for item in saved], [self.source])
        self.assertEqual([path for path, _ in errors], [jpg, png, pq_png])
        self.assertIn("16 位", errors[0][1])
        self.assertIn("sRGB", errors[1][1])
        self.assertIn("sRGB", errors[2][1])
        self.assertFalse(list(self.directory.glob(".filmclear-*")))
        copied, errors = self.export([self.entry], PQ_HEIC, mode="original")
        self.assertEqual(errors, [])
        self.assertEqual(copied[0][2].read_bytes(), self.source.read_bytes())

    def test_hdr_applies_full_resolution_patch_before_crop(self):
        y, x = 30, 140
        crop = {"rect": [30, 10, 227, 63], "rotation": 0}
        patch_positions = np.array([row * 257 + col for row in range(y - 5, y + 6)
                                    for col in range(x - 5, x + 6)])
        patches = ((1, patch_positions, np.full((len(patch_positions), 3), 65535, np.uint16)),)
        entry = ExportEntry(self.source, None, patches, (), crop=crop)
        saved, errors = self.export([entry], ULTRA_HDR_JPEG)
        self.assertEqual(errors, [])
        base = cv2.imread(str(saved[0][2]))
        self.assertEqual(base.shape[:2], (53, 197))
        self.assertGreater(int(base[y - 10, x - 30, 0]), 245)

    def test_hdr_roundtrip_crops_original_highlights_and_copies_unchanged_files(self):
        crop = {"rect": [8, 10, 257, 63], "rotation": 0}
        for fmt in (ULTRA_HDR_JPEG, PQ_HEIC):
            saved, errors = self.export([self.entry], fmt)
            self.assertFalse(errors)
            source = saved[0][2]
            imported = read_scan(source, read_ir=False, keep_pq=True)
            copied, errors = self.export([ExportEntry(source, None, (), ())], SOURCE_FORMAT)
            self.assertFalse(errors)
            self.assertEqual(copied[0][2].read_bytes(), source.read_bytes())
            entry = ExportEntry(source, None, (), (), crop=crop)
            for output in (ULTRA_HDR_JPEG, PQ_HEIC):
                with self.subTest(source=fmt, output=output):
                    edited, errors = self.export([entry], output)
                    self.assertFalse(errors)
                    target = edited[0][2]
                    scan = read_scan(target, read_ir=False, keep_pq=True)
                    self.assertEqual(scan.image.shape, (53, 249, 3))
                    if output == PQ_HEIC:
                        self.assertAlmostEqual(scan.native_pq[25, -1, 0] / 65535, 0.654, delta=0.01)
                        if fmt == PQ_HEIC:
                            np.testing.assert_allclose(scan.native_pq, imported.native_pq[10:63, 8:], atol=256)
                    else:
                        gain = read_gain_map(target, scan.image.shape)
                        self.assertGreater(gain.pixels[25, -1], 240)
                        self.assertLess(gain.pixels[25, 100], 5)

    def test_pq_patch_leaves_unedited_highlights_and_replaces_edited_pixels(self):
        saved, _ = self.export([self.entry], PQ_HEIC)
        source = saved[0][2]
        before = read_scan(source, read_ir=False, keep_pq=True)
        positions = np.array([row * 257 + col for row in range(22, 50) for col in range(220, 244)])
        values = np.full((len(positions), 3), 20000, np.uint16)
        saved, errors = self.export([ExportEntry(source, None, ((1, positions, values),), ())], SOURCE_FORMAT)
        self.assertFalse(errors)
        after = read_scan(saved[0][2], read_ir=False, keep_pq=True)
        self.assertLess(after.native_pq[35, 232, 0], before.native_pq[35, 232, 0] - 7000)
        self.assertAlmostEqual(int(after.native_pq[8, 247, 0]), int(before.native_pq[8, 247, 0]), delta=256)

    def test_sdr_heif_import_and_source_format_export(self):
        path = self.directory / "普通照片.heif"
        pixels = np.full((64, 96, 3), (202, 115, 64), np.uint8)
        pillow_heif.encode("RGB", (96, 64), pixels.tobytes(), path, quality=95)
        scan = read_scan(path, read_ir=False)
        np.testing.assert_allclose(scan.image[32, 48], (64, 115, 202), atol=4)
        entry = ExportEntry(path, None, (), (), crop={"rect": [8, 8, 88, 56], "rotation": 0})
        saved, errors = self.export([entry], SOURCE_FORMAT)
        self.assertFalse(errors)
        self.assertEqual(saved[0][2].suffix, ".heif")
        target = read_scan(saved[0][2], read_ir=False, keep_pq=True)
        self.assertIsNone(target.native_pq)
        self.assertEqual(target.image.shape, (48, 80, 3))
        np.testing.assert_allclose(target.image[24, 40], (64, 115, 202), atol=4)
