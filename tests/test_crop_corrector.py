"""Тесты пересчёта координат кликера (нормализованные ↔ пиксели дисплея) + headless-рендера метки."""
from triton_crop.corrector import render_record_to_file, to_norm, to_px


def test_to_px_rounds_and_scales():
    assert to_px((0.25, 0.5), 800, 600) == (200, 300)


def test_norm_px_roundtrip():
    w, h = 800, 600
    assert to_norm(to_px((0.25, 0.5), w, h), w, h) == (0.25, 0.5)


def test_render_record_to_file_writes_png(tmp_path):
    # headless-рендер метки[index] в PNG — путь --selftest, без GUI/весов (чистый cv2/numpy)
    from PIL import Image

    from triton_crop.labelio import LabelRecord, write_label
    ws = tmp_path / "ws"
    ws.mkdir()
    Image.new("RGB", (64, 48), (90, 120, 90)).save(ws / "img.jpg")
    rec = LabelRecord(md5="m0", rel_path="img.jpg", cohort="TK", individual_id="TK_1",
                      img_w=64, img_h=48,
                      belly_polygon=((0.3, 0.2), (0.7, 0.2), (0.7, 0.8), (0.3, 0.8)),
                      head_xy=(0.5, 0.1), cloaca_xy=(0.5, 0.9))
    labels = tmp_path / "labels"
    write_label(rec, labels)
    out = render_record_to_file(labels, ws, tmp_path / "r.png", index=0)
    assert out.exists() and out.stat().st_size > 0
