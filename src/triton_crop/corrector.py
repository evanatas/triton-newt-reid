"""Мини-кликер ручной правки псевдо-меток брюшка (cv2 highgui).

Чистые пересчёты координат (норм↔пиксели дисплея) — тестируются; интерактивная петля — оболочка.
Headless-режим `--selftest out.png` рисует первую метку в файл (проверка рендера без окна).

Управление:
  ЛКМ — поставить ГОЛОВУ (зелёная)     ПКМ — поставить КЛОАКУ (красная)
  f — флип голова↔клоака (фикс 180°)    r — пере-сегментировать: затем ЛКМ по телу (нужен --reseg/SAM)
  a — принять как верное (corrected)    x — кадр негоден (skip)     d — пометить «перерисовать маску»
  n / пробел — сохранить и далее         b — назад                   u — откатить кадр из файла
  q / Esc — выход (с сохранением)
Прогресс пишется в JSON сразу при переходе между кадрами.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from triton_data.imageio import load_canonical

from .labelio import read_label, scan_labels, write_label
from .pseudo_label import derive_pseudo_label
from .viz import draw_overlay

HELP = ("ЛКМ голова | ПКМ клоака | f флип | r ре-сегм(клик) | a верно | "
        "x негодно | d перерисовать | n/пробел дальше | b назад | u откат | q выход")


def to_px(xy_norm, w, h):
    return (int(round(xy_norm[0] * w)), int(round(xy_norm[1] * h)))


def to_norm(xy_px, w, h):
    return (xy_px[0] / w, xy_px[1] / h)


def _disp(rgb, disp_max):
    h, w = rgb.shape[:2]
    s = min(disp_max / max(h, w), 1.0)
    return cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), s


def _render(rec, disp_rgb):
    h, w = disp_rgb.shape[:2]
    poly = [to_px(p, w, h) for p in rec.belly_polygon] if rec.belly_polygon else None
    head = to_px(rec.head_xy, w, h) if rec.head_xy else None
    cloaca = to_px(rec.cloaca_xy, w, h) if rec.cloaca_xy else None
    return draw_overlay(disp_rgb, None, poly, head, cloaca,
                        redraw=("redraw" in rec.flags or rec.status == "redraw"),
                        label=f"{rec.cohort} {rec.individual_id} [{rec.status}] {','.join(rec.flags)}")


def render_record_to_file(labels_dir, workspace_root, out_png, index=0, disp_max=900):
    """Headless: отрисовать метку[index] в PNG (проверка рендера без GUI)."""
    rec = scan_labels(labels_dir)[index]
    orig = np.array(load_canonical(Path(workspace_root) / rec.rel_path))
    disp, _ = _disp(orig, disp_max)
    cv2.imwrite(str(out_png), _render(rec, disp))
    return out_png


def run_corrector(labels_dir, workspace_root, masker=None, disp_max=900, pending_only=False):
    labels_dir, workspace_root = Path(labels_dir), Path(workspace_root)
    recs = scan_labels(labels_dir)
    if pending_only:
        recs = [r for r in recs if r.source == "pseudo"]   # только нетронутые (даёт resume)
    md5s = [r.md5 for r in recs]
    if not md5s:
        print("Нет кадров для правки в", labels_dir, "(всё обработано?)")
        return
    print(f"К правке: {len(md5s)} кадров")
    print(HELP)
    win = "triton corrector — правка пузика"
    cv2.namedWindow(win)
    st = {"i": 0, "reseg": False, "rec": None, "orig": None, "disp": None}

    def load(i):
        rec = read_label(labels_dir / f"{md5s[i]}.json")
        orig = np.array(load_canonical(workspace_root / rec.rel_path))
        disp, _ = _disp(orig, disp_max)
        st.update(i=i, rec=rec, orig=orig, disp=disp, reseg=False)

    def on_mouse(event, x, y, flags, _):
        rec = st["rec"]
        h, w = st["disp"].shape[:2]
        if event == cv2.EVENT_LBUTTONDOWN:
            if st["reseg"] and masker is not None:
                mask, work = masker.mask_from_point(st["orig"], to_norm((x, y), w, h))
                st["reseg"] = False
                if mask is not None:
                    pl = derive_pseudo_label(mask)
                    wh, ww = work.shape[:2]
                    st["rec"] = replace(
                        rec, source="manual", status="corrected",
                        belly_polygon=tuple((px / ww, py / wh) for px, py in pl.belly_polygon),
                        head_xy=(pl.head_xy[0] / ww, pl.head_xy[1] / wh),
                        cloaca_xy=(pl.cloaca_xy[0] / ww, pl.cloaca_xy[1] / wh))
            else:
                st["rec"] = rec.set_head(to_norm((x, y), w, h))
        elif event == cv2.EVENT_RBUTTONDOWN:
            st["rec"] = rec.set_cloaca(to_norm((x, y), w, h))

    cv2.setMouseCallback(win, on_mouse)
    load(0)
    while True:
        cv2.imshow(win, _render(st["rec"], st["disp"]))
        k = cv2.waitKey(20) & 0xFF
        rec = st["rec"]
        if k in (ord("q"), 27):
            write_label(rec, labels_dir)
            break
        elif k == ord("f"):
            st["rec"] = rec.flip()
        elif k == ord("a"):
            st["rec"] = rec.mark("corrected")
        elif k == ord("x"):
            st["rec"] = rec.mark("skip")
        elif k == ord("d"):
            st["rec"] = rec.mark("redraw", "redraw")
        elif k == ord("r"):
            st["reseg"] = True
        elif k == ord("u"):
            st["rec"] = read_label(labels_dir / f"{rec.md5}.json")
        elif k in (ord("n"), ord(" ")):
            write_label(rec.mark("corrected") if rec.status == "auto" else rec, labels_dir)
            if st["i"] + 1 < len(md5s):
                load(st["i"] + 1)
            else:
                print("Конец пачки — выход.")
                break
        elif k == ord("b"):
            write_label(rec.mark("corrected") if rec.status == "auto" else rec, labels_dir)
            load(max(0, st["i"] - 1))
    cv2.destroyAllWindows()
    print("Сохранено в", labels_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m triton_crop.corrector")
    ap.add_argument("--labels", default="artifacts/labels_pilot")
    ap.add_argument("--workspace", default=None,
                    help="корень с сырыми данными; по умолчанию — workspace_root из configs/paths.yaml")
    ap.add_argument("--reseg", action="store_true", help="включить ре-сегментацию по клику (грузит SAM)")
    ap.add_argument("--selftest", metavar="OUT.png", help="headless: отрисовать первую метку в PNG и выйти")
    a = ap.parse_args(argv)
    if a.workspace is None:                       # без хардкода личного пути — берём из конфигурации
        from triton_data.config import load_config
        a.workspace = str(load_config(validate_dirs=False).workspace_root)
    if a.selftest:
        print("selftest:", render_record_to_file(a.labels, a.workspace, a.selftest))
        return
    masker = None
    if a.reseg:
        from .sam_bootstrap import SamMasker
        masker = SamMasker("sam2_b.pt")
    run_corrector(a.labels, a.workspace, masker=masker)


if __name__ == "__main__":
    main()
