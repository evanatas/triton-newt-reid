"""Схема label-record — контракт между бутстрапом, кликером и экспортом в YOLO. ЧИСТО (json).

Один JSON на кадр: <label_dir>/<md5>.json. Координаты НОРМАЛИЗОВАНЫ [0,1] (не зависят от
размера) → прямой экспорт в YOLO и лёгкий пересчёт под любой дисплей кликера.
"""
import json
from dataclasses import dataclass, replace
from pathlib import Path


def _xy(p):
    return None if p is None else (float(p[0]), float(p[1]))


@dataclass(frozen=True)
class LabelRecord:
    md5: str
    rel_path: str
    cohort: str
    individual_id: str
    img_w: int
    img_h: int
    belly_polygon: tuple              # ((xn, yn), ...) нормализованные вершины
    head_xy: tuple = None             # (xn, yn) | None
    cloaca_xy: tuple = None           # (xn, yn) | None
    source: str = "pseudo"            # pseudo | manual
    status: str = "auto"              # auto | corrected | redraw | skip
    flags: tuple = ()                 # band_suspect | low_head_conf | redraw | ...
    head_conf: float = 0.0
    band_conf: float = 0.0
    pipeline: str = ""

    def to_dict(self):
        return {
            "md5": self.md5, "rel_path": self.rel_path, "cohort": self.cohort,
            "individual_id": self.individual_id, "img_w": int(self.img_w), "img_h": int(self.img_h),
            "belly_polygon": [[float(x), float(y)] for x, y in self.belly_polygon],
            "head_xy": list(self.head_xy) if self.head_xy else None,
            "cloaca_xy": list(self.cloaca_xy) if self.cloaca_xy else None,
            "source": self.source, "status": self.status, "flags": list(self.flags),
            "head_conf": float(self.head_conf), "band_conf": float(self.band_conf),
            "pipeline": self.pipeline,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            md5=d["md5"], rel_path=d["rel_path"], cohort=d["cohort"],
            individual_id=d["individual_id"], img_w=int(d["img_w"]), img_h=int(d["img_h"]),
            belly_polygon=tuple((float(x), float(y)) for x, y in d["belly_polygon"]),
            head_xy=_xy(d.get("head_xy")), cloaca_xy=_xy(d.get("cloaca_xy")),
            source=d.get("source", "pseudo"), status=d.get("status", "auto"),
            flags=tuple(d.get("flags", ())),
            head_conf=float(d.get("head_conf", 0.0)), band_conf=float(d.get("band_conf", 0.0)),
            pipeline=d.get("pipeline", ""),
        )

    # --- мутации (frozen → новый рекорд; любая правка человеком ⇒ source="manual") ---
    def flip(self):
        """Перевернуть ориентацию: поменять местами голову и клоаку (частый фикс 180°)."""
        return replace(self, head_xy=self.cloaca_xy, cloaca_xy=self.head_xy, source="manual")

    def set_head(self, xy):
        return replace(self, head_xy=_xy(xy), source="manual")

    def set_cloaca(self, xy):
        return replace(self, cloaca_xy=_xy(xy), source="manual")

    def mark(self, status, *flags):
        merged = tuple(dict.fromkeys(self.flags + tuple(flags)))
        return replace(self, status=status, flags=merged, source="manual")


def write_label(rec, label_dir) -> Path:
    d = Path(label_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{rec.md5}.json"
    p.write_text(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def read_label(path) -> LabelRecord:
    return LabelRecord.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def scan_labels(label_dir):
    """Все рекорды каталога, отсортированы по md5 (детерминизм)."""
    return [read_label(p) for p in sorted(Path(label_dir).glob("*.json"), key=lambda x: x.stem)]
