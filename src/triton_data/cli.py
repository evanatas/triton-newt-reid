"""CLI слоя данных.

  python -m triton_data.cli build      # собрать data/manifest.csv + data/manifest_external.csv
  python -m triton_data.cli validate   # 12 гейтов + reports/eda.md
"""
import argparse
from pathlib import Path

import pandas as pd

from .config import load_config
from .loader import read_manifests
from .manifest import build_and_write
from .validate import run_gates, write_eda

_REPO = Path(__file__).resolve().parents[2]
_DATA = _REPO / "data"
_REPORTS = _REPO / "reports"


def _summary(target: pd.DataFrame, external: pd.DataFrame) -> str:
    lines = []
    for cohort, sub in target.groupby("cohort"):
        k = sub[sub.dup_keep == True]  # noqa: E712
        scope = k["kpi_scope"].iloc[0] if len(k) else "—"
        folds = {f: int((k.split_fold == f).sum()) for f in ("train", "dev", "test")}
        lines.append(
            f"  {cohort} [{scope}]: файлов {len(sub)}, после дедупа {len(k)}, особей {k.individual_id.nunique()}, "
            f"gallery {int((k.split_role == 'gallery').sum())}, probe {int((k.split_role == 'probe').sum())}, "
            f"fold {folds}")
    ek = external[external.dup_keep == True]  # noqa: E712
    lines.append(f"  GCN(ext): файлов {len(external)}, после дедупа {len(ek)}, особей {ek.individual_id.nunique()}")
    return "\n".join(lines)


def cmd_build(args):
    cfg = load_config()
    target, external = build_and_write(cfg, _DATA)
    print(f"Манифесты записаны в {_DATA}")
    print(f"  manifest.csv          : {len(target)} target-строк")
    print(f"  manifest_external.csv : {len(external)} GCN-строк")
    print(_summary(target, external))


def cmd_validate(args):
    cfg = load_config()
    target, external = read_manifests(_DATA)
    warnings = run_gates(target, external, cfg)
    print("✓ Все 12 гейтов пройдены.")
    for w in warnings:
        print("  ⚠", w)
    write_eda(target, external, cfg, _REPORTS)
    print(f"EDA-отчёт: {_REPORTS / 'eda.md'}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m triton_data.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="собрать манифесты").set_defaults(func=cmd_build)
    sub.add_parser("validate", help="жёсткие гейты + EDA").set_defaults(func=cmd_validate)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
