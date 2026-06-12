"""Smoke-тесты CLI Блока 2: импорт ловит ошибки уровня модуля; парсер регистрирует команды."""
import pytest

import triton_crop.cli as cli


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        cli.main([])


def test_cli_rejects_unknown_subcommand():
    with pytest.raises(SystemExit):
        cli.main(["definitely-not-a-command"])


def test_cli_registers_pipeline_commands():
    p = cli.build_parser()
    expect = {
        "bootstrap": "cmd_bootstrap", "correct": "cmd_correct", "export": "cmd_export",
        "train-seg": "cmd_train_seg", "train-pose": "cmd_train_pose",
        "validate-crops": "cmd_validate_crops",
    }
    for cmd, fn in expect.items():
        assert p.parse_args([cmd]).func.__name__ == fn


def test_cli_crop_and_ab_registered():
    p = cli.build_parser()
    assert p.parse_args(["crop"]).func.__name__ == "cmd_crop"
    assert p.parse_args(["ab"]).func.__name__ == "cmd_ab"


def test_cli_crop_unseal_flag():
    assert cli.build_parser().parse_args(["crop"]).unseal is False             # по умолчанию запечатано
    assert cli.build_parser().parse_args(["crop", "--stages", "test", "--unseal"]).unseal is True


def test_cmd_crop_refuses_sealed_without_unseal():
    # C9: test/open_test нельзя кропать без явного --unseal — гейт срабатывает ДО загрузки моделей
    a = cli.build_parser().parse_args(["crop", "--stages", "test"])
    with pytest.raises(ValueError, match="C9|запечатан|unseal"):
        cli.cmd_crop(a)
    a2 = cli.build_parser().parse_args(["crop", "--stages", "dev,open_test"])
    with pytest.raises(ValueError, match="C9|запечатан|unseal"):
        cli.cmd_crop(a2)


def test_train_entrypoints_importable():
    from triton_crop.train import train_pose, train_seg
    assert callable(train_seg) and callable(train_pose)


def test_cli_registers_embed_commands():
    p = cli.build_parser()
    expect = {
        "embed-build": "cmd_embed_build", "embed-train": "cmd_embed_train",
        "embed-ab": "cmd_embed_ab", "embed-eval-openset": "cmd_embed_eval_openset",
    }
    for cmd, fn in expect.items():
        assert p.parse_args([cmd]).func.__name__ == fn


def test_cli_embed_test_registered():
    a = cli.build_parser().parse_args(["embed-test"])
    assert a.func.__name__ == "cmd_embed_test"
    assert a.unseal is False and a.scope == "kpi_core"     # по умолчанию запечатано


def test_cmd_embed_test_refuses_without_unseal():
    # ФИНАЛ sealed-test: без --unseal — ValueError ДО любых тяжёлых импортов/моделей (гейт C9)
    a = cli.build_parser().parse_args(["embed-test"])
    with pytest.raises(ValueError, match="C9|запечат|unseal"):
        cli.cmd_embed_test(a)


def test_cli_embed_build_defaults():
    a = cli.build_parser().parse_args(["embed-build"])
    assert a.stage == "train" and a.scope == "kpi_core"


def test_cli_embed_pack_registered():
    a = cli.build_parser().parse_args(["embed-pack"])
    assert a.func.__name__ == "cmd_embed_pack"
    assert a.records.endswith("oof_records.csv")
    assert a.dry_run is False and a.include_fallback_png is False


def test_cli_registers_spot_commands():
    p = cli.build_parser()
    assert p.parse_args(["detect-spots"]).func.__name__ == "cmd_detect_spots"
    assert p.parse_args(["spot-ab"]).func.__name__ == "cmd_spot_ab"
    assert p.parse_args(["match", "--probe", "m0"]).func.__name__ == "cmd_match"
    assert p.parse_args(["spot-validate"]).func.__name__ == "cmd_spot_validate"   # S-гейты в CLI


def test_cli_spot_ab_defaults():
    a = cli.build_parser().parse_args(["spot-ab"])
    assert a.scope == "kpi_core" and a.embed_variant == "unroll_ribbon"
    assert "belly_oriented" in a.surfaces and "guided" in a.matchers   # канонический матчер = guided
    assert a.sweep_iters == "120,500,2000"                              # sensitivity-sweep (контроль iteration-starvation)


def test_cli_detect_spots_defaults():
    a = cli.build_parser().parse_args(["detect-spots"])
    assert a.variant == "belly_oriented" and a.method == "deviation"


def test_cli_match_defaults_canonical():
    # страж: дефолты match = канонический конфиг (deviation+guided, как detect-spots/spot.yaml) —
    # оверлеи по умолчанию строятся на доказательной базе, а не на историческом darkness
    a = cli.build_parser().parse_args(["match", "--probe", "m"])
    assert a.method == "deviation" and a.matcher == "guided"


def test_cli_hybrid_registered():
    a = cli.build_parser().parse_args(["hybrid"])
    assert a.func.__name__ == "cmd_hybrid"
    assert a.surface == "belly_oriented" and a.embed_variant == "unroll_ribbon"
    assert a.reuse_sims is False


# ── sealed/provenance-гейты CLI ──

def test_cli_sealed_guard_flags():
    p = cli.build_parser()
    assert p.parse_args(["validate-crops"]).unsealed is False
    assert p.parse_args(["validate-crops", "--unsealed"]).unsealed is True
    assert p.parse_args(["spot-validate"]).unsealed is False
    assert p.parse_args(["detect-spots"]).unseal is False
    assert p.parse_args(["match", "--probe", "m"]).unseal is False
    assert p.parse_args(["embed-test"]).force_rerun_sealed is False
    assert p.parse_args(["embed-verify"]).allow_missing_metrics is False


def test_cli_embed_build_rejects_sealed_stage():
    # choices=[train,gallery] → argparse отвергает test/open_test (нельзя строить train-артефакты из sealed)
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["embed-build", "--stage", "test"])


def test_cmd_detect_spots_refuses_sealed_without_unseal():
    a = cli.build_parser().parse_args(["detect-spots", "--stages", "test"])
    with pytest.raises(ValueError, match="C9|запечат|unseal"):
        cli.cmd_detect_spots(a)


def test_cmd_match_refuses_sealed_gallery_without_unseal():
    a = cli.build_parser().parse_args(["match", "--probe", "m", "--gallery-stage", "open_test"])
    with pytest.raises(ValueError, match="C9|запечат|unseal"):
        cli.cmd_match(a)


def test_cmd_embed_test_rerun_guard(tmp_path, monkeypatch):
    # повторный --unseal при УЖЕ существующем ab_test_headline.json → ValueError (защита финальных чисел)
    monkeypatch.setattr(cli, "_ART", tmp_path)
    (tmp_path / "ab_test_headline.json").write_text("{}", encoding="utf-8")
    a = cli.build_parser().parse_args(["embed-test", "--unseal"])
    with pytest.raises(ValueError, match="уже существует|финальн|force"):
        cli.cmd_embed_test(a)


def test_cmd_embed_test_rerun_guard_heldout_npy(tmp_path, monkeypatch):
    # вскрытие фиксируют И heldout-npy: повторный --unseal при живых artifacts/embed/heldout/*.npy,
    # даже если ab_test_headline.json удалён → ValueError (не перезатирать heldout молча)
    import numpy as np
    monkeypatch.setattr(cli, "_ART", tmp_path)
    ho = tmp_path / "embed" / "heldout"
    ho.mkdir(parents=True)
    np.save(ho / "belly_oriented.npy", np.zeros(3, np.float32))
    a = cli.build_parser().parse_args(["embed-test", "--unseal"])
    with pytest.raises(ValueError, match="уже существуют|force"):
        cli.cmd_embed_test(a)


def test_to_jsonable_numpy_types():
    # _to_jsonable: numpy-скаляры/массивы и Path → нативные JSON-типы (сериализация ab-словарей)
    from pathlib import Path

    import numpy as np
    assert cli._to_jsonable(np.int64(3)) == 3 and isinstance(cli._to_jsonable(np.int64(3)), int)
    assert cli._to_jsonable(np.float32(1.5)) == 1.5 and isinstance(cli._to_jsonable(np.float32(1.5)), float)
    assert cli._to_jsonable(np.bool_(True)) is True
    assert cli._to_jsonable(np.array([1, 2])) == [1, 2]
    assert cli._to_jsonable(Path("x")) == "x"


def test_cmd_embed_verify_hard_fails_on_missing_metrics(tmp_path):
    # отсутствие файла метрик без --allow-missing-metrics → SystemExit (гейт «метрики↔npy» иначе вхолостую)
    a = cli.build_parser().parse_args(["embed-verify", "--metrics", str(tmp_path / "нет.json")])
    with pytest.raises(SystemExit):
        cli.cmd_embed_verify(a)
