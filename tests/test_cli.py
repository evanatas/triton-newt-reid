"""Smoke-тесты CLI. Импорт модуля на этапе сбора ловит ошибки уровня модуля
(например NameError в аннотациях из-за забытого импорта)."""
import pytest

import triton_data.cli as cli


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        cli.main([])


def test_cli_rejects_unknown_subcommand():
    with pytest.raises(SystemExit):
        cli.main(["definitely-not-a-command"])
