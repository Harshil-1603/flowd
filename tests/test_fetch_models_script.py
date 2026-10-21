"""Exercise optional cache warming with fake Moonshine modules, never weights."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def fetcher(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    script = repo / "scripts/fetch_models.sh"
    shutil.copy2(REPO / "scripts/fetch_models.sh", script)
    models = tmp_path / "data/flowd/models"
    models.mkdir(parents=True)
    (models / "sotto-cleanup-lfm25-350m-q4_k_m.gguf").write_bytes(b"fake cleanup")
    for directory in (
        "parakeet-tdt-0.6b-v2-int8",
        "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25",
    ):
        (models / directory).mkdir()
        for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
            (models / directory / name).write_bytes(b"fake model")
    modules = tmp_path / "modules"
    package = modules / "moonshine_voice"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from types import SimpleNamespace\n"
        "def string_to_model_arch(name):\n"
        "    return SimpleNamespace(name=name)\n"
    )
    (package / "download.py").write_text(
        "import os\n"
        "def get_model_for_language(language, arch):\n"
        "    assert language == 'en'\n"
        "    with open(os.environ['FLOWD_TEST_MOONSHINE_CALLS'], 'a') as f:\n"
        "        f.write(arch.name + '\\n')\n"
        "    return '/fake/cache/' + arch.name, arch\n"
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    uv = bindir / "uv"
    uv.write_text(
        '#!/bin/sh\n[ "$1" = run ] && [ "$2" = python ] || exit 90\n'
        f'shift 2\nexec {shlex.quote(sys.executable)} "$@"\n'
    )
    uv.chmod(0o755)
    curl = bindir / "curl"
    curl.write_text("#!/bin/sh\necho 'network forbidden in this test' >&2\nexit 91\n")
    curl.chmod(0o755)
    calls = tmp_path / "cache-calls"
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(modules) + os.pathsep + str(REPO),
        "FLOWD_TEST_MOONSHINE_CALLS": str(calls),
    }
    return script, env, calls


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], ["small-streaming"]),
        (["--with-moonshine-tiny"], ["small-streaming", "tiny-streaming"]),
        (["--with-moonshine-medium"], ["small-streaming", "medium-streaming"]),
        (
            ["--with-moonshine-tiny", "--with-moonshine-medium"],
            ["small-streaming", "tiny-streaming", "medium-streaming"],
        ),
        (
            ["--with-moonshine-tiny", "--with-moonshine-tiny"],
            ["small-streaming", "tiny-streaming"],
        ),
        (["--with-nemotron", "--with-moonshine-medium"], ["small-streaming", "medium-streaming"]),
    ],
)
def test_only_requested_moonshine_previews_are_cached(
    fetcher: tuple[Path, dict[str, str], Path], flags: list[str], expected: list[str]
) -> None:
    script, env, calls = fetcher
    result = subprocess.run(
        ["bash", str(script), *flags], env=env, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines() == expected
    lock = json.loads((script.parent.parent / "models.lock").read_text())
    # Moonshine's downloader still owns its cache verification, not models.lock.
    assert not any("moonshine" in entry["name"] for entry in lock["models"])


def test_help_explains_the_optional_previews_without_fetching(
    fetcher: tuple[Path, dict[str, str], Path],
) -> None:
    script, env, calls = fetcher
    result = subprocess.run(
        ["bash", str(script), "--help"], env=env, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0
    assert "--with-moonshine-tiny" in result.stdout
    assert "--with-moonshine-medium" in result.stdout
    assert not calls.exists()
