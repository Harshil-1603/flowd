"""scripts/install.sh --dry-run: the units it would write for this machine."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "install.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def dry_run(tmp_path: Path, *, script: Path = SCRIPT, **env: str) -> str:
    """Run the installer with a throwaway HOME, so nothing real is touched."""
    home = tmp_path / "home"
    home.mkdir()
    environment = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        **env,
    }
    result = subprocess.run(
        ["bash", str(script), "--dry-run"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout


def test_the_daemon_unit_runs_flowd_from_this_checkout(tmp_path: Path) -> None:
    """The shipped ExecStart is a placeholder; the installer points it here."""
    out = dry_run(tmp_path)
    assert f"ExecStart={REPO}/.venv/bin/flowd" in out


def test_the_model_path_follows_xdg_data_home(tmp_path: Path) -> None:
    """systemd cannot read XDG_DATA_HOME, so the installer writes the path out."""
    out = dry_run(tmp_path)
    model = tmp_path / "data" / "flowd" / "models" / "sotto-cleanup-lfm25-350m-q4_k_m.gguf"
    assert f"Environment=FLOWD_MODEL={model}" in out


def test_llama_server_threads_can_be_overridden(tmp_path: Path) -> None:
    out = dry_run(tmp_path, FLOWD_LLM_THREADS="3")
    assert "llama-server threads: 3" in out
    assert "--port 8177 -t 3" in out


def test_a_dry_run_changes_nothing(tmp_path: Path) -> None:
    dry_run(tmp_path)
    assert not (tmp_path / "config" / "systemd").exists()
    assert not (tmp_path / "home" / ".local" / "bin").exists()


def test_an_unknown_option_is_refused(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--bogus"], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 2
    assert "unknown option" in result.stderr


def test_the_ui_build_is_planned_or_explained(tmp_path: Path) -> None:
    out = dry_run(tmp_path)
    assert "would run: cmake -S" in out or "flowd-ui not built" in out


def test_skipping_the_ui_build_is_not_fatal(tmp_path: Path) -> None:
    out = dry_run(tmp_path, FLOWD_UI_BUILD="0")
    assert "flowd-ui not built" in out
    assert "Writing systemd units" in out


def fake_path(tmp_path: Path, *tools: str) -> str:
    """Only expose the requested tools; never depend on the host's desktop."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("bash", "dirname", "grep", "sort", "wc", "sed", "cat"):
        source = shutil.which(name)
        assert source is not None
        (bindir / name).symlink_to(source)
    commands = {
        "ldconfig": "printf 'libportaudio.so.2\\n'",
        "lscpu": "printf '# Core\\n0\\n1\\n2\\n3\\n'",
        **{name: "exit 0" for name in tools},
    }
    for name, body in commands.items():
        target = bindir / name
        target.write_text("#!/bin/sh\n" + body + "\n")
        target.chmod(0o755)
    return str(bindir)


def test_missing_ubuntu_packages_use_apt_names(tmp_path: Path) -> None:
    """Ubuntu must not be told to install Arch's portaudio with pacman."""
    path = fake_path(tmp_path, "uv", "curl", "pw-cli", "llama-server", "apt-get")
    # Model a machine with no PortAudio library either.
    (Path(path) / "ldconfig").write_text("#!/bin/sh\nexit 0\n")
    out = dry_run(tmp_path, PATH=path, WAYLAND_DISPLAY="wayland-0")
    assert "sudo apt install libportaudio2 wl-clipboard wtype" in out
    assert "pacman" not in out


@pytest.mark.parametrize("desktop", ["GNOME", "ubuntu:GNOME", "Ubuntu:gNoMe"])
def test_gnome_wayland_requires_ydotool_instead_of_wtype(tmp_path: Path, desktop: str) -> None:
    """Mutter rejects wtype's virtual keyboard protocol even when installed."""
    path = fake_path(tmp_path, "uv", "curl", "pw-cli", "llama-server", "wtype", "apt-get")
    out = dry_run(tmp_path, PATH=path, WAYLAND_DISPLAY="wayland-0", XDG_CURRENT_DESKTOP=desktop)
    assert "missing: ydotool" in out
    assert "sudo apt install ydotool" in out


def test_gnome_wayland_can_install_with_only_ydotool(tmp_path: Path) -> None:
    path = fake_path(tmp_path, "uv", "curl", "pw-cli", "llama-server", "ydotool", "apt-get")
    out = dry_run(tmp_path, PATH=path, WAYLAND_DISPLAY="wayland-0", XDG_CURRENT_DESKTOP="GNOME")
    assert "all present" in out
    assert "flowd-ui not built (GNOME Wayland" in out
    assert "uinput" in out


def test_ubuntu_does_not_offer_nonportable_uv_or_llama_packages(tmp_path: Path) -> None:
    """Older Ubuntu releases do not package the cleanup server or uv."""
    path = fake_path(tmp_path, "curl", "pw-cli", "xclip", "xdotool", "apt-get")
    out = dry_run(tmp_path, PATH=path)
    assert "Install uv:" in out
    assert "Install llama-server:" in out
    assert "sudo apt install uv" not in out
    assert "pacman" not in out


def test_ubuntu_ui_hint_uses_development_package_names(tmp_path: Path) -> None:
    path = fake_path(
        tmp_path, "uv", "curl", "pw-cli", "llama-server", "xclip", "xdotool", "apt-get"
    )
    out = dry_run(tmp_path, PATH=path)
    assert "sudo apt install cmake g++ pkg-config libgtkmm-4.0-dev libgtk4-layer-shell-dev" in out


def test_new_virtualenv_requests_a_supported_python(tmp_path: Path) -> None:
    """Ubuntu 22.04's system Python 3.10 does not satisfy requires-python."""
    repo = tmp_path / "repo"
    for directory in ("scripts", "systemd"):
        shutil.copytree(REPO / directory, repo / directory)
    out = dry_run(tmp_path, script=repo / "scripts" / "install.sh")
    assert "would run: uv venv --python >=3.11" in out
