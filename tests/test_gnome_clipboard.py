"""GNOME can paste immediately through ydotool without wtype's protocol."""

from flowd.inject.clipboard import ClipboardBackend
from tests.test_inject import Recorder, clipboard_writes


def backend(rec: Recorder, *, ready: bool = True) -> ClipboardBackend:
    return ClipboardBackend(
        runner=rec.run,
        sleep=lambda _: None,
        wayland=True,
        gnome=True,
        have=lambda tool: tool in {"wl-copy", "wl-paste", "ydotool"},
        ydotool=lambda: ready,
    )


def test_gnome_clipboard_uses_ctrl_v_and_restores_the_original() -> None:
    rec = Recorder()
    paste = backend(rec)
    assert paste.available()
    paste.inject("A complete sentence.", is_terminal=False)
    assert ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"] in rec.calls
    assert not any(call[0] == "wtype" for call in rec.calls)
    assert clipboard_writes(rec, wayland=True) == [b"A complete sentence.", b"previous clipboard"]


def test_gnome_terminal_paste_keeps_shift() -> None:
    rec = Recorder()
    backend(rec).inject("echo hello", is_terminal=True)
    assert ["ydotool", "key", "29:1", "42:1", "47:1", "47:0", "42:0", "29:0"] in rec.calls


def test_gnome_clipboard_declines_before_touching_clipboard_without_the_daemon() -> None:
    rec = Recorder()
    paste = backend(rec, ready=False)
    assert not paste.available()
    assert "ydotoold" in (paste.unavailable_reason() or "")
    assert not rec.calls
