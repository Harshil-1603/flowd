# 0017: Show the GNOME preview through XWayland

## Context

GNOME Wayland does not implement layer-shell or wtype's virtual-keyboard
protocol. The original UI disabled itself even when an XWayland display was
available, leaving Ubuntu users without recording feedback or a live preview.

On Ubuntu 26.04, GNOME Shell 50.1 exposes both `WAYLAND_DISPLAY=wayland-0`
and `DISPLAY=:0`. The existing X11 UI already creates override-redirect
windows, sets `WM_HINTS input=False`, and reads both properties back before
showing a surface. These checks can also protect the XWayland path.

## Decision

On GNOME Wayland with `DISPLAY` available, use the existing X11 backend and
pin only the UI child's GDK backend to X11. The dictation daemon stays in its
Wayland environment and continues to use its configured injection backend.
Without an X display, keep the UI disabled. Never substitute an ordinary
Wayland toplevel for a layer surface.

Keep the per-surface override-redirect and non-input read-back checks. A
failure leaves the surface hidden. Verify focus on the live GNOME desktop
before treating this backend as working.

## Consequences

The installer may build the preview on GNOME when XWayland is present.
Native Wayland application identity remains unavailable without a compositor
integration; XWayland preview support does not change that limitation.

The backend regression test fails on the previous implementation for `GNOME`,
`ubuntu:GNOME`, and mixed-case desktop tokens. The runtime focus verification
and its measured results are recorded in the pull request.
