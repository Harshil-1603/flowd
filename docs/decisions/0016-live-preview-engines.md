# 0016: The live preview is Parakeet re-decode, with Nemotron as an option

**Status:** accepted (2026-10-08)

## Context

ADR 0011 made Parakeet the engine that commits text and kept Moonshine small
for the live preview. On the owner's voice that preview was poor and expensive:

- Moonshine small previewed at about 74% WER against what Parakeet commits, so
  the words on screen were mostly not the words that got typed.
- It cost about 4 s of CPU per second of audio (377 s of CPU for the 70 s
  recording below) and its `feed()` stalled the audio loop for up to 1 s.
- It held a second model in memory (about 190 MB more than Parakeet alone for
  the medium variant).

Two replacements were measured with `--replay` on the same 70.5 s recording
(`20260928-015238_43a00c835267.wav`, seven pieces, 98 words), on a 6-core
Ryzen 5 5600H with a load average of 2 to 4. CPU is process CPU time divided by
audio seconds, so it includes start-up.

| Preview | Release to final | First preview | Preview vs commit | CPU per audio s | Peak RSS |
|---|---|---|---|---|---|
| Parakeet re-decode | 140 to 183 ms | 0.7 to 0.8 s | 8% words differ | 2.3 s | 1.2 GB |
| Nemotron streaming 560 ms | 98 to 113 ms | 2.9 to 3.1 s | 18% words differ | 1.5 s | 1.9 GB |
| Moonshine small (before) | 187 ms | 2.2 to 2.5 s | 32% words differ | 5.4 s | 1.7 GB |

The committed text was identical in all three runs (Parakeet commits it).
Offline benchmarks of the preview engines on the same voice gave 5.2 to 5.7%
WER for Parakeet re-decode and 12.9% for Nemotron.

## Decision

- **`[stt] model = "parakeet-redecode"` is the default.** The preview re-decodes
  the open, uncut piece with the already loaded Parakeet model every 0.7 s.
  Closed pieces show their committed text. No second model is loaded.
- **`nemotron-streaming-en-0.6b-560ms` is opt-in.** It runs
  `sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25` through
  sherpa-onnx's `OnlineRecognizer` (about 660 MB on disk, 4 threads). Parakeet
  still commits. It is lighter on CPU and has no per-tick re-decode, but costs
  about 940 MB of memory and shows its first words later.
- **Moonshine small and medium stay selectable**, labelled legacy. A config
  with `final_model = ""` and no `model` key keeps meaning "Moonshine only".
- **A preview that needs the final model is refused without one.** With
  `final_model = ""`, `parakeet-redecode` and Nemotron fail at config load, with
  a message naming a Moonshine model as the way out. Nemotron only previews and
  re-decode decodes with the final model, so nothing would commit.
- **Scheduling rules for re-decode:**
  - One worker thread decodes everything. sherpa-onnx does not document
    concurrent `decode_stream` calls on one `OfflineRecognizer` as safe, so a
    model instance never decodes two streams at once.
  - A preview is never queued. A tick is skipped when the worker is busy, when
    the interval has not elapsed, or when a piece is waiting.
  - Piece commits have priority. A queued preview is cancelled on a cut,
    finalize or reset, and a finished preview from an older epoch is dropped.
    A preview already running cannot be stopped, so a commit can wait up to one
    decode (about 0.4 s) behind it.
- **Selection from the settings page.** `stt.model` is restart-only. Saving
  shows the existing "takes effect after flowd restarts" banner, and its
  Restart flowd button (`POST /api/restart`) applies it.
- **Missing files do not crash the daemon.** The daemon logs an error naming
  the fetch command (`scripts/fetch_models.sh`, or `--with-nemotron`) and the
  status endpoint reports `stt_missing`, which the page shows as "Unavailable:
  model files missing".
- **Nemotron is fetched only on request.** `scripts/fetch_models.sh
  --with-nemotron` downloads the release tarball and installs four files. They
  are pinned in `models.lock` (SHA-256, size, URL, licence "NVIDIA Open Model
  License"). Pins are verified only while Nemotron is selected, so the default
  install does not need it. Re-pinning keeps optional pins that were not
  fetched.

## Consequences

- The default preview is more accurate (about 8% against the commit instead of
  32%) and cheaper than Moonshine small, and the daemon stays at about 1,140 MB
  anon (ADR 0011). The cost is steady CPU while dictating: about 2.3 s per
  audio second on this machine, which a slower machine may not keep up with.
  Under load the preview updates less often (partial gaps reached 3 s) but
  commits are not delayed.
- Nemotron adds about 940 MB, so the idle budget of 1,600 MB is exceeded by
  design when it is selected. The settings page raises its displayed budget by
  950 MB then; `scripts/idle_check.py` does not, and will report over budget.
- Nemotron's first words arrive about 3 s in, and its preview differs from the
  committed text more often than re-decode does. Its advantage is CPU and a
  slightly faster release to final.
- Moonshine's `feed()` stalls are gone from the default path (p95 0.3 ms). With
  Nemotron, `feed()` runs the recogniser inline and reached 177 ms.
- Numbers come from one speaker, one recording and one machine under varying
  load. Treat the ratios as indicative.
- The Nemotron licence is not CC-BY-4.0 like Parakeet; users who fetch it accept
  the NVIDIA Open Model License.
