"""Live preview from NVIDIA's Nemotron streaming transducer (ADR 0016).

Opt-in alternative to re-decoding with Parakeet: it streams natively, so it
costs about 1.1x real time of CPU instead of 2.5x, at the price of a second
~660 MB model (about 940 MB resident) and a higher word error rate (12.9% on
the reference recording against 5.2-5.7% for the re-decode). `HybridSttEngine`
uses it for the overlay only; Parakeet still commits the text.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from flowd.stt import Committed, Event, Partial

log = logging.getLogger(__name__)

#: Same split as Parakeet's (hybrid.FINAL_THREADS): the machine's other cores
#: are shared with llama-server and the overlay.
NEMOTRON_THREADS = 4
#: The model needs trailing audio after the last word to emit it. 0.8 s is
#: what the benchmark used; the chunk size is 560 ms.
FINISH_PADDING_S = 0.8


class NemotronStreamingEngine:
    """One online stream per dictation, built on the first block."""

    def __init__(self, recognizer: Any, sample_rate: int = 16000) -> None:
        self._recognizer = recognizer
        self._rate = sample_rate
        self._stream: Any | None = None
        self._last = ""

    def feed(self, pcm: np.ndarray) -> list[Event]:
        if pcm.size == 0:
            return []
        stream = self._ensure_stream()
        stream.accept_waveform(self._rate, np.asarray(pcm, dtype=np.float32))
        self._decode(stream)
        text = self._text(stream)
        if text == self._last:
            return []
        self._last = text
        return [Partial(text)] if text else []

    def finalize(self) -> list[Event]:
        stream = self._stream
        if stream is None:
            return []
        stream.accept_waveform(self._rate, np.zeros(int(self._rate * FINISH_PADDING_S), np.float32))
        stream.input_finished()
        self._decode(stream)
        text = self._text(stream)
        self._discard()
        return [Committed(text)] if text else []

    def reset(self) -> None:
        self._discard()

    def _ensure_stream(self) -> Any:
        if self._stream is None:
            self._stream = self._recognizer.create_stream()
        return self._stream

    def _decode(self, stream: Any) -> None:
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)

    def _text(self, stream: Any) -> str:
        return str(self._recognizer.get_result(stream)).strip()

    def _discard(self) -> None:
        self._stream = None
        self._last = ""


def load_nemotron(model_dir: Path, sample_rate: int = 16000) -> NemotronStreamingEngine:
    """Load Nemotron once at daemon start. `scripts/fetch_models.sh
    --with-nemotron` puts the files in `model_dir` and pins them in models.lock."""
    import sherpa_onnx

    recognizer: Any = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(model_dir / "tokens.txt"),
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        joiner=str(model_dir / "joiner.int8.onnx"),
        num_threads=NEMOTRON_THREADS,
        sample_rate=sample_rate,
        decoding_method="greedy_search",
    )
    log.info("loaded preview STT model from %s", model_dir)
    return NemotronStreamingEngine(recognizer, sample_rate)
