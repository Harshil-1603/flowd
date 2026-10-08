import numpy as np

from flowd.nemotron import FINISH_PADDING_S, NemotronStreamingEngine
from flowd.stt import Committed, Partial

RATE = 16000


class FakeRecognizer:
    """sherpa-onnx's OnlineRecognizer surface: a stream is ready once per 0.56 s of audio."""

    def __init__(self) -> None:
        self.streams: list[FakeStream] = []

    def create_stream(self) -> "FakeStream":
        stream = FakeStream()
        self.streams.append(stream)
        return stream

    def is_ready(self, stream: "FakeStream") -> bool:
        return stream.undecoded >= int(0.56 * RATE) or (stream.finished and stream.undecoded > 0)

    def decode_stream(self, stream: "FakeStream") -> None:
        take = min(stream.undecoded, int(0.56 * RATE))
        stream.undecoded -= take
        stream.words += 1

    def get_result(self, stream: "FakeStream") -> str:
        return " ".join(f"w{i}" for i in range(1, stream.words + 1))


class FakeStream:
    def __init__(self) -> None:
        self.undecoded = 0
        self.fed = 0
        self.words = 0
        self.finished = False

    def accept_waveform(self, rate: int, pcm: np.ndarray) -> None:
        assert rate == RATE and pcm.dtype == np.float32
        self.undecoded += pcm.size
        self.fed += pcm.size

    def input_finished(self) -> None:
        self.finished = True


def block(seconds: float = 0.1) -> np.ndarray:
    return np.full(int(seconds * RATE), 0.05, dtype=np.float32)


def test_feed_decodes_whole_chunks_and_reports_the_text_when_it_changes() -> None:
    engine = NemotronStreamingEngine(FakeRecognizer(), RATE)
    events = [engine.feed(block()) for _ in range(12)]
    # 0.56 s chunks: the first words appear at block 6 and 12, nothing in between.
    assert [e for e in events[:5]] == [[]] * 5
    assert events[5] == [Partial("w1")]
    assert events[6:11] == [[]] * 5
    assert events[11] == [Partial("w1 w2")]


def test_empty_blocks_are_ignored() -> None:
    recognizer = FakeRecognizer()
    assert NemotronStreamingEngine(recognizer, RATE).feed(np.empty(0, np.float32)) == []
    assert recognizer.streams == []


def test_finalize_pads_the_tail_signals_the_end_and_commits_what_was_heard() -> None:
    recognizer = FakeRecognizer()
    engine = NemotronStreamingEngine(recognizer, RATE)
    for _ in range(3):
        engine.feed(block())
    events = engine.finalize()
    stream = recognizer.streams[0]
    assert stream.finished
    assert stream.fed == int(0.3 * RATE) + int(FINISH_PADDING_S * RATE)
    assert events == [Committed("w1 w2")]


def test_finalize_without_audio_is_empty() -> None:
    assert NemotronStreamingEngine(FakeRecognizer(), RATE).finalize() == []


def test_each_session_gets_a_new_stream() -> None:
    recognizer = FakeRecognizer()
    engine = NemotronStreamingEngine(recognizer, RATE)
    engine.feed(block(1.0))
    engine.finalize()
    assert engine.feed(block(0.2)) == []
    assert len(recognizer.streams) == 2


def test_reset_drops_the_session_without_committing() -> None:
    recognizer = FakeRecognizer()
    engine = NemotronStreamingEngine(recognizer, RATE)
    engine.feed(block(1.0))
    engine.reset()
    assert engine.finalize() == []
    engine.feed(block(1.0))
    assert len(recognizer.streams) == 2
