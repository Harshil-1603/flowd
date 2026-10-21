import threading
from collections.abc import Callable
from concurrent.futures import Future

import numpy as np
import pytest

from flowd.hybrid import HybridSttEngine
from flowd.stt import Committed, Event, Partial

RATE = 16000
BLOCK = RATE // 10  # 100 ms


def speech(seconds: float) -> list[np.ndarray]:
    """Loud blocks, about -23 dBFS."""
    return [np.full(BLOCK, 0.07, dtype=np.float32) for _ in range(round(seconds * 10))]


def quiet(seconds: float) -> list[np.ndarray]:
    """Near-silent blocks, about -60 dBFS."""
    return [np.full(BLOCK, 0.001, dtype=np.float32) for _ in range(round(seconds * 10))]


class Live:
    """Stands in for Moonshine: one scripted word per fed block, committed on request."""

    def __init__(self) -> None:
        self.fed = 0
        self.words: list[str] = []
        self.finalized = False
        self.was_reset = False
        self.keyterms: list[str] | None = None

    def feed(self, pcm: np.ndarray) -> list[Event]:
        self.fed += 1
        self.words.append(f"w{self.fed}")
        # Moonshine commits some of its own text; the hybrid must not pass it on.
        if self.fed % 5 == 0:
            return [Committed(" ".join(self.words[-5:])), Partial("")]
        return [Partial(" ".join(self.words[-(self.fed % 5) :]))]

    def finalize(self) -> list[Event]:
        self.finalized = True
        tail = self.words[len(self.words) - self.fed % 5 :] if self.fed % 5 else []
        return [Committed(" ".join(tail))] if tail else []

    def reset(self) -> None:
        self.was_reset = True

    def set_keyterms(self, terms: list[str]) -> None:
        self.keyterms = list(terms)


class Now:
    """An executor that runs work at submit, so tests are deterministic."""

    def submit(self, fn: Callable[..., str], *args: object) -> "Future[str]":
        future: Future[str] = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


def seconds_of(pcm: np.ndarray) -> str:
    return f"piece {pcm.size / RATE:.1f}s"


def engine(
    transcribe: Callable[[np.ndarray], str] = seconds_of, live: Live | None = None, **kw: object
) -> tuple[HybridSttEngine, Live]:
    live = live or Live()
    return HybridSttEngine(live, transcribe, sample_rate=RATE, executor=Now(), **kw), live  # type: ignore[arg-type]


def feed_all(e: HybridSttEngine, blocks: list[np.ndarray]) -> list[Event]:
    out: list[Event] = []
    for block in blocks:
        out.extend(e.feed(block))
    return out


def commits(events: list[Event]) -> list[str]:
    return [ev.text for ev in events if isinstance(ev, Committed)]


def test_a_short_dictation_is_transcribed_whole_at_finalize() -> None:
    e, live = engine()
    during = feed_all(e, speech(3) + quiet(0.5) + speech(2))
    assert commits(during) == []
    assert commits(e.finalize()) == ["piece 5.5s"]
    # Reset, not finalized: its tail is not wanted, and finalizing it would
    # make the user wait for Moonshine to decode its backlog.
    assert live.was_reset
    assert not live.finalized


def test_moonshine_commits_never_reach_the_session() -> None:
    e, _ = engine(transcribe=lambda pcm: "accurate")
    events = feed_all(e, speech(2)) + e.finalize()
    assert commits(events) == ["accurate"]


def test_the_preview_shows_the_live_transcript() -> None:
    e, _ = engine()
    events = feed_all(e, speech(0.7))
    partials = [ev.text for ev in events if isinstance(ev, Partial)]
    assert partials[-1] == "w1 w2 w3 w4 w5 w6 w7"


def test_a_pause_after_enough_speech_commits_that_piece_while_recording() -> None:
    e, _ = engine()
    during = feed_all(e, speech(9) + quiet(1) + speech(2))
    assert commits(during) == ["piece 9.6s"]
    assert commits(e.finalize()) == ["piece 2.4s"]


def test_a_pause_too_early_in_a_piece_does_not_cut_it() -> None:
    e, _ = engine()
    during = feed_all(e, speech(3) + quiet(1) + speech(3))
    assert commits(during) == []
    assert commits(e.finalize()) == ["piece 7.0s"]


def test_unbroken_speech_is_cut_at_the_quietest_moment_before_the_limit() -> None:
    e, _ = engine()
    dip = [np.full(BLOCK, 0.02, dtype=np.float32)]
    during = feed_all(e, speech(18) + dip + speech(5))
    # Cut after the dip at 18.1 s, not mid-word at 20 s.
    assert commits(during) == ["piece 18.1s"]
    assert commits(e.finalize()) == ["piece 5.0s"]


def test_the_preview_drops_words_already_committed() -> None:
    e, _ = engine()
    events = feed_all(e, speech(9) + quiet(1) + speech(0.3))
    partials = [ev.text for ev in events if isinstance(ev, Partial)]
    # The cut came at block 96, so the live words from 97 on are the next piece.
    assert partials[-1] == "w97 w98 w99 w100 w101 w102 w103"


def test_silence_commits_nothing() -> None:
    e, _ = engine(transcribe=lambda pcm: "")
    feed_all(e, quiet(12))
    assert commits(e.finalize()) == []


def test_finalize_without_audio_is_empty() -> None:
    e, _ = engine()
    assert e.finalize() == []


def test_a_failed_piece_falls_back_to_the_live_text(caplog: pytest.LogCaptureFixture) -> None:
    def broken(pcm: np.ndarray) -> str:
        raise RuntimeError("decoder crashed")

    e, _ = engine(transcribe=broken)
    feed_all(e, speech(0.4))
    with caplog.at_level("WARNING"):
        assert commits(e.finalize()) == ["w1 w2 w3 w4"]
    assert any("decoder crashed" in r.getMessage() for r in caplog.records)


def test_reset_discards_the_session_and_its_pieces() -> None:
    e, live = engine()
    feed_all(e, speech(9) + quiet(0.5))
    e.reset()
    assert live.was_reset
    assert e.finalize() == []
    feed_all(e, speech(1))
    assert commits(e.finalize()) == ["piece 1.0s"]


def test_a_new_session_starts_clean_after_finalize() -> None:
    e, _ = engine()
    feed_all(e, speech(2))
    e.finalize()
    feed_all(e, speech(1))
    assert commits(e.finalize()) == ["piece 1.0s"]


def test_keyterms_go_to_the_live_model() -> None:
    e, live = engine()
    e.set_keyterms(["flowd"])
    assert live.keyterms == ["flowd"]


def test_pieces_decode_off_the_feeding_thread_and_commit_in_order() -> None:
    gate = threading.Event()
    seen: list[str] = []

    def slow(pcm: np.ndarray) -> str:
        seen.append(threading.current_thread().name)
        gate.wait(5)
        return seconds_of(pcm)

    e = HybridSttEngine(Live(), slow, sample_rate=RATE)  # type: ignore[arg-type]
    during = feed_all(e, speech(9) + quiet(1) + speech(9) + quiet(1))
    # The first piece is still decoding, so nothing is committed yet and
    # feeding carried on regardless.
    assert commits(during) == []
    gate.set()
    # The trailing 0.4 s of quiet is decoded too: judging a piece silent by
    # level alone would drop a quiet speaker's words.
    assert commits(e.finalize()) == ["piece 9.6s", "piece 10.0s", "piece 0.4s"]
    assert threading.current_thread().name not in seen


def test_each_session_hands_its_decode_memory_back() -> None:
    """glibc keeps freed decode buffers, ~250 MB after a 70 s dictation (ADR 0011)."""
    trims: list[str] = []
    e = HybridSttEngine(
        Live(), seconds_of, sample_rate=RATE, executor=Now(), trim=lambda: trims.append("t")
    )  # type: ignore[arg-type]
    feed_all(e, speech(1))
    e.finalize()
    feed_all(e, speech(1))
    e.reset()
    assert trims == ["t", "t"]


# --- preview by re-decoding the open piece (ADR 0016) -------------------------


class Manual:
    """An executor that holds work until the test runs it, like a busy worker.

    With `auto` set it runs work at submit, like `Now`.
    """

    def __init__(self) -> None:
        self.auto = False
        self.queue: list[tuple[Future[str], Callable[..., str], tuple[object, ...]]] = []
        self.submitted: list[float] = []

    def submit(self, fn: Callable[..., str], *args: object) -> "Future[str]":
        future: Future[str] = Future()
        self.submitted.append(len(args[0]) / RATE)  # type: ignore[arg-type]
        if self.auto:
            future.set_result(fn(*args))
        else:
            self.queue.append((future, fn, args))
        return future

    def run_one(self) -> None:
        """Run the oldest job that was not cancelled."""
        while self.queue:
            future, fn, args = self.queue.pop(0)
            if future.set_running_or_notify_cancel():
                future.set_result(fn(*args))
                return


def redecoder(
    transcribe: Callable[[np.ndarray], str] = seconds_of,
    executor: object | None = None,
    interval: float = 0.7,
) -> HybridSttEngine:
    return HybridSttEngine(
        None,
        transcribe,
        sample_rate=RATE,
        executor=executor or Now(),  # type: ignore[arg-type]
        redecode_interval_s=interval,
        trim=lambda: None,
    )


def partials(events: list[Event]) -> list[str]:
    return [ev.text for ev in events if isinstance(ev, Partial)]


def test_the_open_piece_is_decoded_again_every_interval() -> None:
    ex = Manual()
    e = redecoder(executor=ex)
    feed_all(e, speech(0.6))
    assert ex.submitted == []
    feed_all(e, speech(0.1))
    assert ex.submitted == [0.7]
    ex.run_one()
    # The result is picked up by the next block, not the one that asked.
    assert partials(feed_all(e, speech(0.1))) == ["piece 0.7s"]


def test_the_preview_is_the_latest_decode_of_the_whole_open_piece() -> None:
    e = redecoder()
    events = feed_all(e, speech(2.2))
    assert partials(events) == ["piece 0.7s", "piece 1.4s", "piece 2.1s"]
    assert commits(events) == []


def test_a_tick_is_skipped_while_a_decode_is_running_and_never_queued() -> None:
    ex = Manual()
    e = redecoder(executor=ex)
    feed_all(e, speech(5))
    # Seven intervals went by with the worker busy: one decode, no backlog.
    assert ex.submitted == [0.7]
    ex.run_one()
    feed_all(e, speech(0.1))
    # The next one covers everything heard so far, not a stale interval.
    assert ex.submitted == [0.7, 5.1]


def test_a_failed_preview_decode_keeps_the_last_preview(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[int] = []

    def flaky(pcm: np.ndarray) -> str:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("out of memory")
        return seconds_of(pcm)

    e = redecoder(flaky)
    with caplog.at_level("WARNING"):
        events = feed_all(e, speech(2.2))
    assert partials(events) == ["piece 0.7s", "piece 2.1s"]
    assert any("out of memory" in r.getMessage() for r in caplog.records)


def test_commits_are_pieces_decoded_once_and_a_preview_never_commits() -> None:
    e = redecoder()
    during = feed_all(e, speech(9) + quiet(1) + speech(2))
    assert commits(during) == ["piece 9.6s"]
    assert commits(e.finalize()) == ["piece 2.4s"]


def test_a_cut_piece_stays_on_screen_until_it_lands_and_the_next_starts_fresh() -> None:
    ex = Manual()
    ex.auto = True
    e = redecoder(executor=ex)
    feed_all(e, speech(9.2))
    assert e._last_preview == "piece 9.1s"

    ex.auto = False  # the worker is busy when the piece is cut
    events = feed_all(e, quiet(0.6))
    assert commits(events) == [] and ex.queue
    assert e._last_preview == "piece 9.1s"  # no blank flash at the cut
    events = feed_all(e, speech(1.0))
    assert commits(events) == []
    assert e._last_preview == "piece 9.1s"  # still held: nothing decodes behind the piece

    ex.run_one()
    ex.auto = True
    events = feed_all(e, speech(0.1))
    assert commits(events) == ["piece 9.8s"]
    # The committed piece is gone from the preview; what is left is the next piece.
    assert e._pieces == []
    events = feed_all(e, speech(1.0))
    assert partials(events)[-1].startswith("piece 1.")


def test_no_preview_decode_is_started_while_a_piece_is_waiting_for_the_worker() -> None:
    ex = Manual()
    e = redecoder(executor=ex)
    feed_all(e, speech(9) + quiet(1))
    before = len(ex.submitted)
    feed_all(e, speech(3))
    assert len(ex.submitted) == before


def test_a_queued_preview_decode_is_dropped_when_a_piece_is_cut() -> None:
    """Commits have priority: the piece must not wait behind a preview that has
    not started."""
    ex = Manual()
    e = redecoder(executor=ex)
    feed_all(e, speech(9))
    preview, _, _ = ex.queue[0]
    feed_all(e, quiet(1))
    assert preview.cancelled()
    ex.run_one()  # the first job that is left is the piece
    assert commits(feed_all(e, quiet(0.1))) == ["piece 9.6s"]


def test_a_preview_decode_finishing_after_a_cut_is_not_shown_for_the_new_piece() -> None:
    ex = Manual()
    e = redecoder(executor=ex)
    feed_all(e, speech(9))
    future, fn, args = ex.queue.pop(0)
    assert future.set_running_or_notify_cancel()  # started: cannot be cancelled
    feed_all(e, quiet(1))
    future.set_result(fn(*args))
    ex.run_one()
    feed_all(e, speech(0.1))
    assert e._open_text == ""
    assert e._pieces == []  # the piece itself landed


def test_finalize_commits_the_last_piece_and_forgets_previews() -> None:
    e = redecoder()
    feed_all(e, speech(3))
    assert commits(e.finalize()) == ["piece 3.0s"]
    assert e._open_text == "" and e._pieces == []
    feed_all(e, speech(0.8))
    assert e._last_preview == "piece 0.7s"


def test_finalize_waits_for_a_running_preview_decode_then_commits_the_piece() -> None:
    order: list[str] = []
    started = threading.Event()
    release = threading.Event()

    def decode(pcm: np.ndarray) -> str:
        if not order:
            order.append("preview start")
            started.set()
            release.wait(5)
            order.append("preview end")
            return "preview"
        order.append("piece")
        return "final"

    e = HybridSttEngine(None, decode, sample_rate=RATE, trim=lambda: None, redecode_interval_s=0.3)
    feed_all(e, speech(0.5))
    assert started.wait(5)
    threading.Timer(0.2, release.set).start()
    assert commits(e.finalize()) == ["final"]
    assert order == ["preview start", "preview end", "piece"]


def test_a_failed_piece_falls_back_to_its_last_preview(caplog: pytest.LogCaptureFixture) -> None:
    state = {"fail": False}

    def maybe(pcm: np.ndarray) -> str:
        if state["fail"]:
            raise RuntimeError("decoder crashed")
        return seconds_of(pcm)

    e = redecoder(maybe)
    feed_all(e, speech(2.2))
    state["fail"] = True
    with caplog.at_level("WARNING"):
        assert commits(e.finalize()) == ["piece 2.1s"]
    assert any("last preview" in r.getMessage() for r in caplog.records)


def test_reset_discards_previews_and_pieces() -> None:
    ex = Manual()
    e = redecoder(executor=ex)
    feed_all(e, speech(9) + quiet(1))
    e.reset()
    assert e.finalize() == []
    assert e._open_text == "" and e._last_preview == ""
    assert e._pieces == []


def test_a_preview_decode_finishing_after_reset_is_dropped() -> None:
    ex = Manual()
    e = redecoder(executor=ex)
    feed_all(e, speech(1))
    future, fn, args = ex.queue.pop(0)
    assert future.set_running_or_notify_cancel()
    e.reset()
    future.set_result(fn(*args))
    assert partials(feed_all(e, speech(0.1))) == []
    assert e._open_text == ""


def test_keyterms_are_ignored_without_a_live_model() -> None:
    redecoder().set_keyterms(["flowd"])


def test_decodes_run_on_one_worker_thread_off_the_feeding_thread() -> None:
    names: list[str] = []

    def record(pcm: np.ndarray) -> str:
        names.append(threading.current_thread().name)
        return seconds_of(pcm)

    e = HybridSttEngine(None, record, sample_rate=RATE, trim=lambda: None, redecode_interval_s=0.3)
    feed_all(e, speech(9) + quiet(1) + speech(2))
    e.finalize()
    assert len(set(names)) == 1
    assert threading.current_thread().name not in names
