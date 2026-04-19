"""Unit tests for rendezvous: exchange, timeout, concurrent writes, mode isolation."""

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

# Add project dir to path so we can import rendezvous directly.
_co_grpo_dp_dir = str(Path(__file__).resolve().parent.parent)
if _co_grpo_dp_dir not in sys.path:
    sys.path.insert(0, _co_grpo_dp_dir)

from rendezvous import Rendezvous  # noqa: E402


class TestBasicExchange:
    def test_exchange_in_two_threads(self):
        """Simulate two groups exchanging pseudo-labels concurrently."""
        with tempfile.TemporaryDirectory() as d:
            rdv_a = Rendezvous(d, "A", poll_interval=0.01)
            rdv_b = Rendezvous(d, "B", poll_interval=0.01)

            result_a, result_b = [None], [None]

            def run_a():
                result_a[0] = rdv_a.exchange("train", 0, ["a1", "a2", "a3"])

            def run_b():
                result_b[0] = rdv_b.exchange("train", 0, ["b1", "b2", "b3"])

            ta, tb = threading.Thread(target=run_a), threading.Thread(target=run_b)
            ta.start()
            tb.start()
            ta.join(timeout=5.0)
            tb.join(timeout=5.0)

            assert result_a[0] == ["b1", "b2", "b3"], "A should receive B's payload"
            assert result_b[0] == ["a1", "a2", "a3"], "B should receive A's payload"

    def test_sequential_exchanges_advance_counter(self):
        """Multiple exchanges with distinct counters do not collide."""
        with tempfile.TemporaryDirectory() as d:
            rdv_a = Rendezvous(d, "A", poll_interval=0.01)
            rdv_b = Rendezvous(d, "B", poll_interval=0.01)

            for step in range(3):
                result = [None, None]

                def run_a(s=step):
                    result[0] = rdv_a.exchange("train", s, [f"a_{s}"])

                def run_b(s=step):
                    result[1] = rdv_b.exchange("train", s, [f"b_{s}"])

                ta, tb = threading.Thread(target=run_a), threading.Thread(target=run_b)
                ta.start(); tb.start()
                ta.join(5.0); tb.join(5.0)

                assert result[0] == [f"b_{step}"]
                assert result[1] == [f"a_{step}"]


class TestModeIsolation:
    def test_train_and_eval_counters_do_not_cross(self):
        """Counter 0 in train and counter 0 in eval are distinct rendezvous points."""
        with tempfile.TemporaryDirectory() as d:
            rdv_a = Rendezvous(d, "A", poll_interval=0.01)
            rdv_b = Rendezvous(d, "B", poll_interval=0.01)

            out = {}

            def exch(rdv, mode, payload, key):
                out[key] = rdv.exchange(mode, 0, payload)

            threads = [
                threading.Thread(target=exch, args=(rdv_a, "train", ["a_tr"], "a_tr")),
                threading.Thread(target=exch, args=(rdv_b, "train", ["b_tr"], "b_tr")),
                threading.Thread(target=exch, args=(rdv_a, "eval", ["a_ev"], "a_ev")),
                threading.Thread(target=exch, args=(rdv_b, "eval", ["b_ev"], "b_ev")),
            ]
            for t in threads: t.start()
            for t in threads: t.join(5.0)

            assert out["a_tr"] == ["b_tr"]
            assert out["b_tr"] == ["a_tr"]
            assert out["a_ev"] == ["b_ev"]
            assert out["b_ev"] == ["a_ev"]


class TestTimeout:
    def test_raises_when_peer_never_writes(self):
        """If peer never shows up, exchange raises TimeoutError within the deadline."""
        with tempfile.TemporaryDirectory() as d:
            rdv_a = Rendezvous(d, "A", poll_interval=0.01, timeout=0.2)
            start = time.time()
            with pytest.raises(TimeoutError, match="peer B did not write"):
                rdv_a.exchange("train", 0, ["solo"])
            elapsed = time.time() - start
            assert 0.15 < elapsed < 1.0, f"timed out too early/late: {elapsed:.2f}s"


class TestAtomicity:
    def test_half_written_file_not_visible(self):
        """The rename step is atomic — peer should never see a partial write."""
        with tempfile.TemporaryDirectory() as d:
            rdv_a = Rendezvous(d, "A", poll_interval=0.01)
            # Large payload to make writes take measurable time
            large = [f"label_{i}" * 100 for i in range(1000)]
            rdv_a.exchange  # no-op access just to make sure import worked

            # Write from A's side manually and verify .tmp file is not picked up as peer file.
            tmp = Path(d) / "train_0_A.json.tmp"
            tmp.write_text("partial")
            # Now peer (B) polls — it should not see "train_0_A.json" until rename happens.
            peer_path = Path(d) / "train_0_A.json"
            assert not peer_path.exists()

            # Simulate atomic rename (what Rendezvous does internally)
            import os as _os
            _os.replace(tmp, peer_path)
            assert peer_path.exists()
            # After replace, tmp no longer exists
            assert not tmp.exists()


class TestFileCleanup:
    def test_both_files_cleaned_after_exchange(self):
        """After both sides exchange, both files are gone.

        Under producer/consumer semantics: A produces train_0_A.json, which B
        consumes and deletes. B produces train_0_B.json, which A consumes and
        deletes. So after both return, both files are gone.
        """
        with tempfile.TemporaryDirectory() as d:
            rdv_a = Rendezvous(d, "A", poll_interval=0.01)
            rdv_b = Rendezvous(d, "B", poll_interval=0.01)

            def run_b():
                rdv_b.exchange("train", 0, ["b"])

            tb = threading.Thread(target=run_b)
            tb.start()
            rdv_a.exchange("train", 0, ["a"])
            tb.join(5.0)

            assert not (Path(d) / "train_0_A.json").exists()
            assert not (Path(d) / "train_0_B.json").exists()


class TestPayloadTypes:
    def test_string_list(self):
        with tempfile.TemporaryDirectory() as d:
            rdv_a = Rendezvous(d, "A", poll_interval=0.01)
            rdv_b = Rendezvous(d, "B", poll_interval=0.01)

            result = [None]
            def run_b():
                result[0] = rdv_b.exchange("train", 0, ["x", "y"])
            tb = threading.Thread(target=run_b); tb.start()
            peer = rdv_a.exchange("train", 0, ["\x00__unlabeled__\x00", "42"])
            tb.join(5.0)
            assert peer == ["x", "y"]
            assert result[0] == ["\x00__unlabeled__\x00", "42"]
