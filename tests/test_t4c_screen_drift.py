import numpy as np

from research.t3_screen_20261015 import drift_t4c


def test_standardize_capture_uses_verified_pooled_weights(monkeypatch):
    summary = {
        "rounds": {
            "62": {
                "arms": {
                    "B8": {
                        "global_decile_cutpoints": list(np.arange(9, dtype=float)),
                        "consequential": {"strata": {"700_999:D1": {"weight": 1.0}}},
                        "five_seed_mean": {"consequential_change_rate": 0.2},
                    }
                }
            }
        }
    }
    capture = {
        "length": np.array([800]),
        "fingerprint": np.array(["x"]),
        "ctrl_q": np.array([[1.0, 0, 0, 0, 0, 0]]),
        "t4c_q": np.array([[0.0, 1, 0, 0, 0, 0]]),
    }
    monkeypatch.setattr(drift_t4c, "_bank_fingerprints", lambda summary: np.array(["other"]))
    result = drift_t4c.standardize_capture(capture, summary)
    assert result["consequential_change_rate"] == 1.0
