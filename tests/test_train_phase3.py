from scout.train_phase3 import STAGES


def test_phase3_stages_are_registered():
    assert list(STAGES) == ["market", "trajectory", "availability", "resale", "kill_checks"]
