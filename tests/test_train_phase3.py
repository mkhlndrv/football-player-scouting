from scout.train_phase3 import STAGES


def test_phase3_stages_are_registered():
    assert list(STAGES) == ["market", "trajectory", "availability", "resale", "kill_checks"]


def test_phase4_stages_are_registered():
    from scout.train_phase4 import STAGES as PHASE4

    assert list(PHASE4) == ["fit", "similarity", "kill_checks_phase4"]
