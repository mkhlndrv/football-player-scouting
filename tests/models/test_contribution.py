import pandas as pd

from scout.models.contribution import CORE, expected_output


def test_expected_output_is_npxg_plus_xa_only():
    per90 = pd.DataFrame({"npxg": [0.3, 0.1], "xa": [0.2, 0.0], "goals": [9, 9]})
    assert expected_output(per90).tolist() == [0.5, 0.1]
    assert set(CORE) == {"npxg", "xa"}
