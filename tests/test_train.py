import json

import numpy as np

from scout.train import _json_default, _write


def test_json_writer_handles_numpy_and_nan(tmp_path):
    path = tmp_path / "x.json"
    _write(
        path, {"a": np.int64(3), "b": np.float64(0.5), "c": np.float64("nan"), "d": np.bool_(True)}
    )
    assert json.loads(path.read_text()) == {"a": 3, "b": 0.5, "c": None, "d": True}
    assert _json_default(np.float32(1.5)) == 1.5
