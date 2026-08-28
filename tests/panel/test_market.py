import duckdb
import numpy as np
import pandas as pd

from scout.panel.market import KINDS, add_cost, classify


def _con():
    con = duckdb.connect(":memory:")
    transfers = pd.DataFrame(
        {
            "player_id": [1, 1, 2, 3, 4, 5, 6],
            "transfer_date": pd.to_datetime(
                [
                    "2020-08-01",
                    "2021-06-30",
                    "2020-07-01",
                    "2020-07-01",
                    "2020-07-01",
                    "2020-07-01",
                    "2019-08-01",
                ]
            ),
            "transfer_season": ["20/21", "20/21", "20/21", "20/21", "20/21", "20/21", "19/20"],
            "from_club_id": [10, 20, 10, 10, 10, 10, 10],
            "to_club_id": [20, 10, 20, 20, 30, 99, 20],
            "from_club_name": ["A"] * 7,
            "to_club_name": ["B"] * 7,
            "transfer_fee": [0.0, 0.0, 5e6, 0.0, None, None, 2e6],
            "market_value_in_eur": [1e6, 1e6, 4e6, 3e6, 2e6, 5e5, 1e6],
        }
    )
    clubs = pd.DataFrame({"club_id": ["10", "20", "30"]})
    con.register("transfers", transfers)
    con.register("clubs", clubs)
    return con


def test_mirror_rule_and_fee_classes():
    out = classify(_con()).sort_values(["player_id", "transfer_date"])
    assert out.kind.tolist() == [
        "loan_out",
        "loan_return",
        "paid",
        "free",
        "undisclosed",
        "internal",
        "paid",
    ]
    assert set(out.kind) <= set(KINDS)


def test_cost_rule():
    out = add_cost(classify(_con())).set_index(["player_id", "kind"]).cost
    assert out[(2, "paid")] == 5e6
    assert out[(3, "free")] == 3e6 and out[(4, "undisclosed")] == 2e6
    assert np.isnan(out[(1, "loan_out")]) and np.isnan(out[(5, "internal")])
