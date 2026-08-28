import pandas as pd
import pytest

from scout.panel.workrate import PROVIDER_SPECIFIC, SHARED, fotmob_per90, sofascore_per90


def test_shared_metrics_pair_every_regressed_column_once():
    assert len(SHARED) == 16
    assert "possessionWonAttThird" not in {v[0] for v in SHARED.values()}
    assert set(PROVIDER_SPECIFIC) == {"sofascore", "fotmob"}


def test_sofascore_totals_become_per90():
    totals = {column: [0] for column, _, _ in SHARED.values()}
    totals.update({"minutesPlayed": [1800], "tackles": [40], "possessionWonAttThird": [9]})
    out = sofascore_per90(pd.DataFrame(totals))
    assert out.tackles.iloc[0] == 2.0
    assert out.possession_won_att_third_sofascore.iloc[0] == 0.45


def test_fotmob_per90_keeps_rates_and_converts_totals():
    rows = pd.DataFrame(
        {"fm_minutes": [1800], "total_tackle": [2.0], "total_att_assist": [30], "goals": [6]}
    )
    out = fotmob_per90(rows)
    assert out.tackles.iloc[0] == 2.0  # per-90 list: taken as is
    assert out.key_passes.iloc[0] == 1.5  # total list: / minutes * 90
    assert out.goals.iloc[0] == pytest.approx(0.3)
    assert pd.isna(out.xg.iloc[0])  # stat absent from the frame -> NaN, never 0
