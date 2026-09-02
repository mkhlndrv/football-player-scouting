"""Every number quoted in the README must match the committed artifacts."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
FLAT = " ".join(README.split())  # prose wraps across lines; match on normalised whitespace


def artifact(name):
    return json.loads((ROOT / "models" / f"{name}.json").read_text())


def quoted(text):
    assert " ".join(text.split()) in FLAT, f"README no longer quotes {text!r}"


def test_backtest_population():
    bt = artifact("phase5_backtest")
    quoted(f"{bt['population']['cases_scored']:,} real summer departures")
    quoted(f"{bt['population']['cases_with_actual_signing']} of the cases")


@pytest.mark.parametrize(
    ("role", "label"),
    [
        ("W", "Wingers"),
        ("CM", "Central midfielders"),
        ("CB", "Centre-backs"),
        ("FB", "Full-backs"),
        ("ST", "Strikers"),
        ("GK", "Goalkeepers"),
    ],
)
def test_formula_table(role, label):
    row = artifact("phase5_backtest")["formula_all_roles"][role]
    vs_output = f"{row['vs_output']['case_win']:.2f}"
    vs_actual = f"{row['vs_actual']['case_win']:.2f}"
    line = next(li for li in README.splitlines() if li.startswith(f"| {label}"))
    assert vs_output in line and vs_actual in line, f"{label}: {vs_output}/{vs_actual} vs {line}"


def test_vs_market_smaller_club():
    v = artifact("phase5_backtest")["smaller_club_split_elo_rank_gt_6"]["market"]
    quoted(f"{v['case_win']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]")


def test_per_season_worst_year():
    seasons = artifact("phase5_backtest")["per_season_vs_actual"]
    worst = min(seasons.items(), key=lambda kv: kv[1]["case_win"])
    assert worst[0] == "2021"
    quoted(f"2021 was its worst year at {worst[1]['case_win']:.2f}")
    at_least_even = sum(v["case_win"] >= 0.5 for v in seasons.values())
    words = ["zero", "one", "two", "three", "four", "five", "six"]
    quoted(f"parity or better in {words[at_least_even]} of ten seasons")


def test_market_model_error():
    held = artifact("phase3_kill_checks")["market_held_out"]
    quoted(f"typical miss {held['mdape_eur']:.0%}")


def test_availability_beats_baseline():
    a = artifact("phase3_kill_checks")["availability_beats_last_season"]
    improvement = 1 - a["mae_model"] / a["mae_baseline"]
    quoted(f'beats "same as last year" by {improvement:.0%}')


def test_league_factor():
    rows = artifact("phase2_league_factors")
    tier = next(r for r in rows if r["from"] == "feeder" and r["to"] == "big5")
    quoted(f"keep {tier['factor']:.0%} [{tier['p10']:.0%}, {tier['p90']:.0%}] of their output")


def test_minutes_reconciliation():
    report = artifact("phase1_data_report")
    m = report["minutes_reconciliation_per_match"]
    quoted(f"within 5% for {m['within_5pct_share']:.1%} of {m['club_seasons']:,} club-seasons")


def test_identity():
    ident = artifact("phase1_data_report")["identity"]["understat"]
    quoted(f"bridges {ident['minutes_rate_overall']:.1%} of Big-5 minutes")
    quoted(f"worst league-season {ident['minutes_rate_big5_min']:.1%}")


def test_interval_calibration_range():
    cov = artifact("phase2_kill_checks")["checks"]["interval_calibration_80"]
    values = [round(v["coverage"], 2) for v in cov.values()]
    quoted(f"between {min(values):.2f} and {max(values):.2f} of the time")


def test_availability_maes():
    a = artifact("phase3_kill_checks")["availability_beats_last_season"]
    quoted(f"{a['mae_model']:.0f} vs {a['mae_baseline']:.0f} minutes of average error")


def test_resale_coverage():
    r = artifact("phase3_kill_checks")["resale_calibration_base_2021"]
    bands = " / ".join(f"{r[str(h)]['coverage_80']:.2f}" for h in (1, 2, 3))
    quoted(f"{bands} coverage")


def test_market_rmse():
    held = artifact("phase3_kill_checks")["market_held_out"]
    quoted(f"held-out error is {held['rmse_log10']:.3f} in log-value terms")


def test_tier_factor_table():
    rows = artifact("phase2_league_factors")
    tiers = {(r["from"], r["to"]): r["factor"] for r in rows if r["from"] in ("big5", "feeder")}
    line = next(li for li in README.splitlines() if li.startswith("| Feeder league to Big 5"))
    assert f"{tiers[('feeder', 'big5')]:.2f}" in line
    for pair, label in [
        (("big5", "big5"), "| Big 5 to Big 5"),
        (("big5", "feeder"), "| Big 5 to feeder"),
        (("feeder", "feeder"), "| Feeder to feeder"),
    ]:
        line = next(li for li in README.splitlines() if li.startswith(label))
        assert f"{tiers[pair]:.2f}" in line, (label, tiers[pair])


def test_cheapness_disclosure():
    from statistics import median

    cases = artifact("phase5_shortlists")
    picks, fees = [], []
    for case in cases.values():
        signing = case.get("actual_signing")
        shortlist = case["orderings"]["prod_per_eur"]
        if signing and shortlist:
            picks.append(median(p["value_eur"] for p in shortlist))
            fees.append(signing["fee_eur"])
    quoted(f"median price of €{median(picks) / 1e6:.1f}m")
    quoted(f"median actual signing fee of €{median(fees) / 1e6:.0f}m")


def test_defensive_quality_findings():
    d = artifact("phase5_backtest")["defensive_quality_vs_actual"]
    cb, fb, st_ = d["CB"]["model"], d["FB"]["model"], d["ST"]["model"]
    quoted(
        f"at centre-back its picks delivered {cb['shortlist_mean_z']:+.2f} against the "
        f"clubs' {cb['actual_signing_mean_z']:+.2f}"
    )
    quoted(f"at full-back {fb['shortlist_mean_z']:+.2f} against {fb['actual_signing_mean_z']:+.2f}")
    quoted(f"{st_['shortlist_mean_z']:+.2f} against {st_['actual_signing_mean_z']:+.2f}")


def test_defence_weight_tradeoff():
    t = artifact("phase5_backtest")["defence_weight_tradeoff"]["0.25"]
    quoted(f"rises to {t['CB']['duel_quality_delivered']:.2f} at centre-back")
    quoted(f"{t['CM']['duel_quality_delivered']:.2f} at central midfield")


def test_tournament_orderings():
    t = artifact("phase5_backtest")["tournament_defensive_roles"]
    fit = [f"{t[r]['fit_vs_output']['case_win']:.2f}" for r in ("CB", "FB", "CM")]
    quoted(f"at {fit[0]}, {fit[1]} and {fit[2]} for centre-backs")
    blend = [f"{t[r]['blend_vs_fit']['case_win']:.2f}" for r in ("CB", "FB", "CM")]
    quoted(f"beat similarity first at {blend[0]}, {blend[1]} and {blend[2]}")


def test_worked_cases():
    cases = artifact("phase5_shortlists")
    maguire = next(
        c for c in cases.values() if c["departed"] == "Harry Maguire" and c["season"] == 2019
    )
    boly, schar = maguire["orderings"]["o2"][:2]
    assert (boly["name"], schar["name"]) == ("Willy Boly", "Fabian Schär")
    quoted(f"led by Willy Boly at {boly['value_eur'] / 1e6:.0f}m euros")
    quoted(f"Boly played {boly['outcome_minutes']:,.0f} league minutes")
    quoted(f"Schär {schar['outcome_minutes']:,.0f} with {schar['outcome_ga90']:.2f} goals+assists")
    rice = next(c for c in cases.values() if c["departed"] == "Declan Rice" and c["season"] == 2023)
    kovacic, jensen = rice["orderings"]["blend"][:2]
    assert (kovacic["name"], jensen["name"]) == ("Mateo Kovacic", "Mathias Jensen")
    quoted(f"led by Mateo Kovacic at {kovacic['value_eur'] / 1e6:.0f}m")
    quoted(f"Jensen delivered {jensen['outcome_minutes']:,.0f} at {jensen['outcome_ga90']:.2f}")
    actual = rice["actual_signing"]
    quoted(f"played {actual['outcome_minutes']:,.0f} minutes at {actual['outcome_ga90']:.2f}")


def test_confirmation():
    c = artifact("phase5_confirmation")
    pooled = c["pooled"]["outfield"]["actual"]
    quoted(
        f"beat the clubs' actual signings in {pooled['case_win']:.2f} "
        f"[{pooled['lo']:.2f}, {pooled['hi']:.2f}] of {pooled['n']} cases"
    )
    quoted(f"repeated the verdict at {pooled['case_win']:.2f}")
    per = c["design_verdicts_per_role"]
    quoted(f"{per['CM']['actual']['case_win']:.2f} at central midfield")
    quoted(f"{per['W']['actual']['case_win']:.2f} at winger")
    quoted(f"{per['CB']['actual']['case_win']:.2f} at centre-back")
    quoted(f"{per['FB']['actual']['case_win']:.2f} at full-back")
    quoted(f"{per['ST']['actual']['case_win']:.2f} at striker")
    quoted(f"the market ordering it won {c['pooled']['outfield']['market']['case_win']:.2f}")
    quoted(f"rule {c['pooled']['outfield']['naive']['case_win']:.2f}")
    quoted(f"failed again at {per['GK']['actual']['case_win']:.2f}")
    context = c["context_not_preregistered"]
    quoted(
        f"output at {context['output_all']['actual']['case_win']:.2f} and the blend at "
        f"{context['blend']['actual']['case_win']:.2f}"
    )
    quoted(f"all {c['population']['cases_scored']} paid summer-2025 departures")
