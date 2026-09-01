"""Moneyball replacement scouting. Reads committed artifacts only, recomputes nothing.

Shortlists are sorted in-app from the stored candidate pools by the fixed ordering keys;
no model output is computed here.
"""

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

MODELS = Path(__file__).resolve().parent.parent / "models"

GROUP_ORDER = ["finishing", "creation", "build-up", "passing", "defending", "discipline", "keeper"]
ROLE_NAMES = {
    "GK": "Goalkeeper",
    "CB": "Centre-back",
    "FB": "Full-back",
    "CM": "Central midfielder",
    "W": "Winger / attacking midfielder",
    "ST": "Striker",
}
PRETTY_STATS = {
    "npxg": "npxG",
    "xa": "xA",
    "xg_chain": "xG chain",
    "xg_buildup": "xG build-up",
    "possession_won_att_third": "possession won, attacking third",
}
MARQUEE = ["Rodri", "Bukayo Saka", "Vinicius Junior", "Jude Bellingham", "Florian Wirtz"]
METRIC_HELP = {
    "Value (€m)": "the player's market value at 1 July.",
    "Age": "age at the transfer summer.",
    "Distance to X": "how differently he plays from the departing player, over fifteen "
    "stable traits (chance quality, build-up, defending, duels, shot locations). "
    "Lower = more similar.",
    "Contribution per 90": "expected non-penalty xG plus xA per 90, smoothed over three "
    "seasons and shrunk toward the position average when a player has few minutes.",
    "Production next season": "contribution above a freely available player, multiplied by "
    "expected minutes. This is what a player is expected to add over a season, not just his "
    "rate while on the pitch.",
    "Duel quality": "ground and aerial duels won, standardised within the position. It is a "
    "measured trait that repeats year to year and survives a transfer, but it is not defensive "
    "value in goals, which no public data measures.",
    "P(≥ his level)": "the calibrated probability that he matches the departing player's "
    "per-90 contribution next season (for keepers: goals prevented). When this says 70%, "
    "it happens about 70% of the time.",
    "Surplus per €m": "expected contribution above a freely-available player, times expected "
    "minutes, divided by price. This is the number that won the ten-year backtest.",
    "Expected minutes": "forecast league minutes next season, from his last three seasons and age.",
}
GRAVEYARD_TITLES = {
    "finishing_residual": "Finishing skill",
    "defensive_value_on_off": "Defensive value from goals conceded (on/off)",
    "opponent_slope": "The big-game player",
    "price_gaps": "Persistently underpriced players",
    "availability_injuries": "Injury history in minutes forecasts",
    "trajectory": "Player-specific ageing",
    "role_switch_cost": "The cost of switching roles",
    "style_fit_matched_transfer": "Style fit",
}


@st.cache_data
def load(name):
    return json.loads((MODELS / f"{name}.json").read_text())


def plain(text):
    """Artifact strings are written for the state log. Tidy them for a reader."""
    text = re.sub(r"[;,]?\s*\bnotebook \d+[^)\n]*", "", text)
    text = re.sub(r"\s*\(\s*\)", "", text)
    text = text.replace("y2y", "year to year")
    text = re.sub(r";\s*", ". ", text)
    text = re.sub(r"\s+", " ", text).strip(" ;,")
    if text.count("(") > text.count(")"):
        text += ")"
    return re.sub(r"(^|\. )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def explain_metrics():
    with st.expander("What do these numbers mean?"):
        for metric, meaning in METRIC_HELP.items():
            st.markdown(f"**{metric}**: {meaning}")


def shortlist_table(rows):
    frame = pd.DataFrame(rows)
    frame["value_eur"] = frame["value_eur"] / 1e6
    st.dataframe(
        frame,
        hide_index=True,
        column_config={
            "name": st.column_config.TextColumn("Player"),
            "age": st.column_config.NumberColumn("Age", format="%.0f"),
            "value_eur": st.column_config.NumberColumn("Value (€m)", format="%.1f"),
            "similarity": st.column_config.NumberColumn(
                "Distance to X", format="%.2f", help=METRIC_HELP["Distance to X"]
            ),
            "p_bar": st.column_config.ProgressColumn(
                "P(≥ his level)", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "prod_per_eur": st.column_config.NumberColumn(
                "Surplus per €m", format="%.2f", help=METRIC_HELP["Surplus per €m"]
            ),
            "point": st.column_config.NumberColumn(
                "Contribution per 90", format="%.2f", help=METRIC_HELP["Contribution per 90"]
            ),
            "production": st.column_config.NumberColumn(
                "Production next season", format="%.1f", help=METRIC_HELP["Production next season"]
            ),
            "duel_quality": st.column_config.NumberColumn(
                "Duel quality", format="%.2f", help=METRIC_HELP["Duel quality"]
            ),
            "expected_minutes": st.column_config.NumberColumn("Expected minutes", format="%.0f"),
            "outcome_minutes": st.column_config.NumberColumn("Minutes next season", format="%.0f"),
            "outcome_ga90": st.column_config.NumberColumn("G+A/90 next season", format="%.2f"),
        },
    )


def ordered(pool, key, n=10, defence_weight=0.0):
    if defence_weight and key in ("formula", "production") and "duel_quality" in pool:
        usable = pool.dropna(subset=["duel_quality"])
        if len(usable) >= n:
            base = "prod_per_eur" if key == "formula" else "production"
            rank = (1 - defence_weight) * (-usable[base]).rank() + defence_weight * (
                -usable["duel_quality"]
            ).rank()
            return usable.assign(_k=rank).nsmallest(n, "_k").drop(columns="_k")
    if key == "production":
        rank = (-pool["production"]).rank()
    elif key == "output":
        rank = (-pool["p_bar"]).rank()
    elif key == "formula":
        rank = (-pool["prod_per_eur"]).rank()
    else:  # blend: equal rank-mix of similarity and probability
        rank = (pool["similarity"].rank() + (-pool["p_bar"]).rank()) / 2
    return pool.assign(_k=rank).dropna(subset=["_k"]).nsmallest(n, "_k").drop(columns="_k")


def profile_card(entry, height=520):
    card = pd.DataFrame(entry["card"])
    if card.empty:
        st.info("No profile card for this player.")
        return
    groups = load("phase2_role_profiles")["profiles"].get(entry["role"], {})
    group_of = {s["stat"]: g for g, stats in groups.items() for s in stats}
    card["group"] = card["stat"].map(group_of)
    card["stat"] = card["stat"].map(lambda s: PRETTY_STATS.get(s, s.replace("_", " ")))
    card = card.sort_values("group", key=lambda s: s.map({g: i for i, g in enumerate(GROUP_ORDER)}))
    st.dataframe(
        card[["group", "stat", "per90", "role_percentile"]],
        hide_index=True,
        height=height,
        column_config={
            "group": st.column_config.TextColumn(""),
            "stat": st.column_config.TextColumn("Stat"),
            "per90": st.column_config.NumberColumn("Per 90", format="%.2f"),
            "role_percentile": st.column_config.ProgressColumn(
                "Percentile in role", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )


def pick_player(demo, label):
    roles = ["All positions", *ROLE_NAMES]
    role_pick = st.selectbox(
        "Position", roles, format_func=lambda r: ROLE_NAMES.get(r, r), key="role_filter"
    )
    names = sorted(
        n
        for n, e in demo["players"].items()
        if role_pick == "All positions" or e["role"] == role_pick
    )
    default = st.session_state.get("player_pick_value")
    if default not in names:
        default = next((m for m in MARQUEE if m in names), names[0])
    picked = st.selectbox(label, names, index=names.index(default), key="player_select")
    st.session_state["player_pick_value"] = picked
    return picked


st.set_page_config(page_title="Moneyball replacement scouting", layout="wide")
page = st.sidebar.radio(
    "Pages", ["We're losing X", "Players like X", "The backtest", "What failed"]
)

if page == "We're losing X":
    demo = load("phase6_shortlists")
    st.title("We're losing X. Who replaces him?")
    st.caption(
        f"Precomputed for a summer-{demo['as_of_summer']} departure from the last completed "
        "season's data."
    )
    st.markdown(
        "A smaller club is losing a player and wants the same contribution for far less "
        "money. Pick the departing player: every candidate below plays his position and "
        "resembles his statistical profile. Whether these lists actually beat real scouting "
        "was tested on ten years of transfers. See *The backtest*."
    )
    name = pick_player(demo, "The player you are losing")
    entry = demo["players"][name]
    pool = pd.DataFrame(entry["pool"])
    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"{name}, {ROLE_NAMES[entry['role']]}, {entry['club']}")
        st.metric("Market value", f"€{entry['value_eur'] / 1e6:.0f}m")
        max_budget = float(max(round(entry["value_eur"] / 1e6), 1))
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            budget = st.slider(
                "Budget (max candidate value, €m)",
                min_value=1.0,
                max_value=max_budget,
                value=max_budget,
                help="Candidates never cost more than the departing player's own value. "
                "Tighten the budget to see only cheaper options.",
            )
        with fcol2:
            max_age = st.slider("Max age", min_value=18, max_value=40, value=40)
        defence_weight = 0.0
        if entry["role"] in ("CB", "FB", "CM"):
            defence_weight = st.slider(
                "Weight on defensive quality",
                0.0,
                1.0,
                0.0,
                0.25,
                help="Your judgement, not a validated setting. In the backtest a weight of 0.25 "
                "beat what real clubs achieved on duel quality at centre-back and midfield, at "
                "1 to 3 percent of expected production, but it lost on minutes and value per "
                "euro, which are the columns with nine years of validation behind them.",
            )
            if defence_weight:
                st.caption(
                    "Ranking now trades measured production per euro for duel quality. "
                    "See the backtest page for what each weight delivered."
                )
        filtered = pool[(pool["value_eur"] <= budget * 1e6) & (pool["age"].fillna(0) <= max_age)]
        explain_metrics()
        if filtered.empty:
            st.info("No candidates inside these filters. Widen the budget or age range.")
        else:
            labels = {
                "formula": "Best keeper (goals prevented)"
                if entry["role"] == "GK"
                else "Best value",
                "production": "Most production next season",
                "blend": "Most like him, productive",
            }
            keys = {"formula": "output" if entry["role"] == "GK" else "formula"}
            tabs = st.tabs(list(labels.values()))
            for tab, key in zip(tabs, labels, strict=True):
                with tab:
                    if key == "production":
                        st.caption(
                            "Contribution above a freely available player, multiplied by "
                            "expected minutes. Rate alone flatters part-time players. "
                            + (
                                "For defenders and midfielders this measures attacking output "
                                "only, because defending itself is not measurable from these "
                                "data. See What failed."
                                if entry["role"] in ("CB", "FB", "CM")
                                else ""
                            )
                        )
                    if key == "formula" and entry["role"] != "GK":
                        st.caption(
                            "Maximum delivered contribution per euro. This ordering "
                            "won the backtest. It favours proven, affordable players. "
                            "Tighten the budget slider or switch tabs for other views."
                        )
                    shortlist_table(
                        ordered(filtered, keys.get(key, key), defence_weight=defence_weight)
                    )
    with right:
        st.subheader("His profile card")
        st.caption("Only stats that repeat year to year for his position (r ≥ 0.3).")
        profile_card(entry)

    if not filtered.empty:
        st.divider()
        st.subheader(f"Side by side: {name} vs a candidate")
        order_key = "output" if entry["role"] == "GK" else "formula"
        cand_names = [n for n in ordered(filtered, order_key, n=50)["name"] if n in demo["players"]]
        if cand_names:
            pick = st.selectbox("Candidate to compare", cand_names)
            cand_entry = demo["players"][pick]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"**{name}**, {ROLE_NAMES[entry['role']]}, {entry['club']}, "
                    f"€{entry['value_eur'] / 1e6:.0f}m"
                )
                profile_card(entry)
            with c2:
                st.markdown(
                    f"**{pick}**, {ROLE_NAMES[cand_entry['role']]}, {cand_entry['club']}, "
                    f"€{cand_entry['value_eur'] / 1e6:.0f}m"
                )
                profile_card(cand_entry)
        else:
            st.info("No full profile available for the candidates under these filters.")

elif page == "Players like X":
    demo = load("phase6_shortlists")
    st.title("Players like X")
    name = pick_player(demo, "Player")
    entry = demo["players"][name]
    st.caption(
        f"{ROLE_NAMES[entry['role']]}. Nearest statistical neighbours in his role last "
        "season, over the fifteen stable profile traits. No budget filter."
    )
    shortlist_table(entry["similar"])

elif page == "The backtest":
    bt = load("phase5_backtest")
    cases = load("phase5_shortlists")
    st.title("The departure backtest, 2015–2024")
    st.caption(
        f"{bt['population']['cases_scored']:,} replayed departures, all data frozen at each "
        f"sale; {bt['population']['cases_with_actual_signing']} have the club's actual signing "
        "to beat. A case is won on at least two of: minutes per €m, real G+A/90, value change."
    )
    st.subheader("The formula vs the standing ordering and vs the clubs' signings")
    rows = []
    for role, v in bt["formula_all_roles"].items():
        rows.append(
            {
                "Role": role,
                "vs output-ranking": f"{v['vs_output']['case_win']:.2f}",
                "80% interval": f"[{v['vs_output']['lo']:.2f}, {v['vs_output']['hi']:.2f}]",
                "vs actual signing": f"{v['vs_actual']['case_win']:.2f}",
                "interval ": f"[{v['vs_actual']['lo']:.2f}, {v['vs_actual']['hi']:.2f}]",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.subheader("Model vs the clubs, season by season")
    seasons = pd.DataFrame(
        [
            {"Season": k, "Case-win rate": round(v["case_win"], 2), "Cases": v["n"]}
            for k, v in bt["per_season_vs_actual"].items()
        ]
    )
    st.dataframe(seasons, hide_index=True)
    st.subheader("Browse a real case")
    explain_metrics()
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        role_pick = st.selectbox(
            "Position", ["All positions", *ROLE_NAMES], format_func=lambda r: ROLE_NAMES.get(r, r)
        )
    with fcol2:
        season_pick = st.selectbox(
            "Summer", ["All"] + sorted({c["season"] for c in cases.values()}, reverse=True)
        )
    label_of = {
        cid: f"{c['season']}: {c['departed']} ({c['seller']}, €{c['fee_eur'] / 1e6:.0f}m)"
        for cid, c in cases.items()
        if (role_pick == "All positions" or c["role"] == role_pick)
        and (season_pick == "All" or c["season"] == season_pick)
    }
    cid = st.selectbox("Departure", sorted(label_of, key=label_of.get), format_func=label_of.get)
    case = cases[cid]
    if "actual_signing" in case:
        a = case["actual_signing"]
        st.write(
            f"The club actually signed **{a['name']}** for €{a['fee_eur'] / 1e6:.0f}m. "
            f"He played {a['outcome_minutes'] or 0:.0f} minutes with "
            f"{a['outcome_ga90'] or 0} G+A/90 the following season."
        )
    for key, title in [
        ("prod_per_eur", "The formula (surplus production per euro)"),
        ("o2", "Best player (P ≥ the departed's level)"),
        ("blend", "Most like him, productive"),
    ]:
        st.markdown(f"**{title}**")
        shortlist_table(case["orderings"][key])

else:
    st.title("Ideas that failed their tests")
    st.caption("Every claim below was tested and failed its pre-declared check.")
    k2 = load("phase2_kill_checks")["checks"]
    k3, k4 = load("phase3_kill_checks"), load("phase4_kill_checks")
    entries = [
        (k2, ["finishing_residual", "defensive_value_on_off", "opponent_slope"]),
        (k3, ["price_gaps", "availability_injuries", "trajectory"]),
        (k4, ["role_switch_cost", "style_fit_matched_transfer"]),
    ]
    for source, keys in entries:
        for key in keys:
            value = source.get(key)
            st.markdown(f"**{GRAVEYARD_TITLES.get(key, key)}**")
            if isinstance(value, str):
                st.write(plain(value))
            elif key == "style_fit_matched_transfer":
                st.write(
                    f"Moving to a stylistically different club changes output by "
                    f"{value['distance_effect_per_sd']:+.4f} sd, which cannot be told "
                    f"apart from zero "
                    f"(80% CI [{value['ci80'][0]:.4f}, {value['ci80'][1]:.4f}], "
                    f"{value['movers']:,} matched transfers)."
                )
            else:
                st.write(json.dumps(value)[:300])
