"""Moneyball replacement scouting — reads committed artifacts only, recomputes nothing."""

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

MODELS = Path(__file__).resolve().parent.parent / "models"

GROUP_ORDER = ["finishing", "creation", "build-up", "passing", "defending", "discipline", "keeper"]
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


def shortlist_table(rows):
    frame = pd.DataFrame(rows)
    frame["value_eur"] = frame["value_eur"] / 1e6
    st.dataframe(
        frame,
        hide_index=True,
        column_config={
            "name": st.column_config.TextColumn("Player"),
            "value_eur": st.column_config.NumberColumn("Value (€m)", format="%.1f"),
            "similarity": st.column_config.NumberColumn(
                "Distance to X", format="%.2f", help="Lower = more similar to the departing player"
            ),
            "p_bar": st.column_config.ProgressColumn(
                "P(≥ his level)", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "prod_per_eur": st.column_config.NumberColumn(
                "Surplus per €m",
                format="%.2f",
                help="Expected production above a freely-available player, per €m",
            ),
            "expected_minutes": st.column_config.NumberColumn("Expected minutes", format="%.0f"),
            "outcome_minutes": st.column_config.NumberColumn("Minutes next season", format="%.0f"),
            "outcome_ga90": st.column_config.NumberColumn("G+A/90 next season", format="%.2f"),
        },
    )


st.set_page_config(page_title="Moneyball replacement scouting", layout="wide")
page = st.sidebar.radio(
    "Pages", ["We're losing X", "Players like X", "The backtest", "The graveyard"]
)

if page == "We're losing X":
    demo = load("phase6_shortlists")
    st.title("We're losing X — who replaces him?")
    st.caption(
        f"Precomputed for a summer-{demo['as_of_summer']} departure from the last completed "
        "season's data. Every candidate plays his role, resembles his profile, and costs no "
        "more than his market value."
    )
    name = st.selectbox("The player you are losing", sorted(demo["players"]))
    entry = demo["players"][name]
    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"{name} — {entry['role']}, {entry['club']}")
        st.metric("Market value", f"€{entry['value_eur'] / 1e6:.0f}m")
        labels = {
            "default": "Best keeper (goals prevented)" if entry["role"] == "GK" else "Best value",
            "output": "Best player (P ≥ his level)",
            "blend": "Most like him, productive",
        }
        tabs = st.tabs(list(labels.values()))
        for tab, key in zip(tabs, labels, strict=True):
            with tab:
                if key == "default" and entry["role"] != "GK":
                    st.caption(
                        "Maximum delivered contribution per euro — the ordering that won the "
                        "backtest. It favours proven, affordable players; for the marquee "
                        "names see 'Best player'."
                    )
                shortlist_table(entry["shortlists"][key])
    with right:
        st.subheader("His profile card")
        st.caption("Only stats that repeat year to year for his position (r ≥ 0.3).")
        card = pd.DataFrame(entry["card"])
        if not card.empty:
            groups = load("phase2_role_profiles")["profiles"].get(entry["role"], {})
            group_of = {s["stat"]: g for g, stats in groups.items() for s in stats}
            card["group"] = card["stat"].map(group_of)
            card["stat"] = card["stat"].str.replace("_", " ")
            card = card.sort_values(
                "group", key=lambda s: s.map({g: i for i, g in enumerate(GROUP_ORDER)})
            )
            st.dataframe(
                card[["group", "stat", "per90", "role_percentile"]],
                hide_index=True,
                height=520,
                column_config={
                    "group": st.column_config.TextColumn(""),
                    "stat": st.column_config.TextColumn("Stat"),
                    "per90": st.column_config.NumberColumn("Per 90", format="%.2f"),
                    "role_percentile": st.column_config.ProgressColumn(
                        "Percentile in role", min_value=0.0, max_value=1.0, format="%.2f"
                    ),
                },
            )

elif page == "Players like X":
    demo = load("phase6_shortlists")
    st.title("Players like X")
    name = st.selectbox("Player", sorted(demo["players"]))
    entry = demo["players"][name]
    st.caption(
        f"{entry['role']} — nearest statistical neighbours in his role last season, "
        "over the fifteen stable profile traits. No budget filter."
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
    label_of = {
        cid: f"{c['season']} — {c['departed']} ({c['seller']}, €{c['fee_eur'] / 1e6:.0f}m)"
        for cid, c in cases.items()
    }
    cid = st.selectbox("Departure", sorted(label_of, key=label_of.get), format_func=label_of.get)
    case = cases[cid]
    if "actual_signing" in case:
        a = case["actual_signing"]
        st.write(
            f"The club actually signed **{a['name']}** for €{a['fee_eur'] / 1e6:.0f}m — "
            f"{a['outcome_minutes'] or 0:.0f} minutes, "
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
    st.title("The graveyard — ideas the data killed")
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
                st.write(re.sub(r"\s*\(notebook [^)]*\)", "", value))
            elif key == "style_fit_matched_transfer":
                st.write(
                    f"Moving to a stylistically different club changes output by "
                    f"{value['distance_effect_per_sd']:+.4f} sd — indistinguishable from zero "
                    f"(80% CI [{value['ci80'][0]:.4f}, {value['ci80'][1]:.4f}], "
                    f"{value['movers']:,} matched transfers)."
                )
            else:
                st.write(json.dumps(value)[:300])
