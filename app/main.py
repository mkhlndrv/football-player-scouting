"""Moneyball replacement scouting — reads committed artifacts only, recomputes nothing."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

MODELS = Path(__file__).resolve().parent.parent / "models"


@st.cache_data
def load(name):
    return json.loads((MODELS / f"{name}.json").read_text())


st.set_page_config(page_title="Moneyball replacement scouting", layout="wide")
page = st.sidebar.radio(
    "Pages",
    ["We're losing X", "Players like X", "The backtest", "The graveyard"],
)

if page == "We're losing X":
    demo = load("phase6_shortlists")
    st.title("We're losing X — who replaces him?")
    st.caption(
        f"Precomputed for a summer-{demo['as_of_summer']} departure from the last completed "
        "season's data. Shortlists are profile-gated and budget-capped at the player's value."
    )
    name = st.selectbox("The player you are losing", sorted(demo["players"]))
    entry = demo["players"][name]
    left, right = st.columns([2, 1])
    with left:
        st.subheader(f"{name} — {entry['role']}, {entry['club']}")
        st.metric("Market value", f"€{entry['value_eur'] / 1e6:.0f}m")
        labels = {
            "default": "Best value (surplus production per euro)"
            if entry["role"] != "GK"
            else "Best keeper (goals prevented)",
            "output": "Best player (P ≥ his level)",
            "blend": "Most like him, productive",
        }
        tabs = st.tabs(list(labels.values()))
        for tab, key in zip(tabs, labels, strict=True):
            with tab:
                tab.dataframe(
                    pd.DataFrame(entry["shortlists"][key]).assign(
                        value_eur=lambda d: (d.value_eur / 1e6).round(1)
                    ),
                    hide_index=True,
                )
    with right:
        st.subheader("His profile card")
        card = pd.DataFrame(entry["card"])
        if not card.empty:
            st.dataframe(card, hide_index=True)

elif page == "Players like X":
    demo = load("phase6_shortlists")
    st.title("Players like X")
    name = st.selectbox("Player", sorted(demo["players"]))
    entry = demo["players"][name]
    st.caption(f"{entry['role']} — nearest statistical neighbours in his role, last season.")
    st.dataframe(
        pd.DataFrame(entry["similar"]).assign(value_eur=lambda d: (d.value_eur / 1e6).round(1)),
        hide_index=True,
    )

elif page == "The backtest":
    bt = load("phase5_backtest")
    cases = load("phase5_shortlists")
    st.title("The departure backtest, 2015–2024")
    st.caption(
        f"{bt['population']['cases_scored']} replayed departures; "
        f"{bt['population']['cases_with_actual_signing']} with the club's actual signing to beat. "
        "A case is won on at least two of: minutes per €m, real G+A/90, value change."
    )
    st.subheader("The formula vs the standing ordering and vs the clubs' signings")
    rows = []
    for role, v in bt["formula_all_roles"].items():
        rows.append(
            {
                "role": role,
                "vs output-ranking": v["vs_output"].get("case_win"),
                "vs actual signing": v["vs_actual"].get("case_win"),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.subheader("Model vs the clubs, season by season")
    st.dataframe(
        pd.DataFrame([{"season": k, **v} for k, v in bt["per_season_vs_actual"].items()]),
        hide_index=True,
    )
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
            f"{a['outcome_ga90'] or 0} G+A/90 the next season."
        )
    for key, title in [("prod_per_eur", "Formula"), ("o2", "Output"), ("blend", "Blend")]:
        st.markdown(f"**{title} shortlist**")
        st.dataframe(
            pd.DataFrame(case["orderings"][key]).assign(
                value_eur=lambda d: (d.value_eur / 1e6).round(1)
            ),
            hide_index=True,
        )

else:
    st.title("The graveyard — ideas the data killed")
    st.caption(
        "Every claim below was tested and failed its pre-declared check. "
        "Sources: the kill-check artifacts."
    )
    k2 = load("phase2_kill_checks")
    k3, k4 = load("phase3_kill_checks"), load("phase4_kill_checks")
    for source, keys in [
        (k2["checks"], ["finishing_residual", "defensive_value_on_off", "opponent_slope"]),
        (k3, ["price_gaps", "availability_injuries", "trajectory"]),
        (k4, ["role_switch_cost", "style_fit_matched_transfer"]),
    ]:
        for key in keys:
            value = source.get(key)
            st.markdown(f"**{key}**")
            st.write(value if isinstance(value, str) else json.dumps(value)[:400])
