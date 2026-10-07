"""
IPL Dataset - Exploratory Data Analysis
=========================================
Analyzes the IPL `matches.csv` (match-level info) and `deliveries.csv`
(ball-by-ball info) datasets - the standard Kaggle IPL dataset.

Expects both CSV files in the same folder as this script, named:
    matches.csv
    deliveries.csv

(Download from: https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020
 or any similarly-structured IPL ball-by-ball dataset.)

Outputs:
    - Printed summary stats and insights to the console
    - Saved charts (.png) in an `eda_output/` folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
OUTPUT_DIR = "eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

print(f"matches.csv:    {matches.shape[0]} rows, {matches.shape[1]} columns")
print(f"deliveries.csv: {deliveries.shape[0]} rows, {deliveries.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. INITIAL INSPECTION
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. INITIAL INSPECTION")
print("=" * 60)

print("\n--- matches.csv info ---")
print(matches.info())
print("\n--- matches.csv head ---")
print(matches.head())

print("\n--- deliveries.csv info ---")
print(deliveries.info())
print("\n--- deliveries.csv head ---")
print(deliveries.head())

# ---------------------------------------------------------------------------
# 3. MISSING VALUES
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. MISSING VALUES")
print("=" * 60)

print("\nMissing values in matches.csv (top 10 columns):")
print(matches.isnull().sum().sort_values(ascending=False).head(10))

print("\nMissing values in deliveries.csv (top 10 columns):")
print(deliveries.isnull().sum().sort_values(ascending=False).head(10))

# ---------------------------------------------------------------------------
# 4. SEASON-WISE MATCH COUNT
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. SEASON-WISE ANALYSIS")
print("=" * 60)

season_col = "season" if "season" in matches.columns else None
if season_col:
    matches_per_season = matches[season_col].value_counts().sort_index()
    print("\nMatches played per season:")
    print(matches_per_season)

    fig, ax = plt.subplots(figsize=(10, 5))
    matches_per_season.plot(kind="bar", ax=ax, color="#378ADD")
    ax.set_title("Matches Played Per Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Number of Matches")
    save_fig(fig, "matches_per_season.png")

# ---------------------------------------------------------------------------
# 5. TEAM PERFORMANCE - MOST WINS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. TEAM PERFORMANCE")
print("=" * 60)

winner_col = "winner" if "winner" in matches.columns else None
if winner_col:
    top_teams = matches[winner_col].value_counts().head(10)
    print("\nTop 10 teams by total wins:")
    print(top_teams)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_teams.sort_values().plot(kind="barh", ax=ax, color="#1D9E75")
    ax.set_title("Top 10 Teams by Total Wins")
    ax.set_xlabel("Wins")
    save_fig(fig, "top_teams_by_wins.png")

# ---------------------------------------------------------------------------
# 6. TOSS ANALYSIS - DOES WINNING THE TOSS HELP?
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. TOSS ANALYSIS")
print("=" * 60)

if "toss_winner" in matches.columns and winner_col:
    matches["toss_match_won"] = matches["toss_winner"] == matches[winner_col]
    toss_win_pct = matches["toss_match_won"].mean() * 100
    print(f"\nTeams that won the toss also won the match: {toss_win_pct:.1f}% of the time")

    if "toss_decision" in matches.columns:
        decision_counts = matches["toss_decision"].value_counts()
        print("\nToss decision breakdown:")
        print(decision_counts)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(decision_counts, labels=decision_counts.index, autopct="%1.1f%%",
               colors=["#378ADD", "#D85A30"])
        ax.set_title("Toss Decision: Bat vs Field")
        save_fig(fig, "toss_decision_pie.png")

# ---------------------------------------------------------------------------
# 7. TOP RUN SCORERS (from deliveries.csv)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. TOP RUN SCORERS")
print("=" * 60)

batsman_col = "batter" if "batter" in deliveries.columns else "batsman"
runs_col = "batsman_runs"

if batsman_col in deliveries.columns and runs_col in deliveries.columns:
    top_batsmen = (
        deliveries.groupby(batsman_col)[runs_col]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nTop 10 run scorers (all seasons combined):")
    print(top_batsmen)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_batsmen.sort_values().plot(kind="barh", ax=ax, color="#F2B01E")
    ax.set_title("Top 10 Run Scorers (All Seasons)")
    ax.set_xlabel("Total Runs")
    save_fig(fig, "top_run_scorers.png")

# ---------------------------------------------------------------------------
# 8. TOP WICKET TAKERS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("8. TOP WICKET TAKERS")
print("=" * 60)

bowler_col = "bowler"
dismissal_col = "dismissal_kind" if "dismissal_kind" in deliveries.columns else "player_dismissed"

if bowler_col in deliveries.columns and "player_dismissed" in deliveries.columns:
    # Exclude dismissals not credited to the bowler (run out, retired hurt, etc.)
    excluded_kinds = {"run out", "retired hurt", "obstructing the field"}
    if "dismissal_kind" in deliveries.columns:
        wickets_df = deliveries[
            deliveries["player_dismissed"].notna()
            & ~deliveries["dismissal_kind"].isin(excluded_kinds)
        ]
    else:
        wickets_df = deliveries[deliveries["player_dismissed"].notna()]

    top_bowlers = wickets_df.groupby(bowler_col).size().sort_values(ascending=False).head(10)
    print("\nTop 10 wicket takers (all seasons combined):")
    print(top_bowlers)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_bowlers.sort_values().plot(kind="barh", ax=ax, color="#D85A30")
    ax.set_title("Top 10 Wicket Takers (All Seasons)")
    ax.set_xlabel("Wickets")
    save_fig(fig, "top_wicket_takers.png")

# ---------------------------------------------------------------------------
# 9. RUNS PER OVER - SCORING PATTERN ACROSS AN INNINGS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("9. SCORING PATTERN BY OVER")
print("=" * 60)

if "over" in deliveries.columns and "total_runs" in deliveries.columns:
    runs_per_over = deliveries.groupby("over")["total_runs"].mean()
    print("\nAverage runs scored per over (across all matches):")
    print(runs_per_over)

    fig, ax = plt.subplots(figsize=(10, 5))
    runs_per_over.plot(kind="line", marker="o", ax=ax, color="#378ADD")
    ax.set_title("Average Runs Scored Per Over")
    ax.set_xlabel("Over")
    ax.set_ylabel("Average Runs")
    save_fig(fig, "runs_per_over.png")

# ---------------------------------------------------------------------------
# 10. VENUE ANALYSIS - HIGHEST AVERAGE SCORING GROUNDS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("10. VENUE ANALYSIS")
print("=" * 60)

if "venue" in matches.columns and "id" in matches.columns and "total_runs" in deliveries.columns:
    match_id_col = "match_id" if "match_id" in deliveries.columns else "id"
    merged = deliveries.merge(
        matches[["id", "venue"]], left_on=match_id_col, right_on="id", how="left"
    )
    runs_per_match_venue = (
        merged.groupby([match_id_col, "venue"])["total_runs"].sum().reset_index()
    )
    avg_runs_by_venue = (
        runs_per_match_venue.groupby("venue")["total_runs"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nTop 10 venues by average total runs per match:")
    print(avg_runs_by_venue)

    fig, ax = plt.subplots(figsize=(10, 6))
    avg_runs_by_venue.sort_values().plot(kind="barh", ax=ax, color="#1D9E75")
    ax.set_title("Top 10 Highest-Scoring Venues (Avg Runs/Match)")
    ax.set_xlabel("Average Total Runs")
    save_fig(fig, "avg_runs_by_venue.png")

# ---------------------------------------------------------------------------
# 11. CORRELATION HEATMAP (numeric columns in deliveries)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("11. CORRELATION HEATMAP")
print("=" * 60)

numeric_cols = deliveries.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    corr = deliveries[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap - Deliveries Dataset (Numeric Columns)")
    save_fig(fig, "correlation_heatmap.png")

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)
print(f"All charts saved to the '{OUTPUT_DIR}/' folder.")
print("Review the console output above for the underlying numbers behind each chart.")
