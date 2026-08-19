"""
BTPR3203 Python for Data Science - Project (30%)
Title: Analysing Skill-Related Underemployment Among Tertiary-Educated Workers in Malaysia (2017-2026)

Complete analytical pipeline meeting all technical requirements:
- Data Loading and Preparation (Pandas)
- Exploratory Data Analysis
- Feature Engineering (>=1 derived measure)
- Quantitative Analysis (>=5 distinct operations)
- Visualisation (>=3 high-quality plots with interpretation)
- Reporting (save >=1 processed CSV/JSON)

Author: Group Project
Semester B, 2026
"""

def run_rq1():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import json
    import warnings
    warnings.filterwarnings("ignore")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    BASE_DIR = Path(__file__).resolve().parent.parent
    OUTPUT_DIR = BASE_DIR / "outputs" / "rq1"
    DATA_DIR = BASE_DIR / "data"
    FIGURE_DIR = BASE_DIR / "figures" / "rq1"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11

    # ------------------------------------------------------------------
    # 1. DATA LOADING AND PREPARATION
    # ------------------------------------------------------------------
    print("=" * 60)
    print("1. DATA LOADING AND PREPARATION")
    print("=" * 60)

    URL_AGE = "https://storage.dosm.gov.my/labour/lfs_qtr_sru_age.parquet"
    URL_SEX = "https://storage.dosm.gov.my/labour/lfs_qtr_sru_sex.parquet"

    AGE_RAW_FILE = DATA_DIR / "lfs_qtr_sru_age_raw.csv"
    SEX_RAW_FILE = DATA_DIR / "lfs_qtr_sru_sex_raw.csv"

    if AGE_RAW_FILE.exists() and SEX_RAW_FILE.exists():
        print("Existing DOSM datasets found. Loading from data folder...")
        df_age = pd.read_csv(AGE_RAW_FILE)
        df_sex = pd.read_csv(SEX_RAW_FILE)
    else:
        print("DOSM datasets not found. Downloading official datasets...")
        df_age = pd.read_parquet(URL_AGE)
        df_sex = pd.read_parquet(URL_SEX)
        df_age.to_csv(AGE_RAW_FILE, index=False)
        df_sex.to_csv(SEX_RAW_FILE, index=False)
        print(f"Raw data saved to {DATA_DIR}")

    df_age["date"] = pd.to_datetime(df_age["date"])
    df_sex["date"] = pd.to_datetime(df_sex["date"])

    print("\nMissing values check (age):", df_age.isna().sum().to_dict())
    print("Missing values check (sex):", df_sex.isna().sum().to_dict())

    df_age = df_age.dropna(subset=["sru"]).copy()
    df_sex = df_sex.dropna(subset=["sru"]).copy()

    df_age = df_age.rename(columns={"sru": "value"})
    df_sex = df_sex.rename(columns={"sru": "value"})

    print(f"\nAge series shape after cleaning: {df_age.shape}")
    print(f"Sex series shape after cleaning: {df_sex.shape}")
    print(f"Date range: {df_age['date'].min().date()} to {df_age['date'].max().date()}")

    # ------------------------------------------------------------------
    # 2. EXPLORATORY DATA ANALYSIS
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    age_persons = df_age[df_age["variable"] == "persons"].copy()
    age_rate    = df_age[df_age["variable"] == "rate"].copy()
    sex_persons = df_sex[df_sex["variable"] == "persons"].copy()
    sex_rate    = df_sex[df_sex["variable"] == "rate"].copy()

    print("\n--- Overall number of skill-related underemployed ('000) ---")
    overall_p_series = age_persons[age_persons["age"] == "overall"]["value"]
    print(overall_p_series.describe().round(1))

    print("\n--- Overall rate (%) ---")
    overall_r_series = age_rate[age_rate["age"] == "overall"]["value"]
    print(overall_r_series.describe().round(2))

    print("\n--- Age group distribution (latest quarter) ---")
    latest_date = age_persons["date"].max()
    print(age_persons[age_persons["date"] == latest_date][["age", "value"]].to_string(index=False))

    print("\n--- Sex distribution (latest quarter) ---")
    print(sex_persons[(sex_persons["date"] == sex_persons["date"].max()) & (sex_persons["sex"] != "overall")][["sex", "value"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 3. FEATURE ENGINEERING
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. FEATURE ENGINEERING")
    print("=" * 60)

    for d in [age_persons, age_rate, sex_persons, sex_rate]:
        d["year"] = d["date"].dt.year
        d["quarter"] = d["date"].dt.quarter
        d["year_q"] = d["date"].dt.to_period("Q").astype(str)

    overall_p = (age_persons[age_persons["age"] == "overall"]
                 [["date", "value"]].set_index("date").sort_index())
    overall_r = (age_rate[age_rate["age"] == "overall"]
                 [["date", "value"]].set_index("date").sort_index())

    # Year-on-year changes
    overall_p["yoy_change_000"] = overall_p["value"].diff(4)
    overall_p["yoy_pct_change"] = overall_p["value"].pct_change(4) * 100
    overall_r["yoy_pp_change"]  = overall_r["value"].diff(4)

    # Age-group shares
    pivot_age_num = age_persons.pivot(index="date", columns="age", values="value")
    for col in ["15-24", "25-34", "35-44", "45+"]:
        if col in pivot_age_num.columns:
            pivot_age_num[f"share_{col}"] = (pivot_age_num[col] / pivot_age_num["overall"] * 100).round(1)

    print("Created features:")
    print("  - year, quarter, year_q")
    print("  - yoy_change_000, yoy_pct_change (persons)")
    print("  - yoy_pp_change (rate, percentage points)")
    print("  - share_15-24, share_25-34, share_35-44, share_45+ (%)")

    # ------------------------------------------------------------------
    # 4. QUANTITATIVE ANALYSIS (>=5 distinct operations)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. QUANTITATIVE ANALYSIS")
    print("=" * 60)

    # Analysis 1
    print("\n[1] Overall trend (selected quarters)")
    trend = pd.DataFrame({
        "persons_000": overall_p["value"],
        "rate_pct": overall_r["value"],
        "yoy_000": overall_p["yoy_change_000"],
        "yoy_pct": overall_p["yoy_pct_change"]
    }).dropna(how="all")
    selected = trend.loc[trend.index.isin([
        pd.Timestamp("2018-01-01"),
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2022-01-01"),
        pd.Timestamp("2024-01-01"),
        trend.index[-1]
    ])]
    print(selected.round(1).to_string())

    # Analysis 2
    print("\n[2] Age-group numbers ('000) - earliest vs latest")
    earliest_date = age_persons["date"].min()
    age_num_comp = (age_persons[age_persons["date"].isin([earliest_date, latest_date])]
                    .pivot(index="age", columns="date", values="value")
                    .round(1))
    print(age_num_comp.to_string())

    # Analysis 3
    print("\n[3] Age-group rates (%) - earliest vs latest")
    age_rate_comp = (age_rate[age_rate["date"].isin([earliest_date, latest_date])]
                     .pivot(index="age", columns="date", values="value")
                     .round(1))
    print(age_rate_comp.to_string())

    # Analysis 4
    print("\n[4] Latest sex breakdown")
    sex_latest_p = sex_persons[(sex_persons["date"] == sex_persons["date"].max()) & (sex_persons["sex"] != "overall")][["sex", "value"]]
    sex_latest_r = sex_rate[(sex_rate["date"] == sex_rate["date"].max()) & (sex_rate["sex"] != "overall")][["sex", "value"]]
    print("Persons ('000):")
    print(sex_latest_p.set_index("sex").round(1).to_string())
    print("Rates (%):")
    print(sex_latest_r.set_index("sex").round(1).to_string())

    # Analysis 5
    corr = overall_p["value"].corr(overall_r["value"])
    print(f"\n[5] Pearson correlation (overall number vs rate): {corr:.3f}")

    # Analysis 6
    peak_num_idx = overall_p["value"].idxmax()
    peak_rate_idx = overall_r["value"].idxmax()
    print(f"\n[6] Peak absolute number: {overall_p.loc[peak_num_idx, 'value']:.1f} thousand on {peak_num_idx.date()}")
    print(f"    Peak rate:           {overall_r.loc[peak_rate_idx, 'value']:.1f}% on {peak_rate_idx.date()}")

    # Analysis 7
    first_val = overall_p.iloc[0]["value"]
    last_val = overall_p.iloc[-1]["value"]
    n_years = (overall_p.index[-1] - overall_p.index[0]).days / 365.25
    cagr = ((last_val / first_val) ** (1 / n_years) - 1) * 100
    print(f"\n[7] Compound annual growth rate of absolute numbers: {cagr:.2f}% per year")

    # ------------------------------------------------------------------
    # 5. VISUALISATIONS (>=3)  +  NEW PROJECTION CHART
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("5. VISUALISATIONS")
    print("=" * 60)

    # ----- Viz 1: Overall dual-axis trend (actual only) -----
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    color1 = "#2c7bb6"
    color2 = "#d7191c"
    ax1.plot(overall_p.index, overall_p["value"], color=color1, linewidth=2.2, label="Number ('000)")
    ax1.set_ylabel("Persons ('000)", color=color1, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(bottom=1000)
    ax2 = ax1.twinx()
    ax2.plot(overall_r.index, overall_r["value"], color=color2, linewidth=2.2, linestyle="--", label="Rate (%)")
    ax2.set_ylabel("Rate (%)", color=color2, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(25, 42)
    ax1.set_title("Skill-Related Underemployment of Tertiary-Educated Workers\nMalaysia, 2017-2025", fontweight="bold")
    ax1.set_xlabel("Quarter")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_overall_trend.png", bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved: figures/rq1/01_overall_trend.png")

    # ----- Viz 1b: Overall trend + simple linear projection to 2030 -----
    recent_p = overall_p[overall_p.index >= "2021-01-01"].copy()
    recent_r = overall_r[overall_r.index >= "2021-01-01"].copy()
    recent_p["t"] = np.arange(len(recent_p))
    recent_r["t"] = np.arange(len(recent_r))

    coef_p = np.polyfit(recent_p["t"], recent_p["value"], 1)
    coef_r = np.polyfit(recent_r["t"], recent_r["value"], 1)

    last_date = overall_p.index.max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=3),
        end="2030-10-01",
        freq="QS"
    )
    n_future = len(future_dates)
    t_future = np.arange(len(recent_p), len(recent_p) + n_future)

    pred_persons = coef_p[0] * t_future + coef_p[1]
    pred_rate = np.clip(coef_r[0] * t_future + coef_r[1], 30.0, 40.0)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(overall_p.index, overall_p["value"], color=color1, linewidth=2.4, label="Number ('000) – Actual")
    ax1.set_ylabel("Persons ('000)", color=color1, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(1000, 2300)

    ax2 = ax1.twinx()
    ax2.plot(overall_r.index, overall_r["value"], color=color2, linewidth=2.4, linestyle="--", label="Rate (%) – Actual")
    ax2.set_ylabel("Rate (%)", color=color2, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(26, 42)

    # Projection lines (dotted)
    ax1.plot(future_dates, pred_persons, color=color1, linewidth=2.2, linestyle=":", alpha=0.9, label="Number ('000) – Projection")
    ax2.plot(future_dates, pred_rate, color=color2, linewidth=2.2, linestyle=":", alpha=0.9, label="Rate (%) – Projection")

    ax1.axvline(last_date, color="gray", linestyle="-", linewidth=1.2, alpha=0.7)
    ax1.text(last_date, 2220, "  End of actual data", fontsize=9, color="gray", va="top")

    ax1.set_title("Skill-Related Underemployment of Tertiary-Educated Workers\nMalaysia, 2017–2030 (Actual + Simple Linear Projection)",
                  fontweight="bold", fontsize=13, pad=10)
    ax1.set_xlabel("Quarter")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9, framealpha=0.95)

    fig.text(0.5, 0.01,
             "Note: Projection is a simple linear trend fitted on 2021–mid-2025 data and extended to 2030. It is illustrative only and not an official forecast.",
             ha="center", fontsize=8.5, style="italic", color="#444444")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIGURE_DIR / "01b_overall_trend_with_projection.png", bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved: figures/rq1/01b_overall_trend_with_projection.png")
    print("  Interpretation: Actual data to mid-2025; dotted lines are a simple linear projection to 2030 based on the post-2021 trend. Illustrative only.")

    # ----- Viz 2: Age-group rates -----
    fig, ax = plt.subplots(figsize=(11, 5.5))
    palette = {"15-24": "#e41a1c", "25-34": "#377eb8", "35-44": "#4daf4a", "45+": "#984ea3"}
    for age_grp in ["15-24", "25-34", "35-44", "45+"]:
        sub = age_rate[age_rate["age"] == age_grp].sort_values("date")
        ax.plot(sub["date"], sub["value"], label=age_grp, linewidth=2.2, color=palette.get(age_grp))
    ax.set_title("Skill-Related Underemployment Rate by Age Group", fontweight="bold")
    ax.set_ylabel("Rate (%)")
    ax.set_xlabel("Quarter")
    ax.legend(title="Age group", loc="upper right")
    ax.set_ylim(15, 75)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_age_rates.png", bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved: figures/rq1/02_age_rates.png")

    # ----- Viz 3: Sex comparison -----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    latest_sex_p = sex_persons[(sex_persons["date"] == sex_persons["date"].max()) & (sex_persons["sex"] != "overall")]
    order = ["male", "female"]
    latest_sex_p = latest_sex_p.set_index("sex").reindex(order).reset_index()
    sns.barplot(data=latest_sex_p, x="sex", y="value", ax=axes[0], palette=["#377eb8", "#e41a1c"])
    axes[0].set_title("Latest Absolute Numbers by Sex ('000)", fontweight="bold")
    axes[0].set_ylabel("Persons ('000)")
    axes[0].set_xlabel("")
    for i, v in enumerate(latest_sex_p["value"]):
        axes[0].text(i, v + 20, f"{v:.0f}", ha="center", fontweight="bold")

    for s, c in zip(["male", "female"], ["#377eb8", "#e41a1c"]):
        sub = sex_rate[sex_rate["sex"] == s].sort_values("date")
        axes[1].plot(sub["date"], sub["value"], label=s, linewidth=2.2, color=c)
    axes[1].set_title("Rate by Sex Over Time", fontweight="bold")
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_xlabel("Quarter")
    axes[1].legend(title="Sex")
    axes[1].set_ylim(25, 42)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_sex_comparison.png", bbox_inches="tight")
    plt.show()
    plt.close()
    print("Saved: figures/rq1/03_sex_comparison.png")

    # ------------------------------------------------------------------
    # 6. SAVE PROCESSED OUTPUTS
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("6. SAVING PROCESSED OUTPUTS")
    print("=" * 60)

    processed = pd.concat([
        age_persons.assign(dimension="age"),
        age_rate.assign(dimension="age"),
        sex_persons.assign(dimension="sex"),
        sex_rate.assign(dimension="sex")
    ], ignore_index=True)

    processed_path = OUTPUT_DIR / "sru_tertiary_processed.csv"
    processed.to_csv(processed_path, index=False)
    print(f"Saved processed dataset: {processed_path} ({len(processed)} rows)")

    summary = {
        "project": "Skill-Related Underemployment - Tertiary Educated Workers, Malaysia",
        "latest_quarter": str(latest_date.date()),
        "overall_persons_000": round(float(overall_p.iloc[-1]["value"]), 1),
        "overall_rate_pct": round(float(overall_r.iloc[-1]["value"]), 1),
        "peak_persons_000": round(float(overall_p["value"].max()), 1),
        "peak_persons_date": str(peak_num_idx.date()),
        "peak_rate_pct": round(float(overall_r["value"].max()), 1),
        "peak_rate_date": str(peak_rate_idx.date()),
        "cagr_pct_per_year": round(cagr, 2),
        "correlation_number_vs_rate": round(corr, 3),
        "data_source": "DOSM Labour Force Survey - lfs_qtr_sru_age & lfs_qtr_sru_sex",
        "notes": "Skill-related underemployment = tertiary-educated persons in semi-skilled or low-skilled occupations (MASCO)."
    }
    summary_path = OUTPUT_DIR / "summary_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary metrics: {summary_path}")

    trend.round(2).to_csv(OUTPUT_DIR / "overall_trend_table.csv")
    print(f"Saved trend table: {OUTPUT_DIR / 'overall_trend_table.csv'}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"All outputs are in: {OUTPUT_DIR}")
    for p in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {p.name}")
    print(f"Figures are in: {FIGURE_DIR}")
    for p in sorted(FIGURE_DIR.glob("*")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    run_rq1()