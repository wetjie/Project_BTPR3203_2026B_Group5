def run_rq2():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    from pathlib import Path

    warnings.filterwarnings('ignore')
    sns.set_theme(style="whitegrid")

    # ============================= Configuration =============================
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "outputs" / "rq2"
    FIGURE_DIR = BASE_DIR / "figures" / "rq2"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ============================= Step 1: Load Dataset =============================
    FILE_PATH = DATA_DIR / "KRI_GCTS_Dataset.csv"

    df = pd.read_csv(FILE_PATH, low_memory=False)

    print("=== Original Dataset ===")
    print(f"Records: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    analysis_cols = [
        'FieldofStudy',
        'placeofresidence',
        'occupation_cj'
    ]

    missing_columns = [col for col in analysis_cols if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # 清理 99 / 99.0 缺失值编码
    for col in analysis_cols:
        df[col] = df[col].replace([99, 99.0, '99', '99.0'], np.nan)


    # ============================= Additional Step: Filter Currently Working Graduates =============================

    working_statuses = [
        'Working full-time (permanent)',
        'Working full-time (contract)',
        'Working part-time (permanent)',
        'Working part-time (contract)',
        'Self-employed (sole proprietor)',
        'Self-employed (gig)',
        'Self-employed (business owner)'
    ]

    is_working = df['currentempstatus'].isin(working_statuses)

    print("\n--- Working Graduate Filter ---")
    print(f"Working graduates retained: {is_working.sum()}")

    df = df[is_working].copy()


    # ============================= Step 2: Check Missing Values & Cleaning =============================

    missing_summary = df[analysis_cols].isnull().sum()
    missing_percentage = (missing_summary / len(df)) * 100

    missing_table = pd.DataFrame({
        'Missing Count': missing_summary,
        'Percentage (%)': missing_percentage
    })

    print("\n--- Initial Missing Value Summary ---")
    print(missing_table[missing_table['Missing Count'] > 0])

    # 剔除核心变量缺失的记录
    df_clean = df.dropna(subset=analysis_cols).copy()

    print("\n--- Post-Cleaning Dataset Summary ---")
    print(f"Original Records: {len(df)}")
    print(f"Cleaned Records: {len(df_clean)}")
    print(f"Records Removed: {len(df) - len(df_clean)}")
    print(f"Remaining Missing Values: {df_clean[analysis_cols].isnull().sum().sum()}")


    # ============================= Step 3: Feature Engineering =============================

    # 定义高技能职业组（与 RQ3 完全统一）
    high_skilled_occupations = [
        'Managers',
        'Professional',
        'Technician and Associate Professionals'
    ]

    # 当前职业非高技能者判定为 Underemployed Proxy (1)
    df_clean['Is_Underemployed'] = (
        ~df_clean['occupation_cj'].isin(high_skilled_occupations)
    ).astype(int)

    print("\n--- Cleaned Dataset Sample (First 10 Rows) ---")
    print(
        df_clean[
            ['FieldofStudy', 'placeofresidence', 'occupation_cj', 'Is_Underemployed']
        ].head(10)
    )

    print("\n--- Current Occupation Distribution ---")
    print(df_clean['occupation_cj'].value_counts())

    print("\n--- Overall Underemployment Summary ---")
    overall = df_clean['Is_Underemployed'].mean() * 100
    print(f"High-skilled Occupations: {(100 - overall):.2f}%")
    print(f"Non-High-skilled (Underemployed Proxy): {overall:.2f}%")


    # ============================= Finding 1: Field of Study =============================
    # 排除 'Field unknown'
    field_df = df_clean[
        df_clean['FieldofStudy'] != 'Field unknown'
    ].copy()

    field_stats = field_df.groupby('FieldofStudy')['Is_Underemployed'].agg(
        Total_Graduates='count',
        Underemployed_Count='sum',
        Underemployment_Rate='mean'
    ).reset_index()

    field_stats['Underemployment_Rate'] *= 100
    field_stats = field_stats.sort_values(
        by='Underemployment_Rate',
        ascending=False
    )

    highest_field = field_stats.iloc[0]
    lowest_field = field_stats.iloc[-1]
    rate_gap_field = (
        highest_field['Underemployment_Rate']
        - lowest_field['Underemployment_Rate']
    )

    print("\n=== FINDING 1 SUMMARY & KEY INSIGHTS ===")
    print(
        f"• Highest Underemployment Field: "
        f"{highest_field['FieldofStudy']} "
        f"({highest_field['Underemployment_Rate']:.1f}% | N={highest_field['Total_Graduates']})"
    )
    print(
        f"• Lowest Underemployment Field: "
        f"{lowest_field['FieldofStudy']} "
        f"({lowest_field['Underemployment_Rate']:.1f}% | N={lowest_field['Total_Graduates']})"
    )
    print(
        f"• Maximum Disparity Gap: "
        f"{rate_gap_field:.1f}% percentage points"
    )

    print("\nFull Field-level Summary:")
    print(field_stats.to_string(index=False))

    # Figure 1: Horizontal Bar Chart for Field of Study
    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(
        data=field_stats,
        x='Underemployment_Rate',
        y='FieldofStudy',
        hue='FieldofStudy',
        palette='Blues_r'
    )
    if ax1.get_legend():
        ax1.get_legend().remove()

    plt.title(
        'Figure 1: Current Occupation-Based Underemployment Rate by Field of Study',
        fontsize=12,
        fontweight='bold'
    )
    plt.xlabel('Underemployment Rate (%)')
    plt.ylabel('Field of Study')

    for p in ax1.patches:
        val = p.get_width()
        if val > 0:
            ax1.annotate(
                f"{val:.1f}%",
                (val + 0.8, p.get_y() + p.get_height() / 2.),
                ha='left',
                va='center',
                fontsize=9
            )

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'fig1_field_underemployment.png', dpi=300)
    plt.show()


    # ============================= Finding 2: Geographic Disparities =============================
    # 排除 'Overseas'
    state_df = df_clean[
        df_clean['placeofresidence'] != 'Overseas'
    ].copy()

    state_stats = state_df.groupby(
        'placeofresidence'
    )['Is_Underemployed'].agg(
        Total_Graduates='count',
        Underemployed_Count='sum',
        Underemployment_Rate='mean'
    ).reset_index()

    state_stats['Underemployment_Rate'] *= 100
    state_stats = state_stats.sort_values(
        by='Underemployment_Rate',
        ascending=False
    )

    highest_state = state_stats.iloc[0]
    lowest_state = state_stats.iloc[-1]
    rate_gap_state = (
        highest_state['Underemployment_Rate']
        - lowest_state['Underemployment_Rate']
    )

    print("\n=== FINDING 2 SUMMARY & KEY INSIGHTS ===")
    print(
        f"• Highest Underemployment State: "
        f"{highest_state['placeofresidence']} "
        f"({highest_state['Underemployment_Rate']:.1f}% | N={highest_state['Total_Graduates']})"
    )
    print(
        f"• Lowest Underemployment State: "
        f"{lowest_state['placeofresidence']} "
        f"({lowest_state['Underemployment_Rate']:.1f}% | N={lowest_state['Total_Graduates']})"
    )
    print(
        f"• Regional Disparity Gap: "
        f"{rate_gap_state:.1f}% percentage points"
    )

    print("\nFull State-level Summary:")
    print(state_stats.to_string(index=False))

    # Figure 2: Bar Chart for Malaysian States
    plt.figure(figsize=(12, 6))
    ax2 = sns.barplot(
        data=state_stats,
        x='placeofresidence',
        y='Underemployment_Rate',
        hue='placeofresidence',
        palette='Reds_r'
    )
    if ax2.get_legend():
        ax2.get_legend().remove()

    plt.title(
        'Figure 2: Current Occupation-Based Underemployment Rate Across Malaysian States',
        fontsize=12,
        fontweight='bold'
    )
    plt.xlabel('State / Region')
    plt.ylabel('Underemployment Rate (%)')
    plt.xticks(rotation=45, ha='right')

    for p in ax2.patches:
        val = p.get_height()
        if val > 0:
            ax2.annotate(
                f"{val:.1f}%",
                (p.get_x() + p.get_width() / 2., val + 0.8),
                ha='center',
                va='bottom',
                fontsize=8.5
            )

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'fig2_state_underemployment.png', dpi=300)
    plt.show()


    # ============================= Finding 3: Field vs State Heatmap =============================
    heatmap_df = df_clean[
        (df_clean['FieldofStudy'] != 'Field unknown') &
        (df_clean['placeofresidence'] != 'Overseas')
    ].copy()

    pivot_heatmap = pd.crosstab(
        heatmap_df['FieldofStudy'],
        heatmap_df['placeofresidence'],
        values=heatmap_df['Is_Underemployed'],
        aggfunc='mean'
    ) * 100

    max_vulnerability = pivot_heatmap.max().max()

    print(
        f"\nPeak Underemployment Concentration Rate: "
        f"{max_vulnerability:.2f}%"
    )

    # Figure 3: Heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        pivot_heatmap,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={'label': 'Underemployment Rate (%)'}
    )

    plt.title(
        'Figure 3: Underemployment Rate by Field of Study and State',
        fontsize=12,
        fontweight='bold'
    )
    plt.xlabel('State')
    plt.ylabel('Field of Study')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'fig3_heatmap_field_state.png', dpi=300)
    plt.show()


    # ============================= Reporting =============================

    field_stats.to_csv(
        OUTPUT_DIR / 'field_underemployment_summary.csv',
        index=False
    )

    state_stats.to_csv(
        OUTPUT_DIR / 'state_underemployment_summary.csv',
        index=False
    )

    pivot_heatmap.to_csv(
        OUTPUT_DIR / 'field_state_heatmap.csv'
    )

    print("\n--- RQ2 Outputs ---")
    print(f"Field summary saved to: {OUTPUT_DIR / 'field_underemployment_summary.csv'}")
    print(f"State summary saved to: {OUTPUT_DIR / 'state_underemployment_summary.csv'}")
    print(f"Heatmap data saved to: {OUTPUT_DIR / 'field_state_heatmap.csv'}")


if __name__ == "__main__":
    run_rq2()