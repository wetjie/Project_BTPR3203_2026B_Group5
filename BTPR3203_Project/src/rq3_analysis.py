# Research Question 3: What demographic and educational factors are associated
#                    with graduate underemployment in Malaysia?
def run_rq3():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    from pathlib import Path

    # ==============================================================================
    # Configuration
    # ==============================================================================
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "outputs" / "rq3"
    FIGURE_DIR = BASE_DIR / "figures" / "rq3"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ==============================================================================
    # STAGE 3.1: DATA LOADING AND PREPARATION
    # ==============================================================================
    # 1. Load microdata
    file_path = DATA_DIR / 'KRI_GCTS_Dataset.csv'
    raw_df = pd.read_csv(file_path, low_memory=False)

    print("=" * 75)
    print("STAGE 3.1: DATA LOADING & CLEANING")
    print(f"Total raw records loaded: {len(raw_df)}")

    # 2. Filter for active working graduates using exact matching
    working_statuses = [
        'Working full-time (permanent)',
        'Working full-time (contract)',
        'Working part-time (permanent)',
        'Working part-time (contract)',
        'Self-employed (sole proprietor)',
        'Self-employed (gig)',
        'Self-employed (business owner)'
    ]
    is_working = raw_df['currentempstatus'].isin(working_statuses)

    # 3. Explicit missing and invalid occupation filtering (excluding '99.0' / 99 / NaN)
    valid_occupation = (
        raw_df['occupation_cj'].notna() &
        ~raw_df['occupation_cj'].isin(['99.0', 99, 99.0])
    )
    clean_working_df = raw_df[is_working & valid_occupation].copy()

    print(f"Total valid working respondents retained: {len(clean_working_df)}")
    print("=" * 75)


    # ==============================================================================
    # STAGE 3.2: EXPLORATORY DATA ANALYSIS (EDA)
    # ==============================================================================
    print("\n" + "=" * 75)
    print("STAGE 3.2: EXPLORATORY DATA ANALYSIS (EDA)")

    # 1. Missing value audit across analytical target variables
    missing_check = clean_working_df[['age', 'Sex', 'max_educert', 'occupation_cj']].isna().sum()
    print(f"[EDA 1] Missing Values in Key Features:\n{missing_check.to_string()}")

    # 2. Age distribution summary
    age_stats = clean_working_df['age'].describe().round(2)
    print(f"\n[EDA 2] Age Distribution Summary (Min: {age_stats['min']}, Max: {age_stats['max']}, Median: {clean_working_df['age'].median()}):")
    print(age_stats.to_string())

    # 3. Gender composition breakdown
    gender_dist = clean_working_df['Sex'].value_counts(normalize=True).round(4) * 100
    print(f"\n[EDA 3] Gender Proportions (%):\n{gender_dist.to_string()}")

    # 4. Top current occupations
    print(f"\n[EDA 4] Top 5 Current Occupations:\n{clean_working_df['occupation_cj'].value_counts().head(5).to_string()}")
    print("=" * 75)


    # ==============================================================================
    # STAGE 3.3: FEATURE ENGINEERING
    # ==============================================================================
    # Feature 1: Operational definition of skill-related underemployment:
    # - Respondents in high-skilled occupations (MASCO Groups 1-3) are coded as 0.
    # - Respondents outside high-skilled occupational groups (MASCO Groups 4-9) are coded as 1.
    high_skilled_occupations = [
        'Managers', 
        'Professional', 
        'Technician and Associate Professionals'
    ]
    clean_working_df['is_underemployed'] = clean_working_df['occupation_cj'].apply(
        lambda x: 0 if x in high_skilled_occupations else 1
    )

    # Feature 2: Exact Qualification Mapping (Diploma vs. Bachelor's Degree)
    # Analysis focuses on Diploma and Bachelor's Degree levels to provide a direct undergraduate comparison
    qualification_map = {
        'Diploma/ certificates or equivalent': 'Diploma',
        "Bachelor's degree/ advanced diploma or equivalent": "Bachelor's Degree"
    }
    clean_working_df['qualification_tier'] = clean_working_df['max_educert'].map(qualification_map)
    analysis_df = clean_working_df[clean_working_df['qualification_tier'].notna()].copy()

    # Feature 3: Objective Age Cohort Categorisation (Data Range: 23-44)
    analysis_df['age_group'] = pd.cut(
        analysis_df['age'],
        bins=[22, 26, 31, 44],
        labels=['23-26', '27-31', '32-44'],
        include_lowest=True
    )

    print("\n" + "=" * 75)
    print("STAGE 3.3: FEATURE ENGINEERING COMPLETED")
    print("Engineered Features: `is_underemployed`, `qualification_tier`, `age_group`")
    print(f"Clean Analytical Sample Size: N = {len(analysis_df)}")
    print("=" * 75)


    # ==============================================================================
    # STAGE 3.4: QUANTITATIVE ANALYSIS & EFFECT SIZE (CRAMÉR'S V)
    # ==============================================================================
    print("\n" + "=" * 75)
    print("STAGE 3.4: QUANTITATIVE STATISTICAL ANALYSIS")

    # Helper function: Calculate bias-corrected Cramér's V for association strength
    def calculate_cramers_v(contingency_matrix):
        chi2_val = stats.chi2_contingency(contingency_matrix)[0]
        n_obs = contingency_matrix.sum().sum()
        phi2 = chi2_val / n_obs
        r_dim, c_dim = contingency_matrix.shape
        phi2corr = max(0, phi2 - ((c_dim - 1) * (r_dim - 1)) / (n_obs - 1))
        rcorr = r_dim - ((r_dim - 1) ** 2) / (n_obs - 1)
        ccorr = c_dim - ((c_dim - 1) ** 2) / (n_obs - 1)
        return np.sqrt(phi2corr / min((ccorr - 1), (rcorr - 1)))

    # Operation 1: Overall Baseline Underemployment Rate
    total_n = len(analysis_df)
    total_underemployed = analysis_df['is_underemployed'].sum()
    overall_sru_rate = (total_underemployed / total_n) * 100
    print(f"\n[Operation 1] Overall Baseline Underemployment Rate: {overall_sru_rate:.2f}% ({total_underemployed}/{total_n})")

    # Operation 2: Educational Disparity (Diploma vs. Bachelor's Degree)
    qual_summary = analysis_df.groupby('qualification_tier')['is_underemployed'].agg(
        Total='count',
        Underemployed='sum',
        SRU_Rate_Pct=lambda x: round(x.mean() * 100, 2)
    ).reset_index()
    print(f"\n[Operation 2] Underemployment by Qualification Tier:\n{qual_summary.to_string(index=False)}")

    # Operation 3: Chi-Square Test & Cramér's V (Qualification vs. Underemployment)
    ct_qual = pd.crosstab(analysis_df['qualification_tier'], analysis_df['is_underemployed'])
    chi2_q, p_q, dof_q, _ = stats.chi2_contingency(ct_qual)
    v_qual = calculate_cramers_v(ct_qual)
    print(f"\n[Operation 3] Chi-Square Test (Qualification vs Underemployment):")
    print(f"   Chi2 Stat = {chi2_q:.2f}, p-value = {p_q:.4e}, DoF = {dof_q}, Cramér's V = {v_qual:.4f}")

    # Operation 4: Chi-Square Test & Cramér's V (Age Cohort vs. Underemployment)
    ct_age = pd.crosstab(analysis_df['age_group'], analysis_df['is_underemployed'])
    chi2_a, p_a, dof_a, _ = stats.chi2_contingency(ct_age)
    v_age = calculate_cramers_v(ct_age)
    print(f"\n[Operation 4] Chi-Square Test (Age Cohort vs Underemployment):")
    print(f"   Chi2 Stat = {chi2_a:.2f}, p-value = {p_a:.4e}, DoF = {dof_a}, Cramér's V = {v_age:.4f}")

    # Operation 5: Chi-Square Test & Cramér's V (Gender vs. Underemployment)
    ct_sex = pd.crosstab(analysis_df['Sex'], analysis_df['is_underemployed'])
    chi2_s, p_s, dof_s, _ = stats.chi2_contingency(ct_sex)
    v_sex = calculate_cramers_v(ct_sex)
    print(f"\n[Operation 5] Chi-Square Test (Gender vs Underemployment):")
    print(f"   Chi2 Stat = {chi2_s:.2f}, p-value = {p_s:.4f}, DoF = {dof_s}, Cramér's V = {v_sex:.4f}")

    # Descriptive Multi-way Breakdown across Demographics & Qualification
    multi_table = analysis_df.groupby(['qualification_tier', 'Sex', 'age_group'], observed=False)['is_underemployed'].agg(
        Sample_Size='count',
        Underemployed_Count='sum',
        Rate_Pct=lambda x: round(x.mean() * 100, 2)
    ).reset_index()
    print(f"\n[Descriptive Breakdown] Underemployment across Demographic & Qualification Groups:\n{multi_table.to_string(index=False)}")
    print("=" * 75)


    # ==============================================================================
    # STAGE 3.5: DATA VISUALISATION (SEPARATE FIGURES)
    # ==============================================================================
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

    # ------------------------------------------------------------------------------
    # Figure 1: Underemployment Rate by Age Bracket & Qualification
    # ------------------------------------------------------------------------------
    plt.figure(figsize=(7, 4.8))
    sns.barplot(
        data=analysis_df,
        x='age_group',
        y='is_underemployed',
        hue='qualification_tier',
        errorbar=None,
        palette=['#2980b9', '#e67e22']
    )
    plt.title('Figure 1: Underemployment Rate by Age Bracket & Qualification', fontsize=11, fontweight='bold')
    plt.xlabel('Age Cohort', fontsize=10, fontweight='bold')
    plt.ylabel('Skill-Related Underemployment Rate', fontsize=10, fontweight='bold')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.legend(title='Qualification Level', title_fontsize='10', fontsize='9')
    plt.ylim(0, 0.55)
    plt.tight_layout()

    fig1_filename = FIGURE_DIR / 'figure1_age_qualification.png'
    plt.savefig(fig1_filename, dpi=300)
    plt.show()
    print(f"\n[Visualisation] Figure 1 saved to '{fig1_filename}'")

    # ------------------------------------------------------------------------------
    # Figure 2: Underemployment Rate by Gender & Qualification
    # ------------------------------------------------------------------------------
    plt.figure(figsize=(7, 4.8))
    sns.barplot(
        data=analysis_df,
        x='Sex',
        y='is_underemployed',
        hue='qualification_tier',
        errorbar=None,
        palette=['#2980b9', '#e67e22']
    )
    plt.title('Figure 2: Underemployment Rate by Gender & Qualification', fontsize=11, fontweight='bold')
    plt.xlabel('Gender', fontsize=10, fontweight='bold')
    plt.ylabel('Skill-Related Underemployment Rate', fontsize=10, fontweight='bold')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.legend(title='Qualification Level', title_fontsize='10', fontsize='9')
    plt.ylim(0, 0.55)
    plt.tight_layout()

    fig2_filename = FIGURE_DIR / 'figure2_gender_qualification.png'
    plt.savefig(fig2_filename, dpi=300)
    plt.show()
    print(f"[Visualisation] Figure 2 saved to '{fig2_filename}'")


    # ==============================================================================
    # STAGE 3.6: REPORTING & ARTIFACT EXPORT
    # ==============================================================================
    # 1. Export Cleaned & Processed Analytical Dataset (Record-level, N = 4,631)
    processed_columns = [
        'RespondentID',
        'Sex',
        'age',
        'age_group',
        'max_educert',
        'qualification_tier',
        'currentempstatus',
        'occupation_cj',
        'is_underemployed'
    ]
    processed_csv_file = OUTPUT_DIR / 'processed_gcts_analytical_data.csv'
    analysis_df[processed_columns].to_csv(processed_csv_file, index=False)

    # 2. Export Summarized Analytical Output Table
    summary_export = analysis_df.groupby(['qualification_tier', 'Sex', 'age_group'], observed=False).agg(
        total_graduates=('RespondentID', 'count'),
        underemployed_graduates=('is_underemployed', 'sum'),
        underemployment_rate_pct=('is_underemployed', lambda x: round(x.mean() * 100, 2))
    ).reset_index()

    summary_csv_file = OUTPUT_DIR / 'individual_demographic_education_summary.csv'
    summary_export.to_csv(summary_csv_file, index=False)

    print(f"\n[Reporting] Processed dataset (record-level) exported to: '{processed_csv_file}'")
    print(f"[Reporting] Summarised breakdown table exported to:     '{summary_csv_file}'")
    print("=" * 75)


if __name__ == "__main__":
    run_rq3()