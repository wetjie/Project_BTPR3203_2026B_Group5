# Project_BTPR3203_2026B_Group5

# Analysing Skill-Related Underemployment Among Tertiary-Educated Workers in Malaysia

**Course:** BTPR3203 Python for Data Science  
**Semester:** B 2026  
**Project Type:** Group Project

---

## 1. Project Overview

This project analyses skill-related underemployment among tertiary-educated workers in Malaysia using Python and official Malaysian datasets. The project investigates graduate underemployment through three research questions, with each research question following a complete analytical pipeline consisting of data preparation, exploratory data analysis (EDA), feature engineering, quantitative analysis, visualisation, interpretation, and reporting.

---

## 2. Research Questions

### RQ1 — Overall Trend

**Research Question:** What is the present state of graduates in Malaysia who are employed in jobs that are not aligned with their qualifications, and how has this situation changed over the years?

This analysis examines the national trend of skill-related underemployment among tertiary-educated workers, including changes over time, age-group patterns, sex patterns, and an illustrative long-term projection.

### RQ2 — Field and Regional Comparison

**Research Question:** Which fields of study (or regions/states) have the highest rates of graduate underemployment in Malaysia?

This analysis compares graduate underemployment across fields of study and Malaysian states or regions, and examines field–region patterns using descriptive analysis and visualisation.

### RQ3 — Demographic and Educational Factors

**Research Question:** What demographic and educational factors are associated with graduate underemployment in Malaysia?

This analysis investigates graduate underemployment based on:

- Sex
- Age group
- Qualification level

---

## 3. Datasets

### RQ1 Dataset

| Item | Description |
|------|-------------|
| **Source** | Department of Statistics Malaysia (DOSM) |
| **Dataset** | Labour Force Survey – Skill-Related Underemployment |
| **Coverage** | Quarterly data from **2017 Q1 to 2025 Q3** |
| **Measures** | Number of underemployed tertiary-educated workers ('000) and underemployment rate (%) |

### RQ2 Dataset

| Item | Description |
|------|-------------|
| **Source** | Khazanah Research Institute (KRI) |
| **Dataset** | Graduate Career Transition Survey (GCTS) |
| **Coverage** | **8,026 respondents** and **221 variables** |
| **Graduate Cohorts** | Approximately **2010** and **2018** |

### RQ3 Dataset

| Item | Description |
|------|-------------|
| **Source** | Khazanah Research Institute (KRI) |
| **Dataset** | Graduate Career Transition Survey (GCTS) |
| **Coverage** | **8,026 respondents** and **221 variables** |
| **Graduate Cohorts** | Approximately **2010** and **2018** |
| **Final Analytical Sample** | **4,631 working graduates** |

**Key Variables**

- `Sex`
- `age`
- `max_educert`
- `currentempstatus`
- `occupation_cj`

---

## 4. Project Structure

```text
project/
├── data/
│   ├── KRI_GCTS_Dataset.csv
│   └── DOSM source/raw data files
│
├── figures/
│   ├── rq1/
│   ├── rq2/
│   └── rq3/
│
├── outputs/
│   ├── rq1/
│   ├── rq2/
│   └── rq3/
│
├── src/
│   ├── rq1.py
│   ├── rq2.py
│   └── rq3.py
│
└── README.md
```

The project is organised by research question, with separate folders for datasets, figures, outputs, and Python source files.

---

## 5. Analytical Pipeline

RQ1
```text
Dataset
   ↓
Data Loading and Preparation
   ↓
Exploratory Data Analysis (EDA)
   ↓
Feature Engineering
   ↓
Quantitative Analysis
   ↓
Machine Learning
   ↓
Visualisation and Reporting
```

RQ2
```text
Dataset
   ↓
Data Loading and Preparation
   ↓
Exploratory Data Analysis (EDA)
   ↓
Feature Engineering
   ↓
Quantitative Analysis
   ↓
Methodological Alternatives
   ↓
Visualisation and Reporting
```

RQ3
```text
Dataset
   ↓
Data Loading and Preparation
   ↓
Exploratory Data Analysis (EDA)
   ↓
Feature Engineering
   ↓
Quantitative Analysis
   ↓
Methodological Alternatives
   ↓
Visualisation and Reporting
```

---

## 6. Methodology

### RQ1 — Overall Trend Analysis

**Objective**

Analyse the national trend of skill-related underemployment among tertiary-educated workers in Malaysia.

**Analytical Workflow**

1. Load the DOSM quarterly underemployment datasets using Pandas.
2. Clean missing values and convert dates into datetime format.
3. Separate person-count and percentage-rate series.
4. Create year, quarter, year-quarter, percentage change, percentage-point change, and age-share features.
5. Perform trend analysis, age-group analysis, sex comparison, correlation analysis, peak-value analysis, and CAGR analysis.
6. Generate trend and comparison visualisations.
7. Export processed CSV files, JSON summary metrics, and trend tables.

**Outputs**

- Overall trend chart.
- Illustrative projection chart.
- Age-group comparison chart.
- Sex comparison chart.
- CSV, JSON, and trend summary outputs.

> The projection is an **illustrative simple linear trend** and is not an official forecast.

---

### RQ2 — Field and Regional Analysis

**Objective**

Compare graduate underemployment across fields of study and Malaysian states or regions.

**Analytical Workflow**

1. Load the GCTS respondent-level dataset using Pandas.
2. Remove missing and invalid records, including coded values such as `99` and `99.0`.
3. Create a binary skill-related underemployment indicator based on occupational skill categories.
4. Group graduates by field of study and state or region.
5. Perform descriptive comparisons and field–region cross-tabulation.
6. Generate comparison bar charts and a heatmap.
7. Export processed analytical tables for reporting.

**Outputs**

- Field-of-study comparison chart.
- State or regional comparison chart.
- Field–region heatmap.
- CSV summary tables.

---

### RQ3 — Demographic and Educational Analysis

**Objective**

Examine demographic and educational factors associated with graduate skill-related underemployment.

**Analytical Workflow**

1. Load the GCTS dataset and retain only working graduates.
2. Remove respondents with missing or invalid occupation records.
3. Restrict the analysis to Diploma/Certificate and Bachelor's Degree graduates.
4. Create three derived variables: `is_underemployed`, `qualification_tier`, and `age_group`.
5. Calculate the overall underemployment rate and compare qualification groups.
6. Perform Chi-square tests for qualification, age group, and sex, and calculate Cramér's V.
7. Generate demographic comparison visualisations and export processed analytical outputs.

**Outputs**

- Age × Qualification bar chart.
- Sex × Qualification bar chart.
- Processed analytical dataset.
- Summary CSV tables.

---

## 7. Key Findings

### RQ1

- Skill-related underemployment increased substantially over the study period.
- The underemployment rate peaked during the COVID-19 period before moderating.
- Younger age groups generally recorded higher underemployment rates than older groups.

### RQ2

- General Programmes, Arts and Humanities, and Agriculture-related fields recorded higher observed underemployment rates.
- Geographic differences were observed across Malaysian states.
- Several field–region combinations showed particularly high descriptive underemployment rates.

### RQ3

- Diploma graduates consistently recorded higher underemployment rates than Bachelor's Degree graduates across age groups and sex groups.
- Qualification level showed the strongest association with underemployment.
- Age group showed a statistically significant but weak association.
- Sex was not significantly associated with underemployment.

---

## 8. Visualisation Outputs

The project generates multiple visualisations for each research question.

### RQ1

- Overall trend line chart.
- Illustrative projection chart.
- Age-group trend chart.
- Sex comparison chart.

### RQ2

- Field-of-study comparison chart.
- State or regional comparison chart.
- Field–region heatmap.

### RQ3

- Underemployment by age group and qualification level.
- Underemployment by sex and qualification level.

Each visualisation supports the interpretation of the corresponding analytical findings presented in the report.

---

## 9. Output Files

The project exports processed datasets and summary outputs for reporting and reuse.

### RQ1

- Processed CSV files.
- JSON summary metrics.
- Trend summary tables.

### RQ2

- Processed CSV files.
- Field and regional summary tables.

### RQ3

- `processed_gcts_analytical_data.csv`
- `individual_demographic_education_summary.csv`

PNG figures for all research questions are saved in the `figures/` directory.

---

## 10. Python Libraries

The project uses the following Python libraries.

| Library | Purpose |
|---------|---------|
| **Pandas** | Data loading, cleaning, grouping, and aggregation. |
| **NumPy** | Numerical calculations and feature creation. |
| **Matplotlib** | Data visualisation and figure generation. |
| **Seaborn** | Statistical visualisations. |
| **SciPy** | Chi-square tests and Cramér's V calculations. |
| **Pathlib / os** | File and directory management. |
| **JSON** | Export summary metrics for RQ1. |

---

## 11. How to Run

### Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scipy pyarrow
```

### Run Individual Research Question Scripts

```bash
python rq1.py
python rq2.py
python rq3.py
```

Generated figures will be saved in the `figures/` directory, while processed datasets and summary outputs will be saved in the `outputs/` directory.

---

## 12. Interpretation and Limitations

The findings are interpreted as **observed patterns and statistical associations rather than causal effects**.

### RQ1

- Uses national DOSM aggregate data.
- Projection is an illustrative linear trend rather than an official forecast.

### RQ2

- Uses respondent-level GCTS data to compare fields of study and regions.
- Results describe observed differences across groups rather than causal relationships.

### RQ3

- Uses occupation-based operational measures of skill-related underemployment.
- Excludes Master's and Doctoral graduates from qualification-level comparisons.
- Does not capture field-of-study mismatch or other forms of underemployment.

---

## 13. References

The project uses official Malaysian datasets and supporting literature, including:

- Department of Statistics Malaysia (DOSM) — Labour Force Survey skill-related underemployment datasets.
- Khazanah Research Institute (KRI) — Graduate Career Transition Survey (GCTS).
- Azmi, H., & Mustafa, M. (2026). *Malaysia's low unemployment masks deeper crisis of underemployed graduates.*

For complete APA references, refer to the final project report.

