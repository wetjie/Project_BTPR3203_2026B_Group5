# Project_BTPR3203_2026B_Group5

# BTPR3203 Python for Data Science Project

## Project Title

**Analysing Skill-Related Underemployment Among Tertiary-Educated Workers in Malaysia**

---

## 1. Problem Statement

Malaysia's low unemployment rate does not necessarily mean that graduates are obtaining jobs that fully utilise their skills and education. Graduates may be employed while working in occupations that do not fully utilise their skill level. Therefore, employment status alone may not provide a complete picture of graduate employment outcomes.

Graduate underemployment may also differ across fields of study, regions, demographic groups, and educational qualification levels. This project therefore examines graduate underemployment in Malaysia from several perspectives using official and respondent-level datasets.

---

## 2. News Article Justification

The project is motivated by the selected news article:

**Hadi Azmi and Muzliza Mustafa (2026). _Malaysia's low unemployment masks deeper crisis of underemployed graduates_. Asianews.Network.**

The article highlights the gap between Malaysia's low unemployment rate and the continued presence of skill-related underemployment among tertiary-educated workers. This provides the real-world context for examining graduate employment quality rather than focusing only on whether graduates are employed.

---

## 3. Research Questions

### RQ1

**How many graduates in Malaysia are currently working in jobs that don't match their qualification level, and how has this number changed over time?**

RQ1 examines the scale and temporal pattern of skill-related underemployment using quarterly DOSM data.

### RQ2

**Which fields of study (or regions/states) have the highest rates of graduate underemployment in Malaysia?**

RQ2 examines differences in current occupation-based underemployment across fields of study and Malaysian states using the KRI GCTS dataset.

### RQ3

**What demographic and educational factors are associated with graduate underemployment in Malaysia?**

RQ3 examines the association between underemployment and:

- Sex
- Age group
- Qualification level

---

## 4. Datasets

### RQ1 — Department of Statistics Malaysia (DOSM)

RQ1 uses official quarterly skill-related underemployment datasets from the Department of Statistics Malaysia:

- `lfs_qtr_sru_age`
- `lfs_qtr_sru_sex`

These datasets provide skill-related underemployment information by age group and sex over time.

### RQ2 and RQ3 — Khazanah Research Institute (KRI)

RQ2 and RQ3 use the **Graduate Career Transition Survey (GCTS)** dataset provided by the Khazanah Research Institute.

The GCTS dataset contains respondent-level information including:

- Demographic characteristics
- Educational qualifications
- Employment status
- Current occupation
- Field of study
- Place of residence

---

## 5. Analytical Approach

The project follows a common data science pipeline:

```text
Data Loading
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Quantitative Analysis
      ↓
Visualisation
      ↓
Output Generation

RQ1

RQ1 analyses:

Overall underemployment trends
Age-group differences
Sex differences
Year-on-year changes
Peak values
Correlation between underemployment numbers and rates
Compound annual growth

Three visualisations are generated for the overall trend, age groups, and sex.

RQ2

RQ2 uses current occupation to operationalise skill-related underemployment.

The analysis examines:

Underemployment by field of study
Underemployment by Malaysian state
Field-of-study and state combinations

Three visualisations are generated, including field and state comparisons and a heatmap.

RQ3

RQ3 operationalises skill-related underemployment using current occupation.

The analysis includes:

Overall underemployment rate
Qualification-level comparison
Chi-square test for qualification
Chi-square test for age group
Chi-square test for sex
Cramér's V effect size
Multi-way descriptive analysis

Two visualisations are generated for age × qualification and sex × qualification.

6. Key Analytical Definitions

For the KRI GCTS analyses, current occupation is used to operationalise skill-related underemployment.

The following occupational groups are treated as high-skilled:

Managers
Professionals
Technicians and Associate Professionals

Respondents working outside these high-skilled occupational groups are classified as the skill-related underemployment proxy.

For RQ3, respondents are further grouped into:

Diploma
Bachelor's Degree

Age is grouped into:

23–26
27–31
32–44
7. Key Findings
RQ1

The analysis examines how the number and rate of skill-related underemployment among tertiary-educated workers changed over time and how patterns differ by age and sex.

RQ2

The analysis identifies differences in current occupation-based underemployment across fields of study and Malaysian states.

RQ3

Qualification level showed the strongest association with skill-related underemployment. Age group showed a statistically significant but very weak association, while sex was not significantly associated with underemployment.

The findings are interpreted as associations rather than causal effects.

8. Project Structure
BTPR3203_Project/
│
├── main.py
│
├── src/
│   ├── __init__.py
│   ├── rq1_analysis.py
│   ├── rq2_analysis.py
│   └── rq3_analysis.py
│
├── data/
│   ├── KRI_GCTS_Dataset.csv
│   ├── lfs_qtr_sru_age_raw.csv
│   └── lfs_qtr_sru_sex_raw.csv
│
├── outputs/
│   ├── rq1/
│   ├── rq2/
│   └── rq3/
│
├── figures/
│   ├── rq1/
│   ├── rq2/
│   └── rq3/
│
└── README.md
9. Python Libraries

The project uses:

Pandas — data loading, cleaning, transformation, and analysis
NumPy — numerical operations
Matplotlib — visualisation
Seaborn — statistical visualisation
SciPy — statistical testing
Pathlib — file and directory management
JSON — summary output
10. Setup Instructions
Install Python

Python 3.11 or later is recommended.

Install required libraries
pip install pandas numpy matplotlib seaborn scipy pyarrow
Dataset

Place the KRI GCTS dataset in:

data/KRI_GCTS_Dataset.csv

The DOSM datasets used in RQ1 are downloaded automatically when they are not already present in the data folder.

11. How to Run the Project
Run the complete project

From the project root directory:

python main.py

This runs:

RQ1
 ↓
RQ2
 ↓
RQ3
Run individual analyses
python src/rq1_analysis.py
python src/rq2_analysis.py
python src/rq3_analysis.py
12. Outputs
RQ1
outputs/rq1/
figures/rq1/

Contains processed data, summary JSON, trend tables, and three visualisations.

RQ2
outputs/rq2/
figures/rq2/

Contains field-level and state-level summaries, heatmap data, and three visualisations.

RQ3
outputs/rq3/
figures/rq3/

Contains the processed analytical dataset, demographic and education summary, and two visualisations.

13. Reproducibility

All major analytical procedures are implemented in Python source files. The project can be reproduced by running main.py, which executes the three research question modules sequentially.

The project also stores processed outputs and generated figures for reporting and presentation purposes.
