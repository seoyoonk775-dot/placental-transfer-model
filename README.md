# placental-transfer-model
Machine learning models for quantitative prediction and classification of human placental drug transfer

# Placental Drug Transfer Model

Machine learning models for quantitative prediction and classification of human placental drug transfer.

This repository provides the data and reproducible code associated with the study:

**Integrated machine learning framework for quantitative prediction and classification of human placental drug transfer**

The repository contains two main reproducibility workflows:

1. Reconstruction of the final CI and F/M prediction models
2. LLM-based classification of FDA drug labels for external evaluation

---

## Repository Structure

```text
placental-transfer-model/
│
├── data/
│   └── Supplementary Dataset S1.xlsx
│
├── code/
│   ├── 01_build_final_models.py
│   └── 02_llm_fda_labeling.py
│
├── selected_xml.zip
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 1. Reconstruction of the Final CI and F/M Models

### `01_build_final_models.py`

This script reconstructs the final machine learning models used to predict two quantitative measures of human placental drug transfer:

- **Clearance Index (CI)**
- **Fetal-to-maternal concentration ratio (F/M)**

The final training matrices are provided in:

```text
data/Supplementary Dataset S1.xlsx
```

The workbook contains two sheets:

```text
CI_final_training_data
FM_final_training_data
```

### Final CI Model

- Algorithm: Gradient Boosting Regression Trees (GBRT)
- Training compounds: 93
- Final features: 30
- Missing-value handling: median imputation
- Hyperparameters: fixed according to the final model used in the study

### Final F/M Model

- Algorithm: Random Forest
- Training compounds: 117
- Final features: 91
- Missing-value handling: median imputation
- Hyperparameters: fixed according to the final model used in the study

The script reconstructs the final CI and F/M prediction models using the provided training matrices and saves the fitted model pipelines for subsequent prediction.

The saved pipelines include the preprocessing procedure required for prediction. Therefore, new compounds can be supplied using the same 30 CI features or 91 F/M features, and missing feature values can be handled using the fitted median-imputation procedure.

### Run

From the repository root:

```bash
python code/01_build_final_models.py
```

The reconstructed models are saved under:

```text
results/final_models/
```

---

## 2. LLM-Based FDA Drug-Label Classification

### `02_llm_fda_labeling.py`

This script reproduces the large-language-model-based FDA drug-label analysis used to construct the external placental-transfer reference dataset.

The input consists of **2,836 curated FDA Structured Product Labeling (SPL) XML files** provided in:

```text
selected_xml.zip
```

The script can also use an extracted directory named:

```text
selected_xml/
```

---

## LLM Workflow

The FDA label analysis consists of two stages.

### Stage 1: Evidence Extraction

Relevant evidence related to placental drug transfer is extracted from each FDA label.

The extracted evidence may include information related to:

- placental passage
- fetal exposure
- pregnancy pharmacokinetics
- maternal and fetal concentrations
- cord blood concentrations
- transplacental transfer
- other statements relevant to fetal drug exposure

Only evidence that can be matched back to the source FDA label text is retained.

### Stage 2: Placental-Transfer Classification

Using only the evidence extracted in Stage 1, each FDA label is classified into one of three categories:

```text
crosses
no_cross
unknown
```

Labels classified as `unknown` are excluded from the final reference set.

The workflow used in the study was:

```text
2,836 curated FDA labels
        ↓
Stage 1: evidence extraction
        ↓
Stage 2: placental-transfer classification
        ↓
crosses / no_cross / unknown
        ↓
unknown excluded
        ↓
610 retained FDA labels
```

The historical analysis retained **610 non-unknown FDA labels**.

---

## LLM Settings

The original analysis used:

```text
Provider: Groq API
Model: llama-3.3-70b-versatile
Temperature: 0
Maximum output tokens: 4096
```

A personal Groq API key is required to rerun the LLM analysis.

The API key is **not included in this repository**.

Users can either define the following environment variable:

```text
GROQ_API_KEY
```

or enter their own API key when prompted during execution.

### Run

From the repository root:

```bash
python code/02_llm_fda_labeling.py
```

The outputs are saved under:

```text
results/02_llm_fda_labeling/
```

including:

```text
llm_stage1_evidence.csv
llm_stage2_judgements.csv
llm_retained_non_unknown_results.csv
llm_reproduction_summary.json
llm_reproduction_settings.json
```

The final retained FDA-label output is:

```text
llm_retained_non_unknown_results.csv
```

---

## FDA Source Labels

The approximately 50,000 FDA drug-label files collected during the initial data-curation step are not duplicated in this repository.

FDA Structured Product Labeling data are publicly available and can be independently downloaded from official FDA drug-label resources.

The `selected_xml.zip` file provided in this repository contains the **2,836 curated XML labels used as direct input to the LLM analysis**.

Therefore, users who wish to reproduce only the LLM classification step can directly use the provided 2,836 XML files without repeating the complete FDA label collection and deduplication procedure.

Users who wish to reproduce the complete FDA-label curation process from the original source can independently download the publicly available FDA SPL files and perform the corresponding label-selection and deduplication procedures.

---

## From 610 FDA Labels to 802 Ingredient-Level Records

After exclusion of `unknown` classifications, the LLM analysis retained:

```text
610 FDA labels
```

For the external machine-learning evaluation, combination products were subsequently separated according to their **active ingredients**.

Each active ingredient was treated as an individual ingredient-level record and matched to its corresponding molecular and model-input features.

Therefore:

```text
610 retained FDA labels
        ↓
combination products separated by active ingredient
        ↓
802 ingredient-level records
```

The final **802 ingredient-level records** were used for external evaluation of the integrated CI/F/M placental-transfer classification framework.

This 610-to-802 expansion is a data-preparation step and is not performed by `02_llm_fda_labeling.py`.

---

## Reproducibility Note for Hosted LLMs

The LLM analysis was performed with:

```text
temperature = 0
```

However, exact outputs from a hosted LLM API may vary over time because model-serving infrastructure or deployed model versions may be updated by the provider.

For this reason, the historical result of **610 retained non-unknown FDA labels** is provided as the reference result for comparison.

---

## Data Availability

The final CI and F/M training matrices required to reconstruct the machine-learning models are provided in:

```text
data/Supplementary Dataset S1.xlsx
```

The curated FDA XML files required to reproduce the LLM classification workflow are provided in:

```text
selected_xml.zip
```

The original FDA SPL source labels can be independently obtained from publicly available FDA drug-label resources.

---

## Requirements

Python 3 is recommended.

Major Python packages used in this repository include:

```text
numpy
pandas
scikit-learn
joblib
openpyxl
langchain-core
langchain-groq
tqdm
```

Dependencies can be installed using:

```bash
pip install -r requirements.txt
```

---

## Citation

If you use the data, code, or models provided in this repository, please cite the associated publication:

**Integrated machine learning framework for quantitative prediction and classification of human placental drug transfer**

Full citation information and DOI will be added following publication.
