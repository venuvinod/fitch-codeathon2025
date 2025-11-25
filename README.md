# 🌍 FitchGroup Codeathon '25 - Drive Sustainability using AI

Welcome to the FitchGroup Codeathon!

Your challenge is to develop machine learning models that estimate **Scope 1** and **Scope 2** greenhouse gas emissions for non-reporting companies using data from Sustainable Fitch.

This repository contains:

- 📘 README file with all the details you need to get started
- 📦 The training dataset
- 📦 The holdout test dataset
- 📁 Additional relational tables for feature engineering
- 🧪 Starter notebooks

Your goal is to engineer meaningful features, build accurate models, and submit predictions for the company-level emissions values in the **holdout test dataset**.

## 📂 Repository Structure

```
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── revenue_distribution_by_sector.csv
│   ├── environmental_activities.csv
│   ├── sustainable_development_goals.csv
│
├── notebooks/
│   ├── data_and_feature_engineering.ipynb
│   ├── baseline_model_and_inference.ipynb
│   ├── sample_submission.csv
│
└── README.md
```

## ✅ Objective

Predict company-level **Scope 1** and **Scope 2** greenhouse gas emissions for a holdout set of companies using the provided datasets.

At a high-level your workflow should look something like this:
1. Fork or clone this repository
2. Train your models using the files in `data/`
3. Generate predictions using your best models for all the rows in `data/test.csv`

Then you will need to submit the following:
- A `submission.csv` file with your predictions in the format matching the example in `notebooks/submission.csv`
- Your code repository or notebook(s)
- A short explanation of your approach including:
    - the performance your models achieved on the training dataset
    - the type of model used
    - a very brief description of your feature engineering & data transformations


## 🏁 Getting Started

We have provided a couple notebooks to help you get started.

### 1. `notebooks/basic_feature_engineering.ipynb`

This notebook shows you how to:

- Load all datasets
- Merge the 1-to-many relational tables (NACE exposures, environmental adjustments, SDGs)
- Perform basic feature engineering (e.g., aggregations, weighted averages, counts)
- Prepare a modeling dataset 

Use this as your launchpad—modify it, extend it, or build your own pipeline.

### 2. `notebooks/baseline_model_and_inference.ipynb`

This example notebook demonstrates:

- How to split your data for training and validation
- How to train a simple baseline model (e.g., linear model or tree-based model)
- How to generate predictions on the test set
- How to export your submission CSV

## 📦 Dataset Details

The dataset is intentionally structured to reward thoughtful feature engineering and the data is spread across a few tables that will require some transformation to turn them into additional signals for prediction.

### 1. `data/train.csv` (Main Fact Table)
 
 One row per company. This file is used for training and validation.

| **Field Name**          | **Description**                                                                                                                                                                                                                                                |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **entity_id**           | Unique identifier for each company/entity in the dataset. This ID links to additional information in the supplementary tables (e.g., sector exposures, environmental tags, SDG commitments).                                                                   |
| **region_code**         | Short code representing the broad geographic region where the entity primarily operates (e.g., *EU*, *NA*, *APAC*). Useful for high-level geographic clustering and energy-mix differences across regions.                                                     |
| **region_name**         | Full name of the geographic region associated with `region_code` (e.g., *Europe*, *North America*, *Asia-Pacific*).                                                                                                                                            |
| **country_code**        | ISO-style two- or three-letter country code indicating the entity’s country of headquarters or primary reporting location (e.g., *US*, *GB*, *DE*).                                                                                                            |
| **country_name**        | Full name of the entity’s country of headquarters or primary reporting location associated with `country_code`. Important for understanding electricity-grid carbon intensity and regulatory environments.                                                                                    |
| **revenue**             | The company’s annual total revenue provided in standardized units. Serves as a proxy for operational scale, production intensity, and potential emissions footprint.                                                                         |
| **overall_score**       | Sustainable Fitch’s overall sustainability score for the company which focuses on whether the entity's business activities make a positive or negative contribution towards environmental or social goals, as well as the effectiveness of governance. The scores range from 1 to 5, where 1 represents the company has full alignment with best practices and has a positive impact on the environment and on society. |
| **environmental_score** | Sustainable Fitch’s environmental score for the company which comprises 45% of the `overall_score`. May capture practices related to energy efficiency, pollution control, environmental risk management, etc., which can correlate with emissions performance.                                 |
| **social_score**        | Sustainable Fitch’s social score for the company which comprises 30% of the `overall_score`. Included for completeness; generally has lower direct correlation with emissions but may relate to corporate maturity and reporting discipline.                                                                    |
| **governance_score**    | Sustainable Fitch’s governance score for the company which comprises 25% of the `overall_score`. Included for completeness; generally has lower direct correlation with emissions but may relate to corporate maturity and reporting discipline.                                                          |
| ⭐ **target_scope_1**      | Ground-truth **Scope 1 emissions** for the entity (direct emissions from owned or controlled operations). This is one of the two target variables participants must train models to predict.                                                                   |
| ⭐ **target_scope_2**      | Ground-truth **Scope 2 emissions** for the entity (indirect emissions from purchased electricity, heat, or steam). This is the second required prediction target. Scope 1 and Scope 2 together represent operational emissions under the Greenhouse Gas Protocol.         |


### 2. `data/test.csv`

Same structure as `train.csv` **except without** the scope 1 scope 2 emission fields.

You must produce predicted `target_scope_1` and `target_scope_2` values for each `entity_id` in this file.

### 3. `data/revenue_distribution_by_sector.csv`  (1:many)

Granular revenue exposures by NACE Level 2.

| Field Name            | Description                                                                                                                                                                                                                                                                                          |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **entity_id**         | A unique identifier for each company or entity. This matches the `entity_id` field in `train.csv` and `test.csv`. Use this field to join sector exposure data to the main dataset.                                                                                                                   |
| **nace_level_1_code** | The NACE Level 1 sector code (letter-based). This represents the broadest economic activity classification for the portion of the company’s revenue tied to this sector (e.g., *C* for Manufacturing).                                                                                               |
| **nace_level_1_name** | The descriptive label associated with the NACE Level 1 code (e.g., “Manufacturing”). Useful for high-level grouping or manual feature engineering.                                                                                                                                                   |
| **nace_level_2_code** | The NACE Level 2 sector code (two-digit number). This provides a more granular economic classification for the segment of the company’s activity contributing to revenue. Each entity can have multiple Level 2 sectors.                                                                             |
| **nace_level_2_name** | The descriptive label for the NACE Level 2 code (e.g., “Manufacture of Chemicals”). Represents sector-specific revenue segments where emissions intensity can differ meaningfully.                                                                                                                   |
| **revenue_pct**       | Percentage of the company’s total revenue generated from the given Level 2 NACE sector. Values typically sum to 100% across all rows for the same `entity_id`. Use this to compute aggregations for carbon intensity, diversification, major contributing sector, etc. |



### 4. `data/environmental_activities.csv`  (1:many; sparse)

List of additional environmental activities that the company engages in that result in a benefit or penalty to the company's `environmental_score` as assessed by Sustainable Fitch. You can think of these as "extra" things a company may do (on purpose or not) that impact its environmental score which could contain a signal for emission predictions.

| Field Name               | Description                                                                                                                                                                                                                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **entity_id**            | Unique identifier for the company or entity. Used to join this table to the main `train.csv` dataset.                                                                                                                                                                                                                     |
| **activity_type**        | High-level category describing the type of environmental activity conducted by the entity. Can be used for filtering or aggregation purposes.                                                                               |
| **activity_code**        | The internal Sustainable Fitch code associated with the environmental activity (e.g., "A.1.3.B" could be recycling initiatives, renewable energy programs, etc). The list of potential activities is part of our proprietary scoring methodology and we are not able to share the details for each code publicly.                                                         |
| **env_score_adjustment** | Numeric value representing the magnitude and direction of impact to the `environmental score`. Negative values indicate environmentally positive activities as they move the score closer to 1 (good); positive values indicate detrimental activities as they move the score closer to 5 (bad). Values may be aggregated to engineer predictive features. |


### 5. `data/sustainable_development_goals.csv` (1:many; sparse)

List of United Nation Sustainable Development Goals the company reports commitment to (1:many; sparse).

To learn more visit:  https://www.un.org/en/exhibits/page/sdgs-17-goals-transform-world

| Field Name    | Description                                                                                                                                                                      |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **entity_id** | Unique identifier for the company or entity. This links back to `train.csv` and allows joining SDG commitments to the main fact table.                                           |
| **sdg_id**    | The numeric identifier of the Sustainable Development Goal (1–17). Indicates which SDG(s) the entity has publicly committed to.                                                  |
| **sdg_name**  | Human-readable name of the Sustainable Development Goal (e.g., “Climate Action”, “Responsible Consumption and Production”). Provided for clarity and easier feature engineering. |



## 🤖 Using AI During the Hackathon

We strongly encourage participants to use AI tools throughout the codeathon. Treat AI as an **expert advisor** that can help you think faster, explore more ideas, and improve the quality of your modeling.

Here are a couple ways to use AI effectively:

### 1. As a Machine Learning Co-Pilot

Use AI to:

- break down unfamiliar modeling concepts
- debug code
- suggest feature engineering methods
- explain cross-validation strategies
- compare algorithms and tune hyperparameters
- reason about how to handle sparse or normalized datasets

Think of AI as a **senior ML engineer** you can brainstorm with.

### 2. As a Sustainability Domain Expert

AI can also help you think deeply about the why behind emissions:

- How do sector characteristics influence Scope 1 vs. Scope 2?
- What operational activities tend to drive emissions?
- How might NACE sector mixes relate to energy intensity?
- How should environmental “benefits” or “penalties” be interpreted as signals?
- What patterns might appear in high-emissions industries?


Use AI as a **sustainability research assistant** to reason about the impact of sector exposure, energy use, business activities, and company practices.

