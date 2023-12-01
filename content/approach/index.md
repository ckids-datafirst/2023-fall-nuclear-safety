---
title: Approach
summary: Data science methodology used to address the problem, including data preprocessing steps, exploratory data analysis, feature engineering techniques, machine learning models, and evaluation metrics.
date: "2018-06-28T00:00:00Z"
editable: true
share: false
---

This page contains key sections of the **Final Report** for the project focused on the data science methodology used to approach the problem.  It should be no more than 3 pages long.  It should be done after or in combination with the Requirements document.  It should have an initial release after no more than eight weeks into the project, and can serve as an interim project report.  It can be refined as the project progresses and the problem is better understood.  

## Data Quality

Describe any steps that were used to address any issues concerning the quality of the data.  This may include collecting data quality metrics, discarding subsets of the data, or applying specific techniques for handling missing values, dealing with outliers, etc. 
- Raw Annual Reports are extracted from the DCISC website (https://www.dcisc.org/annual-reports/). The focus of this project is on the 32nd Annual Report.
- We read through the entire report, manually extracted all potential issue statements, labeled them according to the INPO Traits of a Healthy Nuclear Safety Culture handout, and provided reasons for the classification to form the finalized gold standard dataset. 
- There are both textual and imagery inputs inside the report. Thus, optical characteristic recognition (OCR) is required to extract the entire report.
- The data quality is low as the issue statement is scattered across the report without clear patterns. In addition, many of the issue statements lack detailed descriptions or the root cause report inside the annual report, which requires additional documentation.

## Data Preprocessing

Describe the steps taken to preprocess the raw data to prepare it for analysis. This may include data transformations to convert to a required format, feature engineering operations, encoding features as binary, etc.

- We used OCR via Python since many issue statements are mentioned with the imagery format rather than text.
- After the OCR, we can get a txt file that is readable for the machine for the issue statement extraction task (left for next semester's project).
- Manually extracted and labeled all issue statements using the INPO Traits of a Healthy Nuclear Safety Culture to form the golden standard csv dataset.
- Vectorize the issue statements and ten safe traits so the LLMs can read them.
- Build additional seed words and secondary seed words based on the previous team's effort.

## Exploratory Data Analysis (EDA)

- After finalizing the CSV dataset, we visualized the distribution of the safety traits, which shows the imbalanced issue and reveals the top safety traits related to the issues.

## Model Development

### Seed Word Model:
-- Python libraries: spacy and scikit-learn
-- The classic NLP method recommended by Professor Ulf also provided a good baseline for the previous team.
-- We utilized the previous team's seed words on the first try and achieved 53% accuracy with a 47% F1 score. 
-- Added weights for safety traits to counter data imbalance issues and added additional seed words to make the model more comprehensive.

### Logistic Regression Model:
-- Python libraries: scikit-learn
-- Vectorized the dataset to help with large textual dataset analysis.
-- Use 10-fold cross-validation and get ten accuracy scores for each logistic regression model

### Large Language Models:
-- Fine-tuned the model with the INPO Traits of a Healthy Nuclear Safety Culture handout
-- Formed output format to include the original/summarized issue statement in one column, traits it labeled with along with the confidence level in one column, and reasoning for the third column. Finally, label all issue statements inside the dataset with the required table format.
-- We tried both ChatGPT3.5 and Claud-2 from perplexity using a similar process.


## Model Evaluation

- We utilized accuracy and F-1 score to evaluate the model performance. 
- We use cross-validation for the overfitting check and model performance evaluation for the Logistic Regression model.
