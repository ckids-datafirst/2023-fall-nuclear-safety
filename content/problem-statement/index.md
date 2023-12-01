<img width="1033" alt="image" src="https://github.com/ckids-datafirst/2023-fall-nuclear-safety/assets/95256481/6781172b-3b0f-4ee4-86b6-a2f22a9aa7c4">---
title: Problem and Requirements
summary: Problem and Requirements document that will drive the work to be done in the project
date: "2018-06-28T00:00:00Z"
editable: true
share: false
---

## Introduction

This project, which will be co-advised by Dr. Yolanda Gil, will use Natural Language Processing (NLP) techniques to analyze voluminous Diablo Canyon Independent Safety Committee (DCISC) annual reports to identify the role and contribution of "Traits of a Healthy Nuclear Safety Culture", as defined by the Nuclear Regulatory Commission and the Institute of Nuclear Power Operations, in incident causation.

## Motivation

Nuclear power plant is a viable clean energy source given it does not generate any pollution while generating power. The diablo canyon nuclear power plant is the focus of this project, which is also the last operating nuclear power plant inside the state of California. Given the tragical incidences from Cernobyl in 1986 and from Fukushima in 2011, it is important to make this plant operates safely until scheduled closing by 2030. 

## Problem for the Semester

We need to identify the issue statment in side the 32nd annual report, form a golden standard where we manually labeled each statement using the ten healthy traits provided in the INPO report. After that, we aim to build models that can automate the labeling processure with reasonable model performance. There should be base line models and LLM models working in parallel so that we could compare the model performances and form a solid ground for the next semster's project.


## State of the Art

- Advanced Large Language Models (LLM) can be state-of-the-art tools to implement for this project. 
- Additional Machine Learning models such as Logistic Regression, KMeans, and traditional NLP methods like seed word are implemented in parallel via Python as the comparison group.

## Design and Approach

1. We split the document among each team member to get extract and labeled the issue statements to form the dataset.
2. With the help of professor Ulf, we finalized the baseline model selection and choose additional state-of-art models as comparison group.
   - The input for the issue statement model should be the entire annual report, and this model should have both the ability to do Optical Characterisitc Recognition (OCR), and also sentiment analysis or seed word extraction so that it could extract the issue statement as the output.
   - The input for all label classification models should be each issue statement, and the output should be the label(s) the model assigned with along with the confidence level and reasoning behind it.
4. We use seed word model as the baseline model, using the seed word from previous team as the base model. The main issue includes unbalanced healthy traits inside the issue statement dataset, and limited seed word is presented. Some clear improvements includes assigning weights to minority traits and put second-layered seed words.
5.  Moving from the seed word base line, we initiated the logistic regression and kmeans clustering models to set additional base line models to explore more potential directions.
6. At the same time, we utilized multiple the state-of-art LLM models and finalized with claud-2 and GPT3.5 models.
7. We test all models on the dataset to compare model performances.
<img width="1033" alt="image" src="https://github.com/ckids-datafirst/2023-fall-nuclear-safety/assets/95256481/c3541929-093d-4a3d-8cf5-59f7db444a17">
(An overview diagram for our system design)


## Use Case Scenario

Show with a use case scenario with examples of who the ultimate users could be, what the system that you plan to build for the semester will do in that scenario.  Provide a mockup of the outputs for the use case that you propose.

Researchers who want to investigate the operation of the Diablo Canyon Nuclear Power Plant healthy traits, or compare different years' performance to determine the main focus of future improvements. 


## Desired Outcomes and Benefits

Connect back the system that you will be building with the initial motivation for the work, and what additional future work might be needed in order for the system to provide benefits to its users.
