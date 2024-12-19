---
title: Results
summary: The main results of the work done to date
date: "2018-06-28T00:00:00Z"
editable: true
share: false
---

# Natural Language Processing of Safety Reports in Nuclear Power Plants

## System and Model Performance

Our project developed several models to classify nuclear safety issues according to safety culture traits, with the following performance metrics:

- Claude-2 Large Language Model: 79% accuracy
- Seed Word Model: 58% accuracy 
- Logistic Regression Model: 51% accuracy

![Model Performance](images/result.png)

## Discussion of Findings

### Key Results
- Successfully developed a classification system for mapping nuclear safety issues to safety traits
- Demonstrated that LLMs (specifically Claude-2) outperform traditional methods significantly
- Created an effective seed word approach that leverages domain expertise through carefully selected primary and secondary keywords
- Established baseline performance metrics for future improvements

### Unexpected Results
- The significant performance gap between traditional ML approaches and LLMs
- The effectiveness of carefully curated seed words even without extensive training data

### Impact
- Helps maintain safety standards at nuclear power facilities
- Enables automated processing of ~900 page annual safety reports
- Supports the continuation of California's nuclear energy supply (10% of total energy)
- Contributes to preventing serious incidents like Chernobyl and Fukushima

## Limitations and Future Work

### Current Limitations
- Limited dataset (13 available reports out of 33)
- Model performance still has room for improvement
- Current focus on primary classifications only

### Future Goals
1. Test different large language models to improve our baseline performance
2. Expand to secondary and tertiary seed words for multi-class classification
3. Create a dashboard with integrated model functionality
4. Build model for automated issue statement extraction from raw reports

### Dataset Details
- Source: Diablo Canyon Independent Safety Committee
- Scope: ~900 page annual reports
- Availability: 13 out of 33 historical reports
- Coverage: Multiple aspects of nuclear safety culture traits

[Visit our GitHub Repository](https://github.com/ckids-datafirst/2023-fall-nuclear-safety)
