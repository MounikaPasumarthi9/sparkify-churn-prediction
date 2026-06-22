# sparkify-churn-prediction
This project predicts which users are likely to stop using Sparkify's music streaming service. Data and code are from a Databricks notebook.
# Sparkify Churn Prediction (Work in Progress)

This project predicts which users are likely to stop using Sparkify's music streaming service.

The code is originally from a **Databricks notebook**. I exported it as a `.py` file and uploaded it to GitHub.

## What I've done so far

- Loaded the data
- Created basic features:
  - Page counts (neutral, negative, positive, etc.)
  - UserActiveTime (days active)
  - hourAvg (average hour of activity)
- Started normalizing and scaling features

## What I'm still working on

- Slope feature (trend over time)
- Final model training and evaluation
- Understanding why we normalize by UserActiveTime
- Understanding how slope is calculated
- Understanding why we scale features

## Status

This is a work in progress. I'm learning PySpark and ML feature engineering step by step.

## Files

- `Predict Customer Churn  2026-06-09 12_13_13.py` (exported from Databricks notebook)
