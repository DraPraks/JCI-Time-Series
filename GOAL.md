Here is the full parsed text of the document **TopicC_IDX_Stock**: [docs.google](https://docs.google.com/document/d/1AqXNLrgDwo1j1F1aeCR7q0esKlGaOp5iqEy_iz2fwZU/edit?tab=t.0)

***

# CSGE603130 Introduction to Artificial Intelligence & Data Science 2025/2026
## Mini Project Topic C: Stock Price Movement Prediction on Indonesia Stock Exchange (IDX)

***

## Time Series Data

Time series data is a sequence of observations recorded at regular intervals over time. Unlike conventional tabular data, where each row represents an independent observation, time series observations carry temporal dependency: what happened yesterday influences what we see today. This property makes time series data both information-rich and technically distinct from other data types used in machine learning.

Time series data appears across many domains, from energy consumption and patient health signals to sales figures and financial markets. In all these settings, the analysis serves a common purpose: to understand how a quantity evolves over time, and to extract patterns that can inform decisions about the future. Three structural components typically characterize a time series:

- **Trend:** the long-term direction in the data, whether rising, falling, or remaining stable over time.
- **Seasonality:** recurring patterns that repeat at fixed intervals, such as daily trading peaks or annual revenue cycles.
- **Noise:** random fluctuations that cannot be attributed to trend or seasonality, and which any model must learn to look past.

Before building any model, time series data typically requires several preprocessing steps. Missing observations must be handled carefully to preserve the continuity of the sequence. Outliers that could distort model learning must be identified and addressed. When required by the chosen method, stationarity checks ensure that the mean and variance of the series remain stable over time; if they do not, transformations such as differencing or log-scaling are applied.

***

## From Time Series to Supervised Learning

Most machine learning algorithms expect a fixed-size feature vector for each sample. Raw time series data, where a single entity generates hundreds or thousands of sequential rows, does not naturally fit this format. The bridge between the two is feature engineering: transforming sequential observations into structured, tabular features that capture meaningful temporal patterns.

### Lag Features

Reference: https://media.geeksforgeeks.org/wp-content/uploads/20240919181832/acf.png

A lag feature is the value of a variable at a prior timestep, brought into the current row as a predictor. The underlying assumption is simple: what happened in the recent past contains useful signal about what will happen next. In a stock price context, for example, the closing prices of the past five trading days can each become a separate input feature for predicting tomorrow's movement.

### Sliding Window and Rolling Statistics

Reference: https://lazyprogrammer.me/wp-content/uploads/2025/08/Sliding-window-technique-for-cutting-the-initial-time-series-into-fixed-length-samples.png (Left) and https://media.geeksforgeeks.org/wp-content/uploads/20201126142256/plot.png (Right)

Rather than carrying every individual past value as a separate feature, rolling statistics summarize a window of recent observations into a single aggregate. Common examples include the moving average, rolling standard deviation, and rolling minimum or maximum over a fixed lookback period. These features smooth out day-to-day noise and expose the underlying trend or volatility of a series within a recent window. The size of the window is a design choice: a short window reacts quickly to recent changes, while a longer window captures more stable, structural patterns.

Together, lag features and rolling statistics allow a time series problem to be reframed as a supervised learning problem. Each row in the resulting dataset represents one point in time for one entity. The features capture what has happened in the recent past; the label captures what we want to predict. From there, any standard classification or regression algorithm can be applied.

***

## Task Description

In this project, you are asked to predict the next-day price movement direction of stocks listed on the Indonesia Stock Exchange (IDX). For each trading day t of each stock in your dataset, you will engineer features from historical OHLCV data in the days preceding t, then predict whether the closing price on day t+1 will be higher or lower than on day t.

The prediction target is defined as follows:

| Label | Value | Condition |
|---|---|---|
| Up | 1 | Close Price on day t+1 > Close Price on day t |
| Down / Flat | 0 | Close Price on day t+1 <= Close Price on day t |

This is a binary classification task. Each row in your modeling dataset represents one trading day for one stock ticker. The features for that row must be derived exclusively from data prior to day t; no information from day t+1 or beyond may be used as input. Violating this constraint constitutes data leakage and will invalidate your results.

You are free to determine your own approach to feature engineering, model selection, and evaluation strategy. Whatever choices you make, justify them clearly in terms of the problem context. A technically valid but unexplained pipeline is not sufficient; the reasoning behind your decisions is as important as the results.

Beyond quantitative evaluation, you are also expected to carry out a qualitative analysis of your results. Consider questions such as: Under what market conditions does your model perform well or poorly? Does predictive accuracy differ across stock sectors or time periods? How do the most informative features relate to established concepts in technical analysis? Observations like these are what transform a model score into a business insight.

The business motivation for this task is to support data-driven investment decision-making. A reliable directional predictor, even an imperfect one, can serve as one input in a broader decision-support system for retail investors navigating the Indonesian equity market.

***

## Dataset Description

The dataset used in this project is the Complete Indonesia Stock Market (IDX) 2000-2024, available publicly on Kaggle. It contains daily OHLCV trading records for 931 companies listed on the IDX. Each row represents one trading day for one stock ticker, with the following columns:

| Column | Description |
|---|---|
| Date | Trading date |
| Ticker | Stock ticker code (unique identifier per company) |
| Open | Opening price of the trading day |
| High | Highest traded price within the day |
| Low | Lowest traded price within the day |
| Close | Closing price of the trading day |
| Adjusted Close | Closing price adjusted for corporate actions such as stock splits and dividend distributions |
| Volume | Total number of lots traded during the day |

The Adjusted Close column deserves particular attention. Because corporate actions such as stock splits and dividend distributions create discontinuities in raw closing prices, Adjusted Close provides a more reliable basis for computing historical returns and building price-based features. You should carefully consider which price column is most appropriate for each feature you engineer.

### Obtaining Sector Classification Data

The primary dataset does not include sector information. Because the emiten subset constraint below requires you to select stocks by IDX-IC sector, you will need to obtain a sector-to-ticker mapping from a separate source and join it with the primary dataset before you begin any analysis.

Two practical options are available, both free and publicly accessible:

- **IDX Official Stock List:** The Indonesia Stock Exchange publishes a regularly updated stock list at idx.co.id/en/market-data/stocks-data/stock-list/. Use the Download button on that page to obtain an Excel file containing each company's ticker code, company name, and IDX-IC sector.
- **Kaggle, Indonesian Public Companies:** A community-maintained dataset at kaggle.com/datasets/zororaka/indonesian-public-companies contains a mapping of IDX tickers to their sector, subsector, and industry under the IDX-IC scheme.

Whichever source you use, the join key is the Ticker column. Document this step clearly in your notebook, including where you obtained the sector mapping and how you handled any tickers that did not match between the two sources.

***

## Scope Constraints

For this project, you are required to apply the following constraints:

- **Period:** Use data from 2020 to 2024. This window spans the COVID-19 crash of 2020, the recovery phase of 2021, global volatility through 2022 and 2023, and relative stabilization in 2024. A model trained and evaluated across these contrasting regimes is more likely to reveal genuine predictive signal rather than patterns specific to a single market environment.
- **Emiten subset:** Select stocks from 2 to 3 IDX-IC sectors, with a total of no more than 100 stocks across all sectors. Aim for a roughly balanced distribution across sectors. Document your selection criteria clearly.

One constraint that applies specifically to time series data must be observed throughout your pipeline: the temporal ordering of your data cannot be violated. When splitting data into training and testing sets, the test set must always consist of later dates than the training set. A random split that shuffles rows across time would allow your model to learn from the future when predicting the past, producing inflated metrics that do not reflect real-world performance.

***

## Project Timeline

| Week | Period | CRISP-DM Phase | Milestone Deliverables |
|---|---|---|---|
| Week 1 | May 18-24 | Business Understanding & Data Understanding | EDA, identification of data patterns, visualizations, data cleaning, and handling of missing values |
| Week 2 | May 25-31 | Data Preparation & Modeling | Feature engineering, data transformation and normalization, construction of the modeling dataset, and classifier training |
| Week 3 | Jun 1-7 | Evaluation & Reporting | Evaluation metric computation, analysis of experimental results, model interpretation, notebook clean-up, and final report submission |

***

## References

- GeeksforGeeks, Time Series Analysis and Forecasting: https://www.geeksforgeeks.org/machine-learning/time-series-analysis-and-forecasting/
- Machine Learning Mastery, 5 Tips for Getting Started with Time Series Analysis: https://machinelearningmastery.com/5-tips-for-getting-started-with-time-series-analysis/
- Machine Learning Mastery, Basic Feature Engineering With Time Series Data in Python: https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/
- Kaggle Dataset, Complete Indonesia Stock Market (IDX) 2000-2024: https://www.kaggle.com/datasets/eren2222/complete-indonesia-stock-exchange-idx-2000-2024
- IDX Official Stock List: https://www.idx.co.id/en/market-data/stocks-data/stock-list/
- Kaggle, Indonesian Public Companies: https://www.kaggle.com/datasets/zororaka/indonesian-public-companies/data

***

*Note: The document contains images (ACF/PACF plots and sliding window diagrams) that are referenced inline but are not reproducible as text.*