# IPL Data Analysis Project

Exploratory Data Analysis (EDA) of IPL cricket data (2008-2020) using Python.

## Contents
- Data cleaning and missing-value checks
- Season-wise match trends
- Team performance (most wins)
- Toss analysis (does winning the toss help win the match?)
- Top run scorers and top wicket takers
- Scoring pattern by over
- Venue analysis (highest average-scoring grounds)
- Correlation heatmap

## How to run
1. Install dependencies:

pip install pandas numpy matplotlib seaborn

2. Download the dataset (see below) and place `matches.csv` and `deliveries.csv`
   in this folder.
3. Run:

python ipl_eda.py

   Charts are saved to an `eda_output/` folder; summary stats print to the console.

## Dataset
Sourced from Kaggle: [IPL Complete Dataset (2008-2020)](https://www.kaggle.com/patrickb1912/ipl-complete-dataset-20082020)

## Author
Mohammad Zain
