import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

df = pd.read_csv("deliveries.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe().T)
print(df.columns)
print(df.isnull().sum())
df['match_sequence'] = range(1, len(df) + 1)

sns.histplot(df['total_runs'])
plt.title("Distribution of Runs per Ball")
plt.xlabel("Runs Scored")
plt.ylabel("Number of Balls")
plt.show()
plt.figure(figsize=(12,6))
sns.scatterplot(x='match_sequence', y='total_runs', data=df, alpha=0.3)
plt.title('Runs per Ball Across IPL Matches')
plt.xlabel('Match ID')
plt.ylabel('Runs Scored per Ball')
plt.yticks(np.arange(0, 8, 1))
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
plt.show()
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap (Numeric Columns)')
plt.show()
top_players = df.groupby('batter')['total_runs'].sum().sort_values(ascending=False)
df['extras_type'].fillna('No Extras', inplace=True)
df['player_dismissed'].fillna('Not Out', inplace=True)
df['dismissal_kind'].fillna('Not Out', inplace=True)
df['fielder'].fillna('No Fielder', inplace=True)
print(top_players.head(5))
top_bowlers = df.groupby('bowler')['is_wicket'].sum().sort_values(ascending=False)
print(top_bowlers.head(5))



