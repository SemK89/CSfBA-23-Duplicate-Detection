import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

LSH_df = (pd.read_excel('results-LSH.xlsx')
          .groupby(['k-shingle', 'n_bands'], as_index=False).mean())
LSH_df['n_bands'] = LSH_df['n_bands'].apply(lambda x: 420/x if x != 0 else 0)
LSH_df = LSH_df.rename(columns={'n_bands': 'rows per band'})
class_df = pd.read_excel('results-full.xlsx')
class_df_grouped = class_df.groupby(['k-shingle', 'n_bands'], as_index=False).mean()

for value in ['pair quality', 'pair completeness', 'f1*']:
    plt.figure()
    df = LSH_df[['k-shingle', value, 'fraction of comparisons']]
    sns.lineplot(data=df, x='fraction of comparisons', y=value, style='k-shingle', hue='k-shingle')

df_f1 = LSH_df[['k-shingle', 'rows per band', 'f1*']]
plt.figure()
sns.lineplot(data=df_f1, x='rows per band', y='f1*', style='k-shingle', hue='k-shingle')

for value in ['precision', 'recall', 'f1']:
    plt.figure()
    df = class_df[['key-shingle', 'value-shingle', value]]
    sns.boxplot(data=df, x='key-shingle', y=value, hue='value-shingle')
