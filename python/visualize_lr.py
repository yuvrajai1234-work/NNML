import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set premium dark theme
plt.style.use('dark_background')
accent_color = '#00d2ff'
trend_color = '#ff007a'

df = pd.read_csv('IBM_HR_Attrition.csv')
df['Attrition_Num'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

features = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'DistanceFromHome']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

for i, col in enumerate(features):
    ax = axes[i//2, i%2]
    sns.regplot(data=df, x=col, y='Attrition_Num', ax=ax,
                scatter_kws={'alpha':0.1, 'color': accent_color},
                line_kws={'color': trend_color, 'linewidth': 3},
                logistic=False)
    ax.set_title(f'Linear Slope: {col} vs Attrition', fontsize=12)
    ax.set_ylim(-0.2, 1.2)
    ax.grid(True, linestyle='--', alpha=0.1)

plt.suptitle('Linear Regression: Feature Correlation Slopes', fontsize=18, y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('lr_visual.png', dpi=300)
print("Graph saved as lr_visual.png")
plt.show()
