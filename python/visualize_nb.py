import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm

# Set premium dark theme
plt.style.use('dark_background')
stay_color = '#00d2ff'
leave_color = '#ff007a'

import os

csv_path = os.path.join(os.path.dirname(__file__), '..', 'employee-attrition-risk', 'IBM_HR_Attrition.csv')
if not os.path.exists(csv_path):
    csv_path = 'IBM_HR_Attrition.csv'

df = pd.read_csv(csv_path)
feature = 'Age'
stay = df[df['Attrition'] == 'No'][feature]
attrition = df[df['Attrition'] == 'Yes'][feature]

plt.figure(figsize=(12, 7))

# Plot Gaussian Curves
sns.kdeplot(stay, color=stay_color, fill=True, label='Class: Stay', alpha=0.3, linewidth=3)
sns.kdeplot(attrition, color=leave_color, fill=True, label='Class: Leave', alpha=0.3, linewidth=3)

# Theoretical Gaussian Overlay (Pure Math)
x = np.linspace(df[feature].min()-5, df[feature].max()+5, 200)
plt.plot(x, norm.pdf(x, stay.mean(), stay.std()), color=stay_color, linestyle='--', alpha=0.6)
plt.plot(x, norm.pdf(x, attrition.mean(), attrition.std()), color=leave_color, linestyle='--', alpha=0.6)

plt.title(f'Naive Bayes: Gaussian Probability Distribution ({feature})', fontsize=16, pad=20)
plt.xlabel(feature, fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.1)
plt.tight_layout()
plt.savefig('nb_visual.png', dpi=300)
print("Graph saved as nb_visual.png")
plt.show()
