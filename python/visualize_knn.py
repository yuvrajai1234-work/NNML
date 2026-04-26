import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Set premium dark theme
plt.style.use('dark_background')
accent_color = '#00d2ff'
secondary_color = '#ff007a'

# Load the dataset
df = pd.read_csv('IBM_HR_Attrition.csv')

# Features: Age vs MonthlyIncome
X = df[['Age', 'MonthlyIncome']]
y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Mesh for boundaries
h = .01
x_min, x_max = -0.1, 1.1
y_min, y_max = -0.1, 1.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plotting
plt.figure(figsize=(12, 7))
plt.contourf(xx, yy, Z, cmap='RdBu', alpha=0.2)
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df['Attrition'], 
                palette={'Yes': secondary_color, 'No': accent_color}, s=50, alpha=0.7)

plt.title('K-Nearest Neighbors: Mathematical Decision Boundaries', fontsize=16, pad=20)
plt.xlabel('Age (Normalized)', fontsize=12)
plt.ylabel('Monthly Income (Normalized)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.1)
plt.tight_layout()
plt.savefig('knn_visual.png', dpi=300)
print("Graph saved as knn_visual.png")
plt.show()
