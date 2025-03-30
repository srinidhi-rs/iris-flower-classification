import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# LOADING THE DATASET
df = pd.read_csv('iris.csv')

# DATA VISUALIZATION
# Histogram of feature distributions
df.hist(figsize=(10, 8), bins=20)
plt.suptitle('Feature Distributions', fontsize=16)
plt.show()

# Pairplot of features
sns.pairplot(df, hue='species', palette='coolwarm')
plt.show()

# Correlation heatmap
corr = df.corr(numeric_only=True)
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# LABEL ENCODING
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# MODEL TRAINING
X = df.drop(columns=['species'])
Y = df['species']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

# KNN CLASSIFICATION
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
#print('KNN Accuracy:', model.score(X_test, Y_test) * 100)

# Scatter plot for Sepal and Petal dimensions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ['red', 'green', 'blue']
species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

for i in range(3):
    subset = df[df['species'] == i]
    axes[0].scatter(subset['sepal_length'], subset['sepal_width'], color=colors[i], label=species[i])
    axes[1].scatter(subset['petal_length'], subset['petal_width'], color=colors[i], label=species[i])

axes[0].set_xlabel('Sepal Length')
axes[0].set_ylabel('Sepal Width')
axes[0].set_title('Sepal Dimensions')
axes[0].legend()

axes[1].set_xlabel('Petal Length')
axes[1].set_ylabel('Petal Width')
axes[1].set_title('Petal Dimensions')
axes[1].legend()

plt.show()
