🌸 Iris Flower Classification (K-Nearest Neighbors)

This project classifies Iris flowers into three species (Setosa, Versicolor, Virginica) using K-Nearest Neighbors (KNN), a fundamental classification algorithm in machine learning.

📌 Features:

Data Preprocessing: Handling and visualizing data with Pandas, Matplotlib & Seaborn

Exploratory Data Analysis (EDA): Histograms, Pairplots, Correlation Heatmap, and Scatter Plots

Machine Learning Model: KNN classifier with train-test split

Performance Evaluation: Model accuracy assessment

⚙️ Technologies Used:

✔ Python

✔ Pandas & NumPy

✔ Matplotlib & Seaborn

✔ Scikit-Learn

🔍 Understanding K-Nearest Neighbors (KNN)

📖 How KNN Works in General

KNN is a supervised learning algorithm used for classification and regression. It works by:

1️⃣ Storing all training data points.

2️⃣ When a new data point needs classification, it calculates the distance (e.g., Euclidean distance) between the new point and all training points.

3️⃣ It selects the K closest data points.

4️⃣ The majority class among these K neighbors determines the predicted class.

KNN is non-parametric, meaning it does not make assumptions about data distribution, making it effective for many real-world datasets.

🔬 How KNN is Used in This Project

📌 We used the Iris dataset, which contains 150 samples of iris flowers with the following features:

Sepal Length

Sepal Width

Petal Length

Petal Width

Species (Target Variable)

📌 Steps Followed:

1️⃣ Data Preprocessing:

Loaded the dataset using Pandas

Encoded categorical labels (species) into numerical values using Label Encoding

2️⃣ Exploratory Data Analysis (EDA):

Used histograms, scatter plots, and heatmaps to visualize feature distributions and relationships

3️⃣ Model Training & Evaluation:

Split the dataset into training (70%) and testing (30%) sets

📂 Dataset:

The classic Iris dataset is included in iris.csv
