import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tkinter as tk
from tkinter import filedialog

# Introduction
def print_intro():
    print("Welcome to the Enhanced Modeling and Simulation Project!")
    print("This project focuses on Python-based data modeling, simulation, and evaluation.\n")
    print("Steps:")
    print("1. Data Generation or Upload")
    print("2. Exploratory Data Analysis (EDA)")
    print("3. Preprocessing")
    print("4. Modeling")
    print("5. Evaluation and Visualization\n")

# Function to upload dataset
def upload_dataset():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    if file_path:
        return pd.read_csv(file_path)
    else:
        print("No file selected. Exiting...")
        exit()

# Function to preprocess data
def preprocess_data(data):
    print("\nPreprocessing Data...")
    # Handle missing values
    if data.isnull().sum().any():
        print("Missing values detected. Filling with column mean for numeric and mode for categorical data.")
        for column in data.columns:
            if data[column].dtype in ['float64', 'int64']:
                data[column].fillna(data[column].mean(), inplace=True)
            else:
                data[column].fillna(data[column].mode()[0], inplace=True)

    # Encode categorical features
    label_encoders = {}
    for column in data.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    print("Preprocessing complete.\n")
    return data

# Generate or load data
def load_data():
    print("Select the type of data:")
    print("1. Synthetic Data")
    print("2. Upload Custom Dataset")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        print("\nGenerating synthetic data...")
        X_reg, y_reg = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)
        X_clf, y_clf = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=42)
        return X_reg, y_reg, X_clf, y_clf, None
    elif choice == "2":
        custom_data = upload_dataset()
        print("\nCustom dataset loaded successfully:")
        print(custom_data.head())
        custom_data = preprocess_data(custom_data)
        target_column = custom_data.columns[-1]  # Assume last column is the target
        X = custom_data.drop(target_column, axis=1)
        y = custom_data[target_column]
        return None, None, X, y, custom_data
    else:
        print("Invalid choice. Exiting...")
        exit()

# Perform EDA
def perform_eda(X, y, task):
    print(f"\nPerforming EDA for {task} Data...")
    data = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(X.shape[1])])
    data["Target"] = y

    # Correlation heatmap
    correlation_matrix = data.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f"Correlation Matrix ({task})")
    plt.show()

    # Additional visualizations based on task
    if task == "Classification":
        sns.pairplot(data, hue="Target", palette="viridis")
        plt.title("Feature Pairplot (Classification)")
        plt.show()
    else:
        plt.figure(figsize=(6, 4))
        plt.scatter(X[:, 0], y, alpha=0.7, color="blue")
        plt.title("Feature vs Target (Regression)")
        plt.xlabel("Feature")
        plt.ylabel("Target")
        plt.grid(True)
        plt.show()

# Standardize data (for better model performance)
def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Build and evaluate models
def build_and_evaluate(X_train, X_test, y_train, y_test, task):
    print(f"\nBuilding and Evaluating {task} Model...")
    if task == "Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"\n{task} Model Performance:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R^2 Score: {r2:.2f}")

        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, y_pred, alpha=0.7, color="red")
        plt.title("Actual vs Predicted (Regression)")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid(True)
        plt.show()

    elif task == "Classification":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f"\n{task} Model Performance:")
        print(f"Accuracy: {accuracy:.2f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
        plt.title("Confusion Matrix (Classification)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

# Main function
def main():
    print_intro()
    X_reg, y_reg, X_clf, y_clf, custom_data = load_data()

    if X_reg is not None and y_reg is not None:
        perform_eda(X_reg, y_reg, "Regression")
        X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        X_train, X_test = standardize_data(X_train, X_test)
        build_and_evaluate(X_train, X_test, y_train, y_test, "Regression")

    if X_clf is not None and y_clf is not None:
        perform_eda(X_clf, y_clf, "Classification")
        X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
        X_train, X_test = standardize_data(X_train, X_test)
        build_and_evaluate(X_train, X_test, y_train, y_test, "Classification")

if __name__ == "__main__":
    main()
