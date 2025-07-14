import pandas as pd
import numpy as np
from multicategory_classifier import multicategory_classification, classify_categories

df = pd.read_csv("glass.csv")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

split_idx = int(len(df) * 0.9)

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

categories = train_df["Type"].to_numpy()
features = train_df.iloc[:, :-1].to_numpy()

classifiers = multicategory_classification(features, categories, 0.1, 100000)

classified = classify_categories(classifiers, features)

print(f"Train accuracy: {np.sum(categories == classified) / len(categories)}")

test_categories = test_df["Type"].to_numpy()
test_features = test_df.iloc[:, :-1].to_numpy()

test_classified = classify_categories(classifiers, test_features)

print(f"Test accuracy: {np.sum(test_categories == test_classified) / len(test_categories)}")
