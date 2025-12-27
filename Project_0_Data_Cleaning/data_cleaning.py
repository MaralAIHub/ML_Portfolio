import pandas as pd
import numpy as np

# Load original Kaggle Titanic dataset
df = pd.read_csv("train.csv")

dirty = df.copy()

np.random.seed(42)

# 1. Introduce missing values
dirty.loc[np.random.choice(dirty.index, 50, replace=False), "Age"] = None
dirty.loc[np.random.choice(dirty.index, 30, replace=False), "Fare"] = None
dirty.loc[np.random.choice(dirty.index, 20, replace=False), "Embarked"] = None

# 2. Introduce wrong datatypes in Age
dirty.loc[np.random.choice(dirty.index, 10, replace=False), "Age"] = "unknown"

# 3. Duplicate random rows
dirty = pd.concat([dirty, dirty.sample(20)], ignore_index=True)

# Save dirty dataset
dirty.to_csv("dirty_titanic_kaggle.csv", index=False)

print("Dirty dataset created.")
