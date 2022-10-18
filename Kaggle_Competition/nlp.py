# %%
import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

# %%
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# %%
train_df

# %%
count_vectorizer = feature_extraction.text.CountVectorizer()

# %%
example_train_vectors = count_vectorizer.fit_transform(train_df['text'])

# %%
print(example_train_vectors[0].todense())

# %%



