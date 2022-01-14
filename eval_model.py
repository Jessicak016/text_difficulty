import pandas as pd
import numpy as np
import spacy
import nltk
import string
from collections import Counter
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
import re
import os
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn_pandas import DataFrameMapper
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")

# Dale_Chall Word List
with open('dale_chall.txt') as f:
    lines = f.readlines()

dale_chall_word_list = [word.strip().lower() for word in lines]
dale_chall_word_set = set(dale_chall_word_list)

# import train and test data
train_data = pd.read_csv("WikiLarge_Train.csv")
final_test_data = pd.read_csv("WikiLarge_Test.csv")
train_data["id"] = list(range(len(final_test_data), len(final_test_data)+len(train_data)))
train_data = train_data[["id", "original_text", "label"]]

# for feature extraction:
all_data = pd.concat([final_test_data, train_data])
all_data = all_data.sort_values(by="id", axis=0)
all_data = all_data.set_index("id")

punctuations = list(string.punctuation)
punctuations = punctuations + ["...", "”", "’", "“", "…"]
punctuations_set = set(punctuations)

original_text_df = all_data[["original_text"]]

def clean_text(row):
    text = row["original_text"]

    text = text.replace("-RRB-", "")
    text = text.replace("-LRB-", "")
    text = text.replace("-LRB-", "")

    # replace any non-alphabetic characters and trailing spaces
    text = text.replace("Å ", "o")
    text = text.replace("Ã ", "a")

    # remove urls
    text = re.sub(r"http[s]?\:[\w\.\/\-]+", "", text)

    # remove any 's
    text = text.replace("'s", "")

    # remove any punctuations
    text = re.sub(r"[!\"#$%&\'()*+,\-\./:;<=>?@[\\\]\^_`{\|}~\.\.\.”’“\†\']", "", text)

    text = text.replace("  ", " ")
    text = text.strip()
    text = text.lower()
    return text

print("cleaning text...")
original_text_df["clean_text"] = original_text_df.apply(clean_text, axis=1)
print("cleaning text Done!")
all_data["cleaned_text"] = original_text_df["clean_text"]

all_features = all_data.copy()

punctuations = list(string.punctuation)
punctuations = punctuations + ["...", "”", "’", "“", "…", "``"]
punctuations_set = set(punctuations)

# get text length
def get_length(row):
    text = row["cleaned_text"]
    return len(text)

def get_tokens(row):
    text = row["cleaned_text"]
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [token for token in tokens if (token not in punctuations_set)]
    cleaned_tokens = [token.strip() for token in cleaned_tokens]
    return cleaned_tokens

def get_num_types(row):
    tokens = row["tokens"]
    types = set(tokens)
    return len(types)

def get_num_tokens(row):
    tokens = row["tokens"]
    return len(tokens)

def get_difficult_tokens(row):
    tokens = row["tokens"]
    difficult_tokens = [token for token in tokens if (token not in dale_chall_word_set)]
    return difficult_tokens

def get_num_difficult_tokens(row):
    difficult_tokens = row["difficult_tokens"]
    return len(difficult_tokens)

def get_prop_difficult_tokens(row):
    proportion = 0
    num_difficult_tokens = len(row["difficult_tokens"])
    num_tokens = row["num_tokens"]
    if num_tokens != 0:
        proportion = num_difficult_tokens/num_tokens

    return proportion

print("Extracting Features")
all_features["text_len"] = all_features.apply(get_length, axis=1)
print("1. Done")
all_features["tokens"] = all_features.apply(get_tokens, axis=1) # takes a long time
print("2. Done")
all_features["num_tokens"] = all_features.apply(get_num_tokens, axis=1)
print("3. Done")
all_features["num_types"] = all_features.apply(get_num_types, axis=1)
print("4. Done")
all_features["difficult_tokens"] = all_features.apply(get_difficult_tokens, axis=1)
print("5. Done")
all_features["num_difficult_tokens"] = all_features.apply(get_num_difficult_tokens, axis=1)
print("6. Done")
all_features["prop_difficult_tokens"] = all_features.apply(get_prop_difficult_tokens, axis=1)
print("7. Done")

# Token Representation
training = all_features[all_features["label"].isnull() == False]
features_list = ["cleaned_text", "text_len", "num_tokens", "num_types", "num_difficult_tokens"]

training_features = training[features_list]

training_labels = training["label"]
training_labels = training_labels.astype(int)

X_tr, X_te, y_train, y_test = train_test_split(training_features,
                                               training_labels,
                                               test_size=0.2,
                                               random_state=0)

max_feat = 1000

mapper = DataFrameMapper([("cleaned_text", CountVectorizer(binary=True, min_df=5, max_df=0.8, max_features=max_feat)),
                          (features_list[1:], None)])

print("Transforming Features...")
X_train = mapper.fit_transform(X_tr)
X_test = mapper.transform(X_te)

print("Shapes of Training and Test Data:")
print(X_train.shape, X_test.shape)

# final_test_data

##########

print("Fitting and Training Random Forest Model with best parameter...")
best_model_rf = RandomForestClassifier(random_state=0, n_estimators=500, max_features=30)
best_model_rf.fit(X_train, y_train)

feature_importances = best_model_rf.feature_importances_

print("Scoring the Random Forest Model...")
rf_train_score = best_model_rf.score(X_train, y_train)
rf_test_score = best_model_rf.score(X_test, y_test)

print("Random Forest: ")
print("Training Accuracy: {}".format(rf_train_score))
print("Test Accuracy: {}".format(rf_test_score))

predictions = best_model_rf.predict(X_test)

# Save the predictions and models feature_importances for later evaluation on Jupyter Notebook
print("Saving predictions to npy...")
with open('predictions.npy', 'wb') as f:
    np.save(f, predictions)

print("Saving Feature Importances...")
with open('feature_importances.npy', 'wb') as f:
    np.save(f, feature_importances)

print("Predicting on Final Test Data for Kaggle...")
final_testing = all_features[all_features["label"].isnull() == True]
final_testing_features = final_testing[features_list]
final_test = mapper.transform(final_testing_features)
final_predictions = best_model_rf.predict(final_test)

print("Saving final predictions to npy...")
with open('final_predictions.npy', 'wb') as f:
    np.save(f, final_predictions)
