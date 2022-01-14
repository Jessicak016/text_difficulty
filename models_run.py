import pandas as pd
import numpy as np
import spacy
import nltk
import string
from collections import Counter
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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

# import matplotlib.pyplot as plt

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

# temporarily save the features (only text and numerical features)

all_features_ts = all_features[["original_text", "cleaned_text",
                                "num_tokens", "num_types",
                                "num_difficult_tokens", "prop_difficult_tokens"]]

all_features_ts.to_csv("all_features.csv", index=True)

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

# Random Baseline
dummy_clf = DummyClassifier(strategy="uniform", random_state=0)
dummy_clf.fit(X_train, y_train)

random_baseline_score = dummy_clf.score(X_test, y_test)
print("Uniform Random Baseline: ")
print("Accuracy: {}".format(random_baseline_score))

# Naive Bayes Classifier Baseline
gnb = GaussianNB()
gnb.fit(X_train, y_train)

bayes_baseline_score = gnb.score(X_test, y_test)
print("Gaussian Naive Bayes Baseline using extracted features: ")
print("Accuracy: {}".format(bayes_baseline_score))

# save the baseline outcome
baselines_outcome = {"Random_Baseline": [random_baseline_score], "Naive_Bayes": [bayes_baseline_score]}
baselines_df = pd.DataFrame.from_dict(baselines_outcome)
baselines_df.to_csv("baseline.csv")


# Logistic Regression
logisticr = LogisticRegression(random_state=0)
param_grid = {"C": [0.1, 1],
              "max_iter": [400, 500, 900]}

print("GridSearch with Logistic Regression")
clf = GridSearchCV(logisticr, param_grid, cv=5, scoring='accuracy', verbose=3)
clf.fit(X_train, y_train)

best_C = clf.best_params_["C"]
best_max_iter = clf.best_params_["max_iter"]

logistic_cv_results = pd.DataFrame.from_dict(clf.cv_results_, orient="columns")
logistic_cv_results.to_csv("logistic_cv_results.csv", index_label=False)

best_logisticr = LogisticRegression(C=best_C,
                                    max_iter=best_max_iter,
                                    random_state=0)
best_logisticr.fit(X_train, y_train)
logistic_train_score = best_logisticr.score(X_train, y_train)
logistic_test_score = best_logisticr.score(X_test, y_test)

print("Logistic Regression: ")
print("Training Accuracy: {}".format(logistic_train_score))
print("Test Accuracy: {}".format(logistic_test_score))
print("Optimal C: {}, Max Iterations: {}".format(best_C, best_max_iter))

# K-Nearest Neighbor
neigh = KNeighborsClassifier()

param_grid = {"n_neighbors": [10, 20]}

print("GridSearch with K-Nearest Neighbor")
clf = GridSearchCV(neigh, param_grid, cv=5, scoring='accuracy', verbose=3)
clf.fit(X_train, y_train)

neigh_cv_results = pd.DataFrame.from_dict(clf.cv_results_, orient="columns")
neigh_cv_results.to_csv("neigh_cv_results.csv", index_label=False)

best_n_neighbors = clf.best_params_["n_neighbors"]

best_neigh = KNeighborsClassifier(n_neighbors=best_n_neighbors)
best_neigh.fit(X_train, y_train)
neigh_train_score = best_neigh.score(X_train, y_train)
neigh_test_score = best_neigh.score(X_test, y_test)

print("K-Neighbors: ")
print("Training Accuracy: {}".format(neigh_train_score))
print("Test Accuracy: {}".format(neigh_test_score))
print("Optimal n_neighbors: {},".format(best_n_neighbors))

# Random Forest
# https://stackoverflow.com/questions/36107820/how-to-tune-parameters-in-random-forest-using-scikit-learn
rf = RandomForestClassifier(random_state=0, n_estimators=500)

param_grid = {"max_features": [10, 30]}
print("GridSearch with Random Forest")
clf = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', verbose=3)
clf.fit(X_train, y_train)

rf_cv_results = pd.DataFrame.from_dict(clf.cv_results_, orient="columns")
rf_cv_results.to_csv("rf_cv_results.csv", index_label=False)

best_max_features = clf.best_params_["max_features"]

best_rf = RandomForestClassifier(random_state=0,
                                 max_features=best_max_features)
best_rf.fit(X_train, y_train)
rf_train_score = best_rf.score(X_train, y_train)
rf_test_score = best_rf.score(X_test, y_test)

print("Random Forest: ")
print("Training Accuracy: {}".format(rf_train_score))
print("Test Accuracy: {}".format(rf_test_score))
print("Optimal max_features: {}, n_estimators: {}".format(best_max_features, best_n_estimators))

# save the best scores of each model to csv file
accuracy_scores = {"Logistic_Regression": [logistic_train_score, logistic_test_score],
"Nearest Neighbor": [neigh_train_score, neigh_test_score],
"Random_Forest": [rf_train_score, rf_test_score]}
accuracy_scores_df = pd.DataFrame.from_dict(accuracy_scores, orient="index", columns=["Train Accuracy", "Test Accuracy"])
accuracy_scores_df.to_csv("best_models_accuracy_scores.csv")
