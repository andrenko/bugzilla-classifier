import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

df = pd.read_csv('bugs-2019-11-25.csv')
# Make target lables from product and component values while dropping labels that have less than 10 occurencies
df['target'] = df[['Product', 'Component']].apply('--'.join, axis=1)
df['target'] = df['target'].astype('category')
df = df.groupby('target').filter(lambda x: len(x) > 10)
df['target_labels'] = df['target'].cat.codes
# df.sort_values(by=['target_labels'], inplace=True)

# Split dataset into train and test data
df = df[['Summary', 'Reporter', 'OS', 'target_labels']]
y = df['target_labels']
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=123)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# Initializing TfidfVectorizer. Using option stop_words=’english’ will stop considering common english words
vectorizer = TfidfVectorizer(stop_words='english')

# Vectorize the train Summary sentences
X_train_vectors = vectorizer.fit_transform(X_train.Summary)

# Vectorize the testing data
X_test_vectors = vectorizer.transform(X_test.Summary)

# Train the SVM, optimized by Stochastic Gradient Descent
clf = MultinomialNB()
clf.fit(X_train_vectors, y_train)

# Make predictions
predicted = clf.predict(X_test_vectors)
accuracy = np.mean(predicted == y_test)

text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultinomialNB()), ])
text_clf = text_clf.fit(X_train.Summary, y_train)
predicted_1 = text_clf.predict(X_test.Summary)
accuracy_1 = np.mean(predicted_1 == y_test)


feature_names = vectorizer.get_feature_names()
df = pd.DataFrame(vectorizer.idf_, index=feature_names, columns=['idf'])
print(df)
df1 = pd.DataFrame(X_train_vectors.todense(), columns=feature_names)
print(df1)
# X = [df.Summary, df.Reporter, df.OS]
# y = df.Componenet
# df_bugzilla_train, df_bugzilla_test = train_test_split(df, test_size=0.2, random_state=123)
#
# print(df_bugzilla_train.shape)
# print(df_bugzilla_test.shape)
