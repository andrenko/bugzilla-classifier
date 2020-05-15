from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from helpers import preprocess, print_roc_curve, plot_table, explore
from keras_neural_networks import KerasNeuralNetwork


df = pd.read_csv('bugs-2019-11-25.csv')
# Make target lables from product and component values while dropping labels that have less than 10 occurencies
df['target'] = df[['Product', 'Component']].apply(' -- '.join, axis=1)
df = df.groupby('target').filter(lambda x: len(x) > 50)
df['target'] = df['target'].astype('category')
df['target_labels'] = df['target'].cat.codes

# Check that there are no missing summaries
print(f"Number of missing comments in comment text: {df['Summary'].isnull().sum()}")

# Explore categories
explore(df['target'], 40)

# Preprocess Summary dataset
print(f"Summary column before preprocessing:\n{df['Summary'].head()}")
df['Summary'] = preprocess(df['Summary'])
print(f"Summary column after preprocessing:\n{df['Summary'].head()}")

# Split dataset into train and test data
X = df[['Summary', 'Reporter', 'Assignee', 'OS']].apply(' '.join, axis=1)
y = df['target_labels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, shuffle=True)
print(f"Train dataset shape: {X_train.shape, y_train.shape}")
print(f"Test dataset shape: {X_test.shape, y_test.shape}")

# Initializing TfidfVectorizer. Using option stop_words=’english’ will stop considering common english words
vectorizer = TfidfVectorizer()

# Vectorize the train dataset
X_train_vectors = vectorizer.fit_transform(X_train)

# Vectorize the testing dataset
X_test_vectors = vectorizer.transform(X_test)

# Scaling of the training data
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler(with_mean=False)
# # Fit only to the training data
# X_train_sc = scaler.fit(X_train_vectors)
# # Apply the transformations to the data:
# X_train_vectors = X_train_sc.transform(X_train_vectors)
# X_test_vectors = X_train_sc.transform(X_test_vectors)

# ------------ Train sklearn classifiers ------------
# Train the Naive Bayes classifier
nb = MultinomialNB()
nb.fit(X_train_vectors, y_train)
# Make predictions
predicted = nb.predict(X_test_vectors)
f1_score_nb = f1_score(y_test, predicted, average='micro')

# Train the SVM, optimized by Stochastic Gradient Descent
svm = SGDClassifier()
svm.fit(X_train_vectors, y_train)
predicted_svm = svm.predict(X_test_vectors)
f1_score_svm = f1_score(y_test, predicted_svm, average='micro')

# Optimize the classifier by using Grid Search
# from sklearn.model_selection import GridSearchCV
# parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
#                   'tfidf__use_idf': (True, False),
#                   'clf__alpha': (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)}
# gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
# gs_clf_svm = gs_clf_svm.fit(X_train.Summary, y_train)
# print(gs_clf_svm.best_score_)
# print(gs_clf_svm.best_params_)

# Train the Boosting classifier
boosting = GradientBoostingClassifier(n_estimators=100)
boosting.fit(X_train_vectors, y_train)
predicted = boosting.predict(X_test_vectors)
f1_score_boosting = f1_score(y_test, predicted, average='micro')

# Train the k-nearest neighbors classifier
knn = KNeighborsClassifier()
knn.fit(X_train_vectors, y_train)
predicted = knn.predict(X_test_vectors)
f1_score_knn = f1_score(y_test, predicted, average='micro')

# Train the Bagging classifier
bagging = BaggingClassifier(KNeighborsClassifier())
bagging.fit(X_train_vectors, y_train)
predicted = bagging.predict(X_test_vectors)
f1_score_bagging = f1_score(y_test, predicted, average='micro')

# Train the Decision Tree classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_vectors, y_train)
predicted = decision_tree.predict(X_test_vectors)
f1_score_decision_tree = f1_score(y_test, predicted, average='micro')

# Train the Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train_vectors, y_train)
predicted = random_forest.predict(X_test_vectors)
f1_score_random_forest = f1_score(y_test, predicted, average='micro')

# Train the Multi-layer perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=400, activation='relu', solver='adam',
                    random_state=1, alpha=1e-5)
mlp.fit(X_train_vectors, y_train)
predicted = mlp.predict(X_test_vectors)
f1_score_MLP = f1_score(y_test, predicted, average='micro')

# ------------ Train keras neural networks classifiers ------------
# Train Convolution Neural Network with word embedding being learned from scratch with embedding layer
neural_network = KerasNeuralNetwork(X_train, X_test)
embedding_matrix = neural_network.build_embedding_matrix()
X_train_nn_vectors, X_test_nn_vectors = neural_network.tokenize()
model_CNN = neural_network.build_CNN(X_train_nn_vectors.shape[1], y.values.max() + 1)
# filepath_model_CNN = "CNN.{epoch:02d}-{val_accuracy:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath=filepath_model_CNN, monitor='val_accuracy', verbose=1, save_best_only=True,
# mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=2, mode='max')
model_CNN.fit(X_train_nn_vectors, y_train, validation_data=(X_test_nn_vectors, y_test), epochs=7, batch_size=32,
              verbose=2, callbacks=[early_stop])
predicted = model_CNN.predict(X_test_nn_vectors)
predicted = np.argmax(predicted, axis=1)
f1_score_cnn = f1_score(y_test, predicted, average='micro')

# Train CNN with Word2Vec word vectors being updated through training
model_CNN_word2vec_trainable = neural_network.build_CNN(X_train_nn_vectors.shape[1], y.values.max() + 1,
                                                        word2vec_weights=embedding_matrix, trainable=True)
filepath_model_CNN_word2vec = "CNN_word2vec.{epoch:02d}-{val_accuracy:.4f}.hdf5"
model_CNN_word2vec_trainable.fit(X_train_nn_vectors, y_train, validation_data=(X_test_nn_vectors, y_test), epochs=6,
                                 batch_size=32, verbose=2, callbacks=[early_stop])
predicted = model_CNN_word2vec_trainable.predict(X_test_nn_vectors)
predicted = np.argmax(predicted, axis=1)
f1_score_cnn_word2vec_trainable = f1_score(y_test, predicted, average='micro')

# Train CNN with static word vectors extracted from Word2Vec
model_CNN_word2vec_nontrainable = neural_network.build_CNN(X_train_nn_vectors.shape[1], y.values.max() + 1,
                                                           word2vec_weights=embedding_matrix, trainable=False)
model_CNN_word2vec_nontrainable.fit(X_train_nn_vectors, y_train, validation_data=(X_test_nn_vectors, y_test), epochs=4,
                                    batch_size=32, verbose=2, callbacks=[early_stop])
predicted = model_CNN_word2vec_nontrainable.predict(X_test_nn_vectors)
predicted = np.argmax(predicted, axis=1)
f1_score_cnn_word2vec_nontrainable = f1_score(y_test, predicted, average='micro')

# Train Long Short-Term Memory recurrent network
model_LSTM = neural_network.build_LSTM(y.values.max() + 1)
model_LSTM.fit(X_train_nn_vectors, y_train, validation_data=(X_test_nn_vectors, y_test), epochs=8, batch_size=32,
               verbose=2, callbacks=[early_stop])
predicted = model_LSTM.predict(X_test_nn_vectors)
predicted = np.argmax(predicted, axis=1)
f1_score_lstm = f1_score(y_test, predicted, average='micro')

# Train deep neural network
model_DNN = neural_network.build_DNN(X_train_vectors.shape[1], y.values.max() + 1)
model_DNN.fit(X_train_vectors, y_train, validation_data=(X_test_vectors, y_test), epochs=6, batch_size=32, verbose=2,
              callbacks=[early_stop])
predicted = model_DNN.predict(X_test_vectors)
predicted = np.argmax(predicted, axis=1)
f1_score_dnn = f1_score(y_test, predicted, average='micro')

# -------------- Print the results --------------
# Print a table with f1 scores for all classifiers
# F1 Score can be interpreted as a weighted average of the precision (True Positives/(True Positives + False Positives))
# and recall (True Positives/(True Positives + False Negatives)). It is created by finding the harmonic mean of
# precision and recall. F1 = 2 * (precision * recall) / (precision + recall)
all_f1_scores = [f1_score_nb, f1_score_svm, f1_score_MLP, f1_score_bagging, f1_score_boosting, f1_score_decision_tree,
                 f1_score_knn, f1_score_random_forest, f1_score_cnn, f1_score_cnn_word2vec_trainable,
                 f1_score_cnn_word2vec_nontrainable, f1_score_lstm, f1_score_dnn]
column_names = ['Naive Bayes', 'SVM', 'MLP', 'Bagging', 'Boosting', 'Decision tree', 'KNN', 'Random forest', 'CNN',
                'CNN_w2v_trainable', 'CNN_w2v_nontrainable', 'LSTM', 'DNN']
plot_table(all_f1_scores, ['accuracy (f1 score)'], column_names)

# Print roc curves for all output classes of the best performing classifier
print_roc_curve(y_test=y_test, y_predicted=predicted_svm, number_of_classes=len(set(y)), title='SVM')

# Print classification report for the best performing classifier
print(classification_report(y_test, predicted_svm, target_names=df['target'].cat.categories))
