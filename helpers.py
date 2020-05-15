from itertools import cycle
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer


def preprocess(text):
    def _remove_stopwords(sentence):
        stop_words = stopwords.words('english')
        return " ".join([word for word in sentence if word not in stop_words])
    # remove unwanted characters, numbers and symbols
    text = text.str.replace("[^a-zA-Z#]", " ")
    # remove short words (length < 3)
    text = text.apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
    # remove stopwords from the text
    text = [_remove_stopwords(sentence.split()) for sentence in text]
    # make entire text lowercase
    text = [sentence.lower() for sentence in text]
    return text


def explore(dataframe, head):
    df_count_values = dataframe.value_counts().reset_index()
    df_count_values.columns = ['category', 'ocurrencies']
    df_count_values.head(head).plot(x='category', y='ocurrencies', kind='bar', legend=False, grid=True, figsize=(10, 6))
    plt.title("Number of ocurrencies per category")
    plt.ylabel('number of occurrences', fontsize=9)
    plt.xlabel('category', fontsize=9)
    plt.show()


def print_roc_curve(y_test, y_predicted, number_of_classes, title):
    # Binirize target labels
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_score = lb.transform(y_predicted)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(number_of_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(number_of_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(number_of_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= number_of_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(number_of_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver operating characteristic for output classes of {title} classifier')
    plt.legend(loc="lower right", prop={'size': 5})
    plt.show()


def plot_table(data, rows, columns):
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    df = pd.DataFrame([data], columns=columns)
    df.update(df.applymap('{:,.3f}'.format))

    table = ax.table(cellText=df.values, rowLabels=rows, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    fig.tight_layout()

    plt.show()
