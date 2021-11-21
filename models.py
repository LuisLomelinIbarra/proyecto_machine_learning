#!/usr/bin/env python3

''' Script for testing multiple models '''

from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc

from sklearn.tree import DecisionTreeClassifier



def test_models(X, y, n_classes):
    ''' Main script '''
    y = label_binarize(y, classes=[i for i in range(n_classes)])
    n_classes = y.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.60, random_state=42
    )

    names = [
        'knn', 'bayes', 'log reg', 'perceptron','decision_tree'
    ]

    clfs = [
        OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5)),
        OneVsRestClassifier(GaussianNB()),
        OneVsRestClassifier(LogisticRegression(max_iter=10000)),
        OneVsRestClassifier(Perceptron(max_iter=10000)),
        OneVsRestClassifier(DecisionTreeClassifier()),
        OneVsRestClassifier(LinearRegression())
    ]

    fig_count = 1
    colors = [
        'orangered', 'darkorange', 'gold', 'yellowgreen', 'darkcyan',
        'steelblue', 'slategrey', 'mediumslateblue', 'mediumorchid', 'deeppink']

    print('%15s%15s' % ('model'.center(15, ' '), 'precision'.center(15, ' ')))

    resscores = []

    for name, clf in zip(names, clfs):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test) * 100

        print('%15s %8.2f%%' % (name.center(15, ' '), score))

        if name in ['knn', 'bayes', 'decision_tree']:
            y_score = clf.predict_proba(X_test)

        else:
            y_score = clf.fit(X_train, y_train).decision_function(X_test)
        
        resscores.append([name,np.array(y_score).flatten()])
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(fig_count)

        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=1,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='black', linestyle=':', linewidth=2)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='black', linestyle=':', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve %s' % name)
        plt.legend(loc="lower right")
        plt.savefig('results/ROC%s.png' % name)
        fig_count += 1

    best = resscores[0]
    for res in resscores:
        if(best[0] != res[0] ):
            print("Result of "+best[0]+" vs "+ res[0]+ ":")
            stat,p = wilcoxon(best[1],res[1],alternative='greater')
            if(p<0.5):
                best = res
            print("The best is "+best[0])    

    print("The one that was ranked as best by all tests is "+best[0])
            

        

    plt.show()


def main():
    ''' Data preparation '''
    # open main table
    dataset = pd.read_csv('dataset/titles.csv')

    # dropping unuseful columns
    dataset = dataset.drop(
        ['titleType', 'primaryTitle', 'originalTitle', 'year', 'genres'],
        axis=1)

    ''' Languages '''

    # open dict_csv's and create values dictionaries
    langs_csv = pd.read_csv('dataset/titles_dict.csv')
    langs_dict = {key: index for index, key in enumerate(langs_csv.key)}

    # substitute strings with numerical value from dictionary

    dataset.language = [langs_dict[lang] for lang in dataset.language]

    ''' Genres '''

    # Open genres table
    genres = pd.read_csv('dataset/genres.csv')

    genres_csv = pd.read_csv('dataset/genres_dict.csv')
    genres_dict = {key: index for index, key in enumerate(genres_csv.key)}

    genres.genre = [genres_dict[g] for g in genres.genre]

    dataset = dataset.merge(genres, left_on='tconst', right_on='tconst')

    ''' Cast '''

    '''
    # Open genres table
    cast = pd.read_csv('dataset/cast.csv')

    cast_csv = pd.read_csv('dataset/cast_dict.csv')
    cast_dict = {key: index for index, key in enumerate(cast_csv.key)}

    cast.category = [cast_dict[cast] for cast in cast.category]

    dataset = dataset.merge(cast, left_on='tconst', right_on='tconst')
    '''


    ''' Ratings '''

    # open ratings table
    ratings = pd.read_csv('dataset/ratings.csv')
    dataset = dataset.merge(ratings, left_on='tconst', right_on='tconst')

    # preparing attributes and target metric
    y = dataset.averageRating.transform(lambda x: round(x / 2))
    X = dataset.drop(['tconst', 'averageRating'], axis=1)

    test_models(X, y, len(set(y)))


if __name__ == '__main__':
    main()
