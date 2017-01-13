import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score
from case_study import load_scrub_churn, split_df


def load_data():
    df = load_scrub_churn()
    X_train, X_test, y_train, y_test = split_df(df)
    return df, X_train, X_test, y_train, y_test


def print_scores(y_train, y_test, train_pred, test_pred):
    print 'Train Accuracy:', accuracy_score(y_train, train_pred)
    print 'Train Recall:', recall_score(y_train, train_pred)
    print 'Train Precision:', precision_score(y_train, train_pred)
    print 'Test Accuracy:', accuracy_score(y_test, test_pred)
    print 'Test Recall:', recall_score(y_test, test_pred)
    print 'Test Precision:', precision_score(y_test, test_pred)


def knn_gridsearch_fit(X_train, y_train):
    knn_grid = {'n_neighbors': [31, 33, 35, 37, 39, 41, 43, 45],
                'weights': ['uniform'],
                'algorithm': ['auto'],
                'leaf_size': [1, 5, 10],
                'metric': ['euclidean']}
    knn_gridsearch = GridSearchCV(KNeighborsClassifier(),
                                  knn_grid,
                                  verbose=True,
                                  n_jobs=-1,
                                  scoring='accuracy')
    knn_gridsearch.fit(X_train, y_train)
    knn_optimized = knn_gridsearch.best_estimator_
    print knn_gridsearch.best_params_
    return knn_optimized


def knn_fit(X_train, y_train):
    knn = KNeighborsClassifier(
        n_neighbors=41, metric='euclidean', weights='uniform', leaf_size=5)
    knn.fit(X_train, y_train)
    return knn


def roc_plot(model, X_train, X_test, y_train, y_test):
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    roc_train = model.predict_proba(X_train)[:, 1]
    roc_test = model.predict_proba(X_test)[:, 1]
    pos_train = y_train.sum()
    pos_test = y_test.sum()
    neg_train = len(y_train) - pos_train
    neg_test = len(y_test) - pos_test
    tpr_train, tpr_test = [], []
    fpr_train, fpr_test = [], []
    # thresholds = []
    prob_train = np.append(roc_train.reshape(len(roc_train), 1),
                           y_train.reshape(len(y_train), 1), axis=1)
    s_probs_train = prob_train[prob_train[:, 0].argsort()][::-1]
    for idx, probability in enumerate(s_probs_train):
        # thresholds.append(probability[0])

        tpr_train.append(float(s_probs_train[0:idx + 1, 1].sum()) / pos_train)
        fpr_train.append(
            float(len(s_probs_train[0:idx + 1, 1]) - s_probs_train[0:idx + 1, 1].sum()) / neg_train)
    prob_test = np.append(roc_test.reshape(len(roc_test), 1),
                          y_test.reshape(len(y_test), 1), axis=1)
    s_probs_test = prob_test[prob_test[:, 0].argsort()][::-1]
    for idx, probability in enumerate(s_probs_test):
        # thresholds.append(probability[0])

        tpr_test.append(float(s_probs_test[0:idx + 1, 1].sum()) / pos_test)
        fpr_test.append(
            float(len(s_probs_test[0:idx + 1, 1]) - s_probs_test[0:idx + 1, 1].sum()) / neg_test)

    ax.plot(fpr_train, tpr_train, lw=2, label='Train Accuracy')
    ax.plot(fpr_test, tpr_test, lw=2, label='Test Accuracy')
    ax.plot([0, 1], [0, 1], '--k', lw=2)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity, Recall)")
    ax.set_title('ROC Plot of KNN Model')
    plt.legend(loc='best')
    plt.show()


def run_model():
    df, X_train, X_test, y_train, y_test = load_data()
    knn = knn_fit(X_train, y_train)
    train_pred = knn.predict(X_train)
    test_pred = knn.predict(X_test)
    print_scores(y_train, y_test, train_pred, test_pred)
    roc_plot(knn, X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    run_model()
