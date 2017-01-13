import pandas as pd
import numpy as np
from case_study import load_scrub_churn
from case_study import split_df
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split

df = load_scrub_churn()
y = df.pop('churn').values
X = df.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
rf = RandomForestClassifier(max_features='sqrt', n_estimators=100, n_jobs=-1, random_state=1)
gb = GradientBoostingClassifier(learning_rate=.1, n_estimators=200, max_features='sqrt', max_depth=5, random_state=1)
ab = AdaBoostClassifier(n_estimators=200)
plt.style.use('fivethirtyeight')


def stage_graph(model, model_name, X_test, y_test, scoring=accuracy_score, lw=2, color='k'):
    stage_preds = [pred for pred in model.staged_predict(X_test)]
    stage_lst = []
    score_lst = []
    for stage in np.arange(1, len(stage_preds)):
        stage_lst.append(stage)
        score_lst.append(scoring(y_test, stage_preds[stage]))
    ax.plot(stage_lst, score_lst, label=model_name, color=color)

def make_graph(score_method=accuracy_score):
    stage_graph(gb, 'Gradient Boost', X_test, y_test, score_method, color='k', lw=2)
    stage_graph(ab, 'AdaBoost', X_test, y_test, score_method, color='r', lw=2)
    ax.plot((0,200), (accuracy_score(y_test, rf.predict(X_test)), accuracy_score(y_test, rf.predict(X_test))), label='Random Forest', lw=2, color='y', linestyle='dashed')
    plt.xlabel('Stage')
    plt.legend()
    plt.show()

def make_feature_graph(model, model_name):
    feat = zip(df.columns, model.feature_importances_)
    feat = sorted(feat, key=lambda x: x[1], reverse=True)
    feat = zip(*feat)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    plt.bar(np.arange(0, len(model.feature_importances_)), feat[1], tick_label=feat[0], color='b', alpha=.6)
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.suptitle('{}'.format(model_name))
    plt.show()

def roc_curve(model, model_name):
    fpr, tpr, thresholds = roc_curve(y_test, model.predict(X_test))
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    plt.suptitle('{}'.format(model_name))
    plt.show()

if __name__ == "__main__":
    #sqrt, n_estimators: 50
    # Random Forest Accuracy: 0.768579492004
    # Random Forest Precision: 0.806872780447
    # Random Forest Recall: 0.828862660944

    # rf_params = {'n_estimators': [100, 500],
    #                 'max_features': ['sqrt', 'log2', 'auto', None],
    #                 'n_jobs': [-1],
    #                 'random_state': [1]}
    # print 'starting Random Forest grid search'
    # rf_grid = GridSearchCV(rf, rf_params, scoring='accuracy', n_jobs=-1, cv=5)
    # rf_grid.fit(X_train, y_train)
    # print rf_grid.best_params_
    # rf_opt = rf_grid.best_estimator_
    # print 'Random Forest Accuracy: {}'.format(rf_opt.score(X_test, y_test))
    # print 'Random Forest Precision: {}'.format(precision_score(y_test, rf_opt.predict(X_test)))
    # print 'Random Forest Recall: {}'.format(recall_score(y_test, rf_opt.predict(X_test)))

    gb.fit(X_train, y_train)
    ab.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    plt.ylabel('Accuracy')
    plt.suptitle('Accuracy Over Stages')
    make_graph()

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    plt.ylabel('Recall')
    plt.suptitle('Recall Over Stages')
    make_graph(recall_score)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    plt.ylabel('Precision')
    plt.suptitle('Precision Over Stages')
    make_graph(precision_score)

    make_feature_graph(rf, 'Random Forest Feature Importance')
    make_feature_graph(gb, 'Gradient Boost Feature Importance')
    make_feature_graph(ab, 'AdaBoost Feature Importance')

    # roc_curve(rf, 'Random Forest')
    # roc_curve(gb, 'Gradient Boost')
    # roc_curve(ab, 'AdaBoost')













    # gb_params = {'learning_rate': [.1, 1, 10],
    #                 'max_features': ['sqrt'],
    #                 'max_depth': [3, 5, 7],
    #                 'random_state': [1]}
    # print 'starting Gradient Boosting grid search'
    # gb_grid = GridSearchCV(gb, gb_params, scoring='accuracy', n_jobs=-1, cv=5)
    # gb_grid.fit(X_train, y_train)
    # gb_opt = gb_grid.best_estimator_
    # print gb_grid.best_params_
    # print 'Gradient Boosting Accuracy: {}'.format(gb_opt.score(X_test, y_test))
    # print 'Graident Boosting Precision: {}'.format(precision_score(y_test, gb_opt.predict(X_test)))
    # print 'Gradient Boosting Recall: {}'.format(recall_score(y_test, gb_opt.predict(X_test)))


    # ab_params = {'learning_rate': [.1, .5, 1, 10, 100],
    #                 'n_estimators': [10, 50, 100, 500],
    #                 'random_state': [1]}
    # print 'starting Adaboost Boosting grid search'
    # ab_grid = GridSearchCV(ab, ab_params, scoring='accuracy', n_jobs=-1, cv=5)
    # ab_grid.fit(X_train, y_train)
    # ab_opt = ab_grid.best_estimator_
    # print ab_grid.best_params_
    # print 'Adaboost Boosting Accuracy: {}'.format(ab_opt.score(X_test, y_test))
    # print 'Adaboost Boosting Precision: {}'.format(precision_score(y_test, ab_opt.predict(X_test)))
    # print 'Adaboost Boosting Recall: {}'.format(recall_score(y_test, ab_opt.predict(X_test)))
