from case_study import split_df, load_scrub_churn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
plt.style.use('ggplot')

def plot_svm_decision(model, X, y, label_sizes, name):
    # get the separating hyperplane
    w = model.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min(X[:,0]), max(X[:,0]))
    yy = a * xx - (model.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    margin = 1 / np.sqrt(np.sum(model.coef_ ** 2))
    yy_down = yy + a * margin
    yy_up = yy - a * margin

    # plot the line, the points, and the nearest vectors to the plane
    colors = ['red' if x else 'blue' for x in y]
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)

    ax.scatter(X[:,0], X[:,1], color=colors, s=label_sizes*40, alpha=0.5)
    ax.plot(xx, yy, 'k-')
    ax.plot(xx, yy_down, 'k--')
    ax.plot(xx, yy_up, 'k--')
    plt.ylim(-10,10)
    plt.title('SVM Decision Boundary %s' % name)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def decision_boundary(clf, X, Y, name, h=.02):
    """Inputs:
        clf - a trained classifier, with a predict method
    """
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(10, 7))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('%s' % name)
    plt.show()

'''
0 avg_dist
1 avg_rating_by_driver
2 avg_rating_of_driver
3 avg_surge
4 last_trip_date
5 signup_date
6 surge_pct
7 trips_in_first_30_days
8 luxury_car_user
9 weekday_pct
10 iphone
11 King's Landing
12 Winterfell
13 churn
 '''

def CVS(model, X, y):
    return np.mean(cross_val_score(model, X, y, scoring='accuracy', cv=5, n_jobs=-1))

def score(names, models, X, y):
    scores = []
    for model in models:
        scores.append(CVS(model, X, y))
    for i, score in enumerate(scores):
        print '{0} kernel svm model | accuracy: {1} '.format(names[i], score)

def plot_C(kernel, X, y):
    #create list of accuracies at varying Cs
    Cs = np.logspace(-3,1, num=5)
    accuracies = []
    for C in Cs:
        model = Pipeline([('scaler', StandardScaler()),
                             ('svc', SVC(C=C, kernel=kernel))])
        model.fit(X, y)
        accuracies.append(CVS(model, X, y))

    #plot result
    fig, ax1 = plt.subplots(1, figsize=(10,10))
    ax1.plot(Cs, accuracies, color='b')
    ax1.set_title('Varying Margin Size with C on different Kernels')
    ax1.set_xlabel('C')
    ax1.set_ylabel('Accuracy')
    plt.savefig('c_scores_rbf.jpg')
    plt.show()
    return accuracies

def zeros_ones(x):
    if x == True:
        return 1
    else:
        return 0

if __name__ == '__main__':

    df = load_scrub_churn()
    df['luxury_car_user'] = df['luxury_car_user'].apply(zeros_ones)
    df['churn'] = df['churn'].apply(zeros_ones)
    X_train, X_test, y_train, y_test = split_df(df)

    # kernels = linear, poly, rbf, sigmoid
    # svm_linear = Pipeline([('scaler', StandardScaler()),
    #                      ('svc', SVC(kernel='linear'))])
    # svm_linear.fit(X_train, y_train)

    svm_rbf = Pipeline([('scaler', StandardScaler()),
                         ('svc', SVC(kernel='rbf'))])
    svm_rbf.fit(X_train, y_train)
    '''
    rbf kernel svm model | accuracy: 0.768561548152
    '''

    '''
    cross_val accuracy
    rbf kernel svm model | accuracy: 0.768561548152
    linear kernel svm model | accuracy: 0.700535989692
    poly kernel svm model | accuracy: 0.734577496117


    rbf C=0.5 (on test data) | accuracy: 0.77025937374008868
    rbf C=1.0 (on test date) | accuracy: 0.77025937374008868
    rbf C=0.01 (on test date) | accuracy: 0.73861040182771132

    '''

    # svm_poly = Pipeline([('scaler', StandardScaler()),
    #                      ('svc', SVC(kernel='poly'))])
    # svm_poly.fit(X_train, y_train)
    # #
    # score(['rbf', 'linear', 'poly'],[svm_rbf, svm_linear, svm_poly], X_train, y_train)
    # score(['rbf'], [svm_rbf], X_train, y_train)

    plot_C('rbf', X_train, y_train)
