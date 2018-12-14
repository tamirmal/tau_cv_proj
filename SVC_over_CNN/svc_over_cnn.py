import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pylab as pl

def data_preprocessing(X):
    print("#### NOT DOING ANY PREPROCESSING NOW!! Consider changing this")
    return X
# End of data_preprocessing


def train_classifier(X, Y):
    X = data_preprocessing(X)
    # Trying grid-search to find best classifier
    C_range = 10. ** np.arange(-3, 8)
    gamma_range = 10. ** np.arange(-5, 4)
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(SVC(), param_grid=param_grid)

#    import ipdb; ipdb.set_trace()

    grid.fit(X, Y)
    print("The best classifier is: ", grid.best_params_)
    print("")
    print("The best classifier is: ", grid.best_estimator_)


    # plot the scores of the grid
    # grid_scores_ contains parameter settings and scores
    score_dict = grid.cv_results_

    # We extract just the scores
    print("Grid scores on training set:")
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    return grid.best_estimator_


def train_classifier_v2(X_train, X_test, y_train, y_test):
    print("Number of samples for train & validation = {}, samples for test {}".format(len(y_train), len(y_test)))

    param = [
        {
            "kernel": ["linear"],
            "C": [1, 10, 100, 1000]
        },
#        {
#            "kernel": ["rbf"],
#            "C": [1, 10, 100, 1000],
#            "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
#        }
    ]


    if True:
        # request probability estimation
        svm = SVC(probability=True)
        # 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
        clf = GridSearchCV(svm, param, cv=10, n_jobs=4, verbose=3)
        clf.fit(X_train, y_train)
        print("\nBest parameters set:")
        print(clf.best_params_)
        clf = clf.best_estimator_
    else:
        clf = SVC(kernel='linear', probability=True, C=1)
        clf.fit(X_train, y_train)

    print("Run on test set :")
    y_predict = clf.predict(X_test)
    labels = sorted(list(set(y_test)))
    print("\nConfusion matrix:")
    print("Labels: {0}\n".format(",".join(labels)))
    print(confusion_matrix(y_test, y_predict, labels=labels))

    print("\nClassification report:")
    print(classification_report(y_test, y_predict))

    return clf

# End of train_classifier
