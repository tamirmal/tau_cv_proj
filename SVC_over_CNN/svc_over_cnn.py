import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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

# End of train_classifier
