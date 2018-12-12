import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
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
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(y=Y, k=5))
    grid.fit(X, Y)
    print("The best classifier is: ", grid.best_estimator_)

    # plot the scores of the grid
    # grid_scores_ contains parameter settings and scores
    score_dict = grid.grid_scores_

    # We extract just the scores
    scores = [x[1] for x in score_dict]
    scores = np.array(scores).reshape(len(C_range), len(gamma_range))

    # Make a nice figure
    pl.figure(figsize=(8, 6))
    pl.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    pl.imshow(scores, interpolation='nearest', cmap=pl.cm.spectral)
    pl.xlabel('gamma')
    pl.ylabel('C')
    pl.colorbar()
    pl.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    pl.yticks(np.arange(len(C_range)), C_range)
    pl.show()

    return grid.best_estimator_

# End of train_classifier
