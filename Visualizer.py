import matplotlib.pyplot as plt
import numpy as np
from LearningModel import LearningModel
from matplotlib.colors import ListedColormap

class Visualizer:


    @staticmethod
    def draw_class_scatter(model):
        # Color schemes for the plot, bright for the points, normal for the probability contour
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        cm = plt.cm.RdBu
        # Step of the contour plot calculation
        h = 10

        ax = plt.subplot()
        # Creating the coordinates matrix
        relevant_features = Visualizer.get_n_relevant_features(model, 2)
        features = [model.features[feature] for feature in relevant_features]
        X = np.column_stack(features)
        y = model.classifier
        # Set the boundaries of our plot
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        # Create the meshgrid for the contour plot calculation
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        clf = model.model
        # Since our model uses n features, but we can plot only for 2
        # we need to create a big matrix for n features, with only 2 relevant features included,
        # all the other one we fill with mean values
        stub = Visualizer.create_stub_values(model.features, relevant_features, xx, yy)
        # print(stub)
        # Depending on the clacifier type, call different functions for the probability countoure
        # for some reason, decision_function isn't always accurate (especially for SCV models)
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(stub)
        else:
            Z = clf.predict_proba(stub)[:, 1]
        Z = Z.reshape(xx.shape)

        #Draw the calculated probability contour and the points
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        ax.scatter(X[:,0], X[:,1], c = y, cmap=cm_bright)
        # Set the boundaries
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())

        ax.set_xlabel(relevant_features[0])
        ax.set_ylabel(relevant_features[1])

        plt.show()


    '''
    Returns an array of 2 passed features listed as-is
    With all the others replaced by their means
    '''
    @staticmethod
    def create_stub_values(features, valuable, xx, yy):
        res = None
        len = xx.shape[0] * xx.shape[1]
        for feature in features:
            values = features[feature]
            if feature not in valuable:
                if res is None:
                    res = np.full((len, 1),values.mean())
                else:
                    res = np.c_[res, np.full((len, 1),values.mean())]
            else:
                if feature == valuable[0]:
                    values = xx
                else:
                    values = yy
                if res is None:
                    res = values.ravel()
                else:
                    res = np.c_[res, values.ravel()]
        return res

    @staticmethod
    def get_n_relevant_features(model, n):
        if hasattr(model.model, "feature_importances_"):
            scores = model.model.feature_importances_
            # sorted_features = sorted(model.features, key=lambda score: scores)
            sorted_features = [feature for (score, feature) in sorted(zip(scores, model.features),reverse=True)]
            # print("Features and scores:", model.features.columns, scores, "\n Sorted by score:", sorted_features)
            # print("Relevant features: ", sorted_features[:n])
            return sorted_features[:n]
        else:
            print("MODEL DOESN'T HAVE FEATURE_IMPORTANCES_ FIELD")
            return None