import matplotlib.pyplot as plt
import numpy as np
from LearningModel import LearningModel
from matplotlib.colors import ListedColormap

class Visualizer:


    @staticmethod
    def draw_class_scatter(model):
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        cm = plt.cm.RdBu
        h = 1
        ax = plt.subplot()
        X = model.features['PRB_DL_Used_Rate']
        X = np.column_stack((X, model.features['L_thrp_Kbps_Dl_aVG']))
        print(X)
        y = model.classifier
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        clf = model.model
        stub = Visualizer.create_stub_values(model.features, ['PRB_DL_Used_Rate','L_thrp_Kbps_Dl_aVG'], xx, yy)
        print(stub)
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(stub)
        else:
            Z = clf.predict_proba(stub)[:, 1]
        len = model.features.shape[0]
        print(Z, Z.shape)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        ax.scatter(X[:,0], X[:,1], c = y, cmap=cm_bright)
        ax.set_xlim(xx.min()-1, xx.max()+1)
        ax.set_ylim(yy.min()-1000, yy.max()+1000)


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
    def most_relevant_features(model):
        pass
        # return(set[], set[])
