import matplotlib.pyplot as plt
import numpy as np
from LearningModel import LearningModel

class Visualizer:


    @staticmethod
    def draw_class_scatter(model):
        h = 0.2
        plt.figure()
        X = model.features['PRB_DL_Used_Rate']
        X = np.column_stack((X, model.features['L_thrp_Kbps_Dl_aVG']))
        print(X)
        y = model.classifier
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # plt.show()

    @staticmethod
    def most_relevant_features(model):
        pass
        # return(set[], set[])
