from enum import Enum
import pandas
import seaborn
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class ModelType(Enum):
    svc = 0
    randforest = 1
    logregression = 2


def determine_type(filename):
    return filename[-3:]


class Stats_Analyzer():

    def __init__(self):
        self.data = None
        self.data_filename = None
        self.models = []
        self.cv_scores = []

    def open(self, filename, type=None):
        if type is None:
            type = determine_type(filename)
        if type == "xls":
            self.data = pandas.read_excel(filename)
        if type == "csv":
            self.data = pandas.read_csv(filename)
        print("succ")

    def drop_unnecessary_columns(self, column_names):
        for column in column_names:
            self.data = self.data.drop(column, axis=1)

    def create_model(self, type, features, classifier, **kwargs):
        model = None
        # Remove the classifier from the feature list, if it contains it
        # TODO: also check for objects
        if type(classifier) is str:
            if classifier in features:
                features.drop(classifier, axis = 1)
        # TODO: kwargs passing
        if type == ModelType.svc:
            model = SVC()
        elif type == ModelType.randforest:
            model = RandomForestClassifier()
        elif type == ModelType.logregression:
            model = LogisticRegression()
        if model is not None:
            model.fit(features, classifier)
            self.models.append(model)
        else:
            print("No such type of regression")

    def aggregate_cv_score(self, features, classifier, model_index = -1):
        if model_index == -1:
            self.cv_scores = []
            for model in self.models:
                score = cross_val_score(model, features, classifier)
                self.cv_scores.append(score)
        else:
            model = self.models[model_index]
            score = cross_val_score(model, features, classifier)
            self.cv_scores[model_index] = score

    def print_cv_scores(self):
        for score in self.cv_scores:
            print(score)




