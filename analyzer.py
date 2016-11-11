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
        self.column_names = None

    def open(self, filename, type=None):
        if type is None:
            type = determine_type(filename)
        if type == "xls":
            self.data = pandas.read_excel(filename)
        if type == "csv":
            self.data = pandas.read_csv(filename)

    def drop_unnecessary_columns(self, column_names):
        for column in column_names:
            self.data = self.data.drop(column, axis=1)

    def create_model(self, model_type, features, classifier, **kwargs):
        model = None
        # Remove the classifier from the feature list, if it contains it
        features, classifier = Stats_Analyzer.remove_classifier(features, classifier)
        # TODO: kwargs passing
        if model_type == ModelType.svc:
            model = SVC()
        elif model_type == ModelType.randforest:
            model = RandomForestClassifier()
        elif model_type == ModelType.logregression:
            model = LogisticRegression()
        if model is not None:
            model.fit(features, classifier)
            self.models.append(model)
        else:
            print("No such type of regression")

    def aggregate_cv_score(self, features, classifier, model_index = -1):
        features, classifier = Stats_Analyzer.remove_classifier(features, classifier)
        print(features)
        print(classifier)
        if model_index == -1:
            self.cv_scores = []
            for model in self.models:
                score = cross_val_score(model, features, classifier, scoring="roc_auc", cv=5)
                self.cv_scores.append(score)
        else:
            model = self.models[model_index]
            score = cross_val_score(model, features, classifier)
            self.cv_scores[model_index] = score

    @staticmethod
    def remove_classifier(features, classifier):
        if type(classifier) is str:
            if classifier in features:
                features.drop(classifier, axis = 1)
                classifier = features[classifier]
        else:
            if classifier.name in features.columns:
                features = features.drop(classifier.name, axis=1)
        return (features, classifier)




