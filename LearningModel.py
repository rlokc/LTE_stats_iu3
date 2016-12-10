from enum import Enum
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class ModelType(Enum):
    svc = 0
    randforest = 1
    logregression = 2
    dectree = 3

# TODO: use decorators for properties
class LearningModel:

    def __init__(self, model_type, learning_set):
        self.model = self.determine_model_type(model_type)
        self.cv_score = None
        # Remove the classifier from the feature list, if it contains it
        self.features, self.classifier = LearningModel.remove_classifier(learning_set)
        self.feature_names = self.features.columns
        self.fit_model()
        self.aggregate_cv_score()

    def fit_model(self):
        if self.model is not None:
            self.model.fit(self.features, self.classifier)
            # print (self.model.feature_importances_)

    def aggregate_cv_score(self):
        self.cv_score = cross_val_score(self.model, self.features, self.classifier, cv=5)

    @staticmethod
    def remove_classifier(learning_set):
        features, classifier = learning_set
        if type(classifier) is str:
            if classifier in features:
                features.drop(classifier, axis=1)
                classifier = features[classifier]
        else:
            print(features.columns)
            print(classifier.name)
            if classifier.name in features.columns:
                features = features.drop(classifier.name, axis=1)
        return (features, classifier)

    def determine_model_type(self, model_type):
        if model_type == ModelType.svc:
            self.model = SVC()
        elif model_type == ModelType.randforest:
            self.model = RandomForestClassifier()
        elif model_type == ModelType.logregression:
            self.model = LogisticRegression()
        elif model_type == ModelType.dectree:
            self.model = DecisionTreeClassifier()
        else:
            # TODO: make it into an exception
            print("No such type of regression")
            self.model = None
        return self.model
