import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from LearningModel import ModelType, LearningModel



'''
Wow this is a dirty hack
'''
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

    def create_model(self, model_type, arguments, features, classifier):
        self.models.append(LearningModel(model_type, arguments, (features, classifier)))


    def draw_model(self, model):
        pass






