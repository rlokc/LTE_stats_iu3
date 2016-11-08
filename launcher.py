from dataset import Dataset
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV


matplotlib.rc('font', family='Arial')

class Launcher():
    def __init__(self):
        self.dataset = None
        self.data = None
        self.features = None

    def open_data(self, filename, shuffle = False):
        self.dataset = Dataset(filename)
        self.data = self.dataset.data
        self.features = self.dataset.features
        if shuffle:
            np.random.shuffle(self.data)

    def draw_user_avg(self):
        users = self.data["L_Traffic_User_Avg"]
        plt.figure()
        plt.plot(np.arange(len(users)), users)
        plt.title("Среднее количество пользователей в секунду")
        plt.grid(True)

    def draw_modulation(self):
        x = np.arange(len(self.data["CQI_64QAM"]))
        plt.figure()
        plt.title("Процент использования разных видов модуляции")
        plt.plot(x, self.data["CQI_QPSK"])
        plt.plot(x, self.data["CQI_16QAM"])
        plt.plot(x, self.data["CQI_64QAM"])

    def perform_feature_selection(self, p):
        sel = VarianceThreshold(threshold=.6)
        self.new_features = sel.fit_transform(self.features)

    def model(self):
        self.model = SVC(kernel='linear')
        rfecv = RFECV(estimator=self.model, step=1, cv=StratifiedKFold(2),
                      scoring='accuracy')

    def regr_test(self):
        data = np.vstack([self.data["RRC_Conn_Ratio"], self.data["CQI_64QAM"]])
        plt.figure()
        plt.scatter(data[0], data[1])



if __name__ == "__main__":
    launcher = Launcher()
    launcher.open_data("KRD.csv")
    launcher.regr_test()
    # launcher.draw_user_avg()
    # launcher.draw_modulation()
    plt.show()