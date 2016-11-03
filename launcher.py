from dataset import Dataset
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rc('font', family='Arial')

class Launcher():
    def __init__(self):
        self.data = None

    def open_data(self, filename):
        self.data = Dataset(filename).data

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


if __name__ == "__main__":
    launcher = Launcher()
    launcher.open_data("KRD.csv")
    print(launcher.data)
    launcher.draw_user_avg()
    launcher.draw_modulation()
    plt.show()