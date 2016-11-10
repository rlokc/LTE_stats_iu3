from analyzer import ModelType
from analyzer import Stats_Analyzer

# matplotlib.rc('font', family='Arial')

class Launcher():
    def __init__(self):
        self.analyzer = Stats_Analyzer()

    def load_data(self, filename):
        self.analyzer.open(filename)




if __name__ == "__main__":
    launcher = Launcher()
    filename = "KRD.xls"
    launcher.load_data(filename)