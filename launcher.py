from analyzer import ModelType
from analyzer import Stats_Analyzer
import numpy as np

# matplotlib.rc('font', family='Arial')

class Launcher():
    def __init__(self):
        self.analyzer = Stats_Analyzer()

    def load_data(self, filename):
        self.analyzer.open(filename)

    def test1(self):
        useless = ['RegionID', 'resulttime', 'CellID',  'cnt_averload_cell']
        analyzer.drop_unnecessary_columns(useless)
        analyzer.create_model(ModelType.svc,
                              analyzer.data,
                              analyzer.data['L_Traffic_User_Avg'].astype(np.float))

        analyzer.aggregate_cv_score(analyzer.data,
                                    analyzer.data['L_Traffic_User_Avg'].astype(np.float))
        print(analyzer.cv_scores)

    def test2(self):
        useless = ['RegionID', 'resulttime', 'CellID', 'L_Traffic_User_Avg']
        analyzer.drop_unnecessary_columns(useless)
        analyzer.create_model(ModelType.randforest,
                              analyzer.data,
                              analyzer.data['cnt_averload_cell'])

        analyzer.aggregate_cv_score(analyzer.data,
                                    analyzer.data['cnt_averload_cell'])
        print("Cross variation scores:\n" + str(analyzer.cv_scores))
        print("Feature importances:\n" + str(analyzer.models[0].feature_importances_))

if __name__ == "__main__":
    launcher = Launcher()
    filename = "KRD.xls"
    launcher.load_data(filename)
    analyzer = launcher.analyzer
    # launcher.test2()
    launcher.test1()

    # print(analyzer.data['DL_MCS_QPSK'].name)

