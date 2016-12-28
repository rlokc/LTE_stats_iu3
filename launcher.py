from LearningModel import ModelType
from analyzer import Stats_Analyzer
import numpy as np
from Visualizer import Visualizer
from settings import Settings

# matplotlib.rc('font', family='Arial')

class Launcher():
    def __init__(self):
        self.analyzer = Stats_Analyzer()
        self.settings = Settings()

    '''
    Currently a stub, probably will add an interactive mode later
    '''
    def run(self):
        self.launch_state()

    def launch_state(self):
        print(self.settings.welcome_string)
        for function in self.settings.main_functions:
            print("{} -- {}".format(function, self.settings.function_descriptions[function]))


    def load_data(self, filename):
        self.analyzer.open(filename)

    def test1(self):
        useless = ['RegionID', 'resulttime', 'CellID',  'cnt_averload_cell']
        analyzer.drop_unnecessary_columns(useless)
        arguments = {"C": 0.8, "kernel": 'linear'}
        analyzer.create_model(ModelType.svc,
                              arguments,
                              analyzer.data,
                              analyzer.data['L_Traffic_User_Avg'].astype(np.float))

        print("Cross variation score:\n" + str(analyzer.models[0].cv_score))


    def test2(self):
        useless = ['RegionID', 'resulttime', 'CellID','PRB_DL_Used_Rate']
        analyzer.drop_unnecessary_columns(useless)
        arguments = {"C": 0.3}
        analyzer.create_model(ModelType.logregression,
                              arguments,
                              analyzer.data,
                              analyzer.data['cnt_averload_cell'])

        print("Cross variation score:\n" + str(analyzer.models[0].cv_score))
        # print("Feature importances:\n" + str(analyzer.models[0].model.feature_importances_))
        print(analyzer.models[0].feature_names)
        Visualizer.draw_class_scatter(analyzer.models[0])

    def test3(self):
        classifier = "DL_MCS_64QAM"
        # TODO: redo it, so it doesn't delete the classifier from the dataset, what if you want to reuse it?
        classifier_values = analyzer.data[classifier]
        arguments = {"C": 0.8, "kernel":'linear'}
        useless = ['RegionID', 'resulttime', 'CellID', 'cnt_averload_cell', classifier]
        analyzer.drop_unnecessary_columns(useless)
        analyzer.create_model(ModelType.svc,
                              arguments,
                              analyzer.data,
                              classifier_values)

        print("Cross variation score:\n" + str(analyzer.models[0].cv_score))
        # print("Feature importances:\n" + str(analyzer.models[0].model.feature_importances_))
        print(analyzer.models[0].feature_names)
        Visualizer.draw_class_scatter(analyzer.models[0])

if __name__ == "__main__":
    launcher = Launcher()
    # Launch interactive mode
    # TODO: make a quick way to pass all the parameters through the state machine (args would be super useful)
    # launcher.run()
    filename = "KRD.xls"
    launcher.load_data(filename)
    analyzer = launcher.analyzer
    launcher.test2()
    # launcher.test1()

    # print(analyzer.data['DL_MCS_QPSK'].name)

