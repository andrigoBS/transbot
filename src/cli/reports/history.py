from matplotlib import pyplot as plt

from src.file_path_helper import FilePathHelper


class History:
    def __init__(self):
        self.history = None

    def set_history(self, history):
        self.history = history
        return self

    def plot_accuracy(self):
        self.__base_plot('accuracy')
        return self

    def plot_loss(self):
        self.__base_plot('loss')
        return self

    def plot_precision(self):
        self.__base_plot('precision')
        return self

    def plot_recall(self):
        self.__base_plot('recall')
        return self

    def __base_plot(self, plot_type):
        plt.plot(self.history[plot_type])
        plt.plot(self.history['val_' + plot_type])
        plt.title('model ' + plot_type)
        plt.ylabel(plot_type)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(FilePathHelper().get_reports_file_path(plot_type))
        plt.close()
