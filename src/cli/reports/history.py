from matplotlib import pyplot as plt

from src.file_path_helper import FilePathHelper


class History:
    def __init__(self):
        self.history = None

    def set_history(self, history):
        self.history = history
        return self

    def plot_accuracy(self):
        self.__base_plot('accuracy', 'Acurácia')
        return self

    def plot_loss(self):
        self.__base_plot('loss', 'Perda')
        return self

    def plot_precision(self):
        self.__base_plot('precision', 'Precisão')
        return self

    def plot_recall(self):
        self.__base_plot('recall', 'Recall')
        return self

    def __base_plot(self, plot_type, plot_title):
        print(self.history)
        plt.plot(self.history[plot_type])
        plt.plot(self.history['val_' + plot_type])
        plt.title(plot_title + ' por época')
        plt.ylabel(plot_title)
        plt.xlabel('Época')
        plt.legend(['Treino', 'Teste'], loc='upper left')
        plt.savefig(FilePathHelper().get_reports_file_path(plot_type))
        plt.close()
