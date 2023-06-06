from src.chatbot.chatbot import Chatbot
from src.chatbot.network.dto.model_params_dto import ModelParamsDTO
from src.cli.reports.history import History
from src.cli.reports.summary import Summary
from src.file_path_helper import FilePathHelper


class ChatbotCli:
    def __init__(self):
        self.chatbot = Chatbot()
        self.summary_report = Summary()
        self.history_report = History()
        self.params = ModelParamsDTO()

    def fit(self):
        summary = self.chatbot.create(self.params)

        self.summary_report.set_summary(summary).plot_img()

        summary_string = self.summary_report.get_text()
        print(summary_string)

        self.chatbot.plot_model(FilePathHelper().get_reports_file_path('schema'))

        result = self.chatbot.fit(self.params)

        print(result.history.keys())
        self.history_report \
            .set_history(result.history) \
            .plot_accuracy() \
            .plot_loss() \
            .plot_precision() \
            .plot_recall()

    def main(self):
        self.chatbot.load()
        sentence = input('Você : ')
        last_result = ''
        while sentence not in ['adeus', 'tchau', 'bye', 'falou', 'flw']:
            last_result = self.chatbot.execute(sentence, last_result)
            print('Chatbot: ' + last_result)
            sentence = input('Você : ')


if __name__ == "__main__":
    cli = ChatbotCli()
    cli.fit()
    cli.main()
