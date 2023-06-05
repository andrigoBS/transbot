import os


class FilePathHelper:
    def get_model_file_path(self):
        return self.__get_base_path('model_data/chatbot.h5')

    def get_vocabulary_file_path(self):
        return self.__get_base_path('model_data/tokenizer.pickle')

    def get_reports_file_path(self, filename):
        return self.__get_base_path('assets/'+filename+'.png')

    def get_base_file_path(self):
        return self.__get_base_path('database/**/*.docx')

    def __get_base_path(self, path):
        dirname = os.getcwd()
        index = dirname.find('src')
        if index != -1:
            dirname = dirname[0:index]
        filename = os.path.join(dirname, path)
        print(filename)
        return filename
