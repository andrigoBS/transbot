from docx import Document
import random
import glob

from src.file_path_helper import FilePathHelper


class DatabasePLN:
    def __init__(self):
        self.converastions = []

    def get_questions_and_answers(self):
        if len(self.converastions) == 0:
            self.__load()

        random.shuffle(self.converastions)

        questions = []
        answers = []

        for item in self.converastions:
            questions += item['questions']
            answers += item['answers']

        print("questions: {}, answers: {}".format(len(questions), len(answers)))

        return questions, answers

    def __load(self):
        list_of_files = glob.glob(FilePathHelper().get_base_file_path(), recursive=True)

        print('Read {} files'.format(len(list_of_files)))

        for path in list_of_files:
            print(path)
            together = self.__read_file(path)
            questions, answers = self.__split_questions_answers(together, path)
            if len(questions) > 0:
                self.converastions.append({'questions': questions, 'answers': answers})

        print('conversations: {}'.format(len(self.converastions)))

    def __read_file(self, path):
        f = open(path, 'rb')
        document = Document(f)
        f.close()
        return document.paragraphs

    def __split_questions_answers(self, together, file_name):
        questions = []
        answers = []
        last_item_is_question = False
        for item in together:
            item = item.text.strip()
            if item == '':
                continue

            index = item.find('<pessoa>:')
            if index != -1:
                item_processed = item.replace('<pessoa>:', '').strip()
                if item_processed != '':
                    questions.append(item_processed)
                last_item_is_question = True
            else:
                index = item.find('<chatbot>:')
                if index != -1:
                    item_processed = item.replace('<chatbot>:', '').strip()
                    if item_processed != '':
                        answers.append(item_processed)
                    last_item_is_question = False
                else:
                    if last_item_is_question:
                        questions[-1] += ' \n '+item
                    else:
                        answers[-1] += ' \n '+item
        if len(questions) != len(answers):
            raise Exception("Invalid file format: {}, questions: {}, answers: {}".format(
                file_name, len(questions), len(answers)
            ))
        return questions, answers
