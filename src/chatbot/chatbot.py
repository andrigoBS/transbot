from src.chatbot.network.dto.model_fit_data_dto import ModelFitDataDTO
from src.chatbot.network.dto.model_params_dto import ModelParamsDTO
from src.chatbot.network.model import Sec2SecModel
from src.chatbot.pln.helpers.database_pln import DatabasePLN
from src.chatbot.pln.helpers.text_pln import TextPLN
from src.chatbot.pln.vocabulary import Vocabulary


class Chatbot:
    def __init__(self):
        self.database = DatabasePLN()
        self.text_pln = TextPLN()
        self.vocabulary = Vocabulary()
        self.model = Sec2SecModel()
        self.questions = []
        self.answers = []

    def create(self, params: ModelParamsDTO):
        self.questions, self.answers = self.database.get_questions_and_answers()
        self.questions = self.text_pln.clear_and_tagging_phrases(
            self.questions,
            self.vocabulary.END_TAG
        )
        self.answers = self.text_pln.clear_and_tagging_phrases(self.answers, self.vocabulary.END_TAG)

        max_phrase_size = 0
        max_phrase = ''
        for phrase in self.questions + self.answers:
            length = len(phrase.split())
            if length > max_phrase_size:
                max_phrase_size = length
                max_phrase = phrase

        print('Max phrase length: {} From: {}'.format(max_phrase_size, max_phrase))
        self.vocabulary.create(self.questions, self.answers, max_phrase_size)
        vocab_size = self.vocabulary.get_size()
        print('Vocabulary Size: {}'.format(vocab_size))
        return self.model.create(max_phrase_size, vocab_size, params)

    def fit(self, params: ModelParamsDTO):
        encoder_input_data, decoder_input_data, decoder_output_data = self.vocabulary.phrases2data(
            self.questions, self.answers
        )

        return self.model.fit(
            ModelFitDataDTO(encoder_input_data, decoder_input_data, decoder_output_data),
            params
        )

    def execute(self, question, last_answer):
        last_answer = self.text_pln.clear_and_tagging_phrases(
            [last_answer],
            self.vocabulary.END_TAG
        )
        last_answer = self.vocabulary.phrase2ints(last_answer[0])

        new_phrase = []
        for word_pln in self.text_pln.split_pln(question):
            word = word_pln.text
            if not self.vocabulary.has_word(word):
                print('Não tem no dicionário: {}'.format(word))
                correct_word = self.text_pln.corrector(word)
                print('Correção: {}'.format(correct_word))
                if not self.vocabulary.has_word(correct_word):
                    print('Não tem a correção: {}'.format(correct_word))
                    new_word = self.text_pln.get_more_similarity(word, self.vocabulary.get_words())
                    print('Similaridade: {}'.format(new_word))
                    if not self.vocabulary.has_word(new_word):
                        print('Não tem a similaridade: {}'.format(new_word))
                        new_correct_word = self.text_pln.get_more_similarity(correct_word, self.vocabulary.get_words())
                        print('Similaridade com a correção: {}'.format(new_correct_word))
                        if not self.vocabulary.has_word(new_correct_word):
                            print('Não tem a similaridade com a correção: {}'.format(new_correct_word))
                        else:
                            new_phrase.append(new_correct_word)
                    else:
                        new_phrase.append(new_word)
                else:
                    new_phrase.append(correct_word)
            else:
                new_phrase.append(word)
        new_phrase = " ".join(new_phrase)
        new_phrase = self.text_pln.pos_tagging(new_phrase, self.vocabulary.END_TAG)
        question = self.vocabulary.phrase2ints(new_phrase)

        dec_outputs = self.model.predict_phrase(question, last_answer)
        words = self.vocabulary.ints2words(dec_outputs)
        return self.text_pln.words2phrase(words)

    def load(self):
        self.vocabulary.load()
        self.model.load()

    def plot_model(self, file_name):
        self.model.plot_model(file_name)
