from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, pad_sequences
import pickle
import numpy as np
from src.file_path_helper import FilePathHelper


class Vocabulary:
    def __init__(self):
        self.tokenizer = None
        self.max_phrase_size = 0
        self.serialized_file_name = FilePathHelper().get_vocabulary_file_path()
        self.END_TAG = '<eos>'

    def create(self, questions, answers, max_phrase_size):
        self.tokenizer = Tokenizer(filters='')
        self.tokenizer.fit_on_texts(questions + answers)
        self.max_phrase_size = max_phrase_size
        self.save()

    def phrases2data(self, questions, answers):
        encoder_input_data = self.phrases2ints(questions)
        decoder_input_data = self.phrases2ints([self.END_TAG] + answers[:len(answers)-1])
        decoder_output_data = self.phrases2ints(answers)

        return encoder_input_data, decoder_input_data, decoder_output_data

    def phrases2ints(self, phrases):
        tokens_list = self.tokenizer.texts_to_sequences(phrases)
        padded_phrase = self.__pad_sequences(tokens_list)
        return to_categorical(padded_phrase, self.get_size())

    def phrase2ints(self, phrase):
        return self.phrases2ints([phrase])

    def ints2words(self, ints):
        words = []
        # (1, n, m)[0] = (n, m)
        for word_sequences in ints[0]:
            sampled_word_index = np.argmax(word_sequences)
            # se é uma palavra desconhecida
            if sampled_word_index == 0:
                continue
            word = self.tokenizer.index_word[sampled_word_index]
            if word == self.END_TAG:
                break
            words.append(word)
        return words

    def get_size(self):
        # soma 1 por causa do 0 que sisgnifica palavras desconhecida
        return len(self.tokenizer.word_index) + 1

    def has_word(self, word):
        return self.tokenizer.word_index.get(word) is not None

    def get_words(self):
        return self.tokenizer.word_index

    def load(self):
        if self.tokenizer is not None:
            return

        with open(self.serialized_file_name, 'rb') as handle:
            data = pickle.load(handle)
            self.tokenizer = data["tokenizer"]
            self.max_phrase_size = data["max_phrase_size"]

    def save(self):
        with open(self.serialized_file_name, 'wb') as handle:
            data = {
                "tokenizer": self.tokenizer,
                "max_phrase_size": self.max_phrase_size
            }
            pickle.dump(data, handle)

    def __pad_sequences(self, sequences):
        return pad_sequences(sequences, maxlen=self.max_phrase_size, value=0, padding='post')
