import re
from src.chatbot.network.model import Sec2SecModel
from src.chatbot.pln.conversations import Conversations
from src.chatbot.pln.helpers.text_pln import TextPLN
from src.chatbot.pln.vocabulary import Vocabulary

singleton_text_pln = TextPLN()
singleton_conversations = Conversations(singleton_text_pln)
singleton_vocabulary = Vocabulary()


class Chatbot:
    def __init__(self):
        self.text_pln = singleton_text_pln
        self.vocabulary = singleton_vocabulary
        self.conversations = singleton_conversations

        self.model = Sec2SecModel()

    def create(self):
        self.conversations.load_and_process(self.vocabulary.END_TAG)
        questions, answers = self.conversations.get_all_questions_and_answers()

        max_phrase_size = 0
        max_phrase = ''
        for phrase in questions + answers:
            length = len(phrase.split())
            if length > max_phrase_size:
                max_phrase_size = length
                max_phrase = phrase

        print('Max phrase length: {} From: {}'.format(max_phrase_size, max_phrase))
        self.vocabulary.create(questions, answers, max_phrase_size)
        vocab_size = self.vocabulary.get_size()
        print('Vocabulary Size: {}'.format(vocab_size))

        return self.model.create(max_phrase_size, vocab_size)

    def fit(self, epochs, metrics, continue_fit=True):
        if continue_fit:
            self.conversations.load_and_process(self.vocabulary.END_TAG)
            self.conversations.get_all_questions_and_answers()

        def get_conversation_data(index):
            questions, answers = self.conversations.get_questions_and_answers_of_index(index)
            return self.vocabulary.phrases2data(questions, answers)

        return self.model.fit(
            get_conversation_data,
            self.conversations.get_size(),
            self.conversations.get_questions_len(),
            epochs,
            metrics,
            continue_fit
        )

    def execute(self, question, last_answer):
        last_answer = self.text_pln.clear_and_tagging_phrases(
            [last_answer],
            self.vocabulary.END_TAG
        )
        last_answer = self.vocabulary.phrases2ints(last_answer)

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
