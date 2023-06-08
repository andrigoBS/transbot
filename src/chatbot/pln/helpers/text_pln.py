from autocorrect import Speller
import spacy
import string
import re


class TextPLN:
    def __init__(self):
        self.nlp = spacy.load('pt_core_news_lg')
        self.spell = Speller(lang='pt')

    def clear_and_tagging_phrases(self, phrases, end_tag):
        def clear_and_tag_item(phrase):
            phrase = self.clear_phrase(phrase)
            return self.pos_tagging(phrase, end_tag)
        return list(map(clear_and_tag_item, phrases))

    def clear_phrase(self, phrase):
        phrase = phrase.lower()
        phrase = re.sub('  +', ' ', phrase)
        phrase = re.sub('\w\.[ \n]', ' . ', phrase)
        return phrase

    def pos_tagging_regex_generic(self, phrase, tag_name, regex: re):
        regex_search = re.search(regex, phrase)
        if regex_search is None:
            return phrase

        index = regex_search.start()
        end_index = regex_search.end()
        tag_and_value = '<'+tag_name+'> '+phrase[index:end_index]+' </'+tag_name+'> '
        return phrase[:index] + tag_and_value + self.pos_tagging_phone(phrase[end_index:])

    def pos_tagging_phone(self, phrase):
        regex = '((\(\d{2}\) )?\d?\d{4}([ -])?\d{4})|(0800([ -])\d{3}([ -])\d{3})'
        return self.pos_tagging_regex_generic(phrase, 'phone', regex)

    def pos_tagging_zip(self, phrase):
        regex = '\d{5}-\d{3}'
        return self.pos_tagging_regex_generic(phrase, 'zip', regex)

    def pos_tagging_email(self, phrase):
        regex_name = '([a-z]|-|\d)+(\.([a-z]|-|\d)+)*'
        regex_doman = '([a-z]|-|\d)+(\.([a-z]|-|\d)+)*'
        regex = regex_name + '@' + regex_doman
        return self.pos_tagging_regex_generic(phrase, 'email', regex)

    def pos_tagging_url(self, phrase):
        regex_http = '(http|https)://'
        regex_doman = '([a-z]|-|\d)+(\.([a-z]|-|\d)+)*'
        regex_path = '(/([a-z]|-|\d)+)*'
        regex_params = '([?&#]([a-z]|-|\d)+=([a-z]|-|\d)+)*'
        regex = regex_http + regex_doman + regex_path + regex_params
        return self.pos_tagging_regex_generic(phrase, 'url', regex)

    def pos_tagging(self, phrase, end_tag):
        phrase = self.pos_tagging_phone(phrase)
        phrase = self.pos_tagging_zip(phrase)
        phrase = self.pos_tagging_email(phrase)
        phrase = self.pos_tagging_url(phrase)

        phrase_spacy = self.nlp(u"{}".format(phrase))
        new_phrase = ""
        open_tag = False
        concat = ''
        for word in phrase_spacy:
            if word.text == '<':
                open_tag = True
            if open_tag:
                concat += word.text
                if re.search('<([a-z])*>.*</([a-z])*>', concat) is not None:
                    end_index_open = re.search('<([a-z])*>', concat).end()
                    start_index_close = re.search('</([a-z])*>', concat).start()
                    new_phrase += ' ' + concat[:end_index_open] + ' ' + concat[end_index_open:start_index_close]
                    open_tag = False
                    concat = ''
                continue
            pos_tag = word.pos_.lower()
            new_phrase += " <{}> {}".format(pos_tag, word)
        return "{} {}".format(new_phrase, end_tag)

    def split_pln(self, phrase):
        phrase = self.clear_phrase(phrase)
        return self.nlp(u"{}".format(phrase))

    def get_similarity(self, word1, word2):
        word1_spacy = self.nlp(u"{}".format(word1))
        word2_spacy = self.nlp(u"{}".format(word2))
        return word1_spacy.similarity(word2_spacy)

    def get_more_similarity(self, word, words):
        max_similarity = {"similarity": 0, "word": None}
        for current_word in words:
            similarity = self.get_similarity(word, current_word)
            if similarity > max_similarity["similarity"]:
                max_similarity = {"similarity": similarity, "word": current_word}
        return max_similarity["word"]

    def corrector(self, phrase_or_word):
        return self.spell(phrase_or_word)

    def words2phrase(self, words):
        phrase = ''
        next_upper = True
        for word in words:
            if word[0] == '<' and word[-1] == '>':
                continue

            if word in string.punctuation:
                if word != ',':
                    next_upper = True
                phrase += word
                continue

            if word == '(' or word == ')':
                phrase += word
                continue

            if next_upper:
                word = word.capitalize()
                next_upper = False
            phrase += ' ' + word

        return phrase
