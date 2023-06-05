from autocorrect import Speller
import spacy
import string
import re


class TextPLN:
    def __init__(self):
        self.nlp = spacy.load('pt_core_news_lg')
        self.spell = Speller(lang='pt')

    def clear_punctuation_and_tagging_phrases(self, phrases, end_tag):
        def clear_and_tag_item(phrase):
            result = self.clear_punctuation_phrase(phrase)
            return self.pos_tagging(result, end_tag)

        return list(map(clear_and_tag_item, phrases))

    def clear_and_tagging_phrases(self, phrases, end_tag):
        def clear_and_tag_item(phrase):
            result = self.clear_phrase(phrase)
            return self.pos_tagging(result, end_tag)
        return list(map(clear_and_tag_item, phrases))

    def pos_tagging_phone(self, phrase):
        regex = re.search('((\(\d{2}\) )?\d?\d{4}([ -])?\d{4})|(0800([ -])\d{3}([ -])\d{3})', phrase)
        if regex is None:
            return phrase

        index = regex.start()
        end_index = regex.end()
        phone = '<phone> '+phrase[index:end_index]+' </phone> '
        return phrase[:index] + phone + self.pos_tagging_phone(phrase[end_index:])

    def pos_tagging(self, phrase, end_tag):
        phrase = re.sub('  +', ' ', phrase)
        phrase = self.pos_tagging_phone(phrase)

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

    def clear_phrase(self, phrase):
        phrase = phrase.lower()
        result = ''
        for char in phrase:
            if char in ['(', ')', '-']:
                result += char
                continue
            if char in string.punctuation:
                result += ' ' + char
                continue
            result += char
        return result

    def clear_punctuation_phrase(self, phrase):
        phrase = phrase.lower()
        result = ''
        for char in phrase:
            if char in ['(', ')', '-']:
                result += char
                continue
            if char in string.punctuation:
                continue
            result += char
        return result

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
