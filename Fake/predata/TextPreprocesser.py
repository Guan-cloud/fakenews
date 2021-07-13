import re
from TurkishStemmer import TurkishStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *

# lang:语言，english或者turkish
# lower：将所有字符都转换为小写吗？是或否。示例：TextPreprocesser（lower = False）默认为True
# digits:数字会被删除吗？对或错。示例：TextPreprocesser（digits = False）默认为True
# link:链接会被删除吗？对或错。示例：TextPreprocesser（link = False）默认为True
# punc:标点符号会被删除吗？对或错。示例：TextPreprocesser（punc = False）默认为True。
# stem:会做词干吗？对或错。示例：TextPreprocesser（stem = False）默认为True。               ？？？？？？？？？？？？？？？
# stop_words:停用词会被删除吗？对或错。示例：TextPreprocesser（stop_words = False）默认为True。
# min_length_count:删除短于单词长度的字符。诠释示例：TextPreprocesser（min_length_count = 1）的默认值为2。

class TextPreprocesser:
    def __init__(self, lang="english", lower=True, digits=True, link=True, punc=True,
                 stem=True, stop_words=True, min_length_count=2):
        self.lower = lower
        self.digits = digits
        self.link = link
        self.punc = punc
        self.stem = stem
        self.stop_words = stop_words
        self.min_length_count = min_length_count
        self.stopwords = stopwords
        self.lang = lang
        if self.lang == "turkish":
        	self.stemmer = TurkishStemmer()
        else:
        	self.stemmer = PorterStemmer()

    def lower_turkish(self, text):
        new_text = re.sub('I', 'ı', text)
        new_text = new_text.lower()
        return new_text

    def lower_text(self, text):
        new_text = text.lower()
        return new_text

    def remove_digits(self, text):
        new_text = re.sub('[0-9]', ' ', text)
        return new_text

    def remove_blank(self, text):
        new_text = re.sub(r'\s\s+', ' ', text)
        return new_text

    def remove_link(self, text):
        new_text = re.sub(r'http\S+', '', text, flags=re.MULTILINE)
        new_text = re.sub(r'www\S+', '', new_text, flags=re.MULTILINE)
        return new_text

    def remove_punc(self, text, blank=True):
        if blank:
            new_text = re.sub(r'[^\w\s]', ' ', text)
        else:
            new_text = re.sub(r'[^\w\s]', '', text)
        return new_text

    def stem_word(self, text):
        new_text = self.stemmer.stem(text)
        return new_text

    def remove_short_words(self, text):
        new_text = ' '.join([word for word in text.split()
                             if len(word) > self.min_length_count])
        return new_text

    def remove_stop_words(self, text):
        if self.lang == "turkish":
            new_text = ' '.join([word for word in text.split() if word not in self.stopwords.words('turkish')])
        else:
            new_text = ' '.join([word for word in text.split() if word not in self.stopwords.words('english')])

        return new_text

    def stem_string(self, text):
        new_text = ' '.join([self.stem_word(word) for word in text.split()])
        return new_text

    def tokenize_word(self, text):
        new_text = word_tokenize(text)
        return new_text

    def tokenize_sentence(self, text):
        new_text = sent_tokenize(text)
        return new_text

    def string_to_preprocessed_string(self, text):
        new_text = text

        if self.lower:
            if self.lang == "turkish":
            	new_text = self.lower_turkish(new_text)
            else:
            	new_text = self.lower_text(new_text)
        if self.digits:
            new_text = self.remove_digits(new_text)
        if self.link:
            new_text = self.remove_link(new_text)
        if self.punc:
            new_text = self.remove_punc(new_text)
        if self.min_length_count:
            new_text = self.remove_short_words(new_text)
        if self.stem:
            new_text = self.stem_string(new_text)
        if self.stop_words:
            new_text = self.remove_stop_words(new_text)

        return new_text

    def string_to_sentence_tokenized_string_list(self, text):
        new_text = self.tokenize_sentence(text)
        return new_text

    def string_to_preprocessed_sentence_tokenized_string_list(self, text):
        new_text = self.tokenize_sentence(text)
        new_text = [self.string_to_preprocessed_string(word) for word in new_text]
        return new_text

    def string_to_word_tokenized_string_list(self, text):
        new_text = self.tokenize_sentence(text)
        new_text = self.sentence_tokenized_string_list_to_word_tokenized_string_list(new_text)
        return new_text

    def string_to_preprocessed_word_tokenized_string_list(self, text):
        new_text = self.string_to_preprocessed_sentence_tokenized_string_list(text)
        new_text = self.sentence_tokenized_string_list_to_word_tokenized_string_list(new_text)
        return new_text

    def sentence_tokenized_string_list_to_preprocessed_word_tokenized_string_list(self, text):
        new_text = text.copy()
        for i in range(len(new_text)):
            new_text[i] = self.string_to_preprocessed_sentence_tokenized_string_list(new_text[i])
        for i in range(len(text)):
            new_text[i] = new_text[i][0].split()

        return new_text

    def sentence_tokenized_string_list_to_word_tokenized_string_list(self, text):
        new_text = text.copy()
        for i in range(len(new_text)):
            new_text[i] = self.tokenize_word(new_text[i])

        return new_text

    def sentence_tokenized_string_list_to_string(self, text):
        text_copy = text.copy()
        new_text = ""
        for i in text_copy:
            new_text += i + " "

        return new_text
    
    def sentence_tokenized_string_list_to_preprocessed_sentence_tokenized_string_list(self, text):
        new_text = text.copy()
        for i in range(len(new_text)):
            new_text[i] = self.string_to_preprocessed_string(new_text[i])

        return new_text
    

    def sentence_tokenized_string_list_to_preprocessed_string(self, text):
        text_copy = text.copy()
        new_text = ""
        for i in text_copy:
            new_text += i + " "

        new_text = self.string_to_preprocessed_string(new_text)
        return new_text

    def word_tokenized_string_list_to_preprocessed_word_tokenized_string_list(self, text):
        new_text = text.copy()
        for i in range(len(new_text)):
            new_text[i] = ' '.join(new_text[i])
        new_text = self.sentence_tokenized_string_list_to_preprocessed_word_tokenized_string_list(
            new_text)
        
        return new_text
    
    def word_tokenized_string_list_to_sentence_tokenized_string_list(self, text):
        new_text = text.copy()
        for i in range(len(new_text)):
            new_text[i] = ' '.join(new_text[i])
        
        return new_text

    def word_tokenized_string_list_to_preprocesed_sentence_tokenized_string_list(self, text):
        new_text = text.copy()
        for i in range(len(new_text)):
            new_text[i] = ' '.join(new_text[i])
        
        new_text = self.sentence_tokenized_string_list_to_preprocessed_sentence_tokenized_string_list(new_text)
        
        return new_text

    def word_tokenized_string_list_to_string(self, text):
        new_text = text.copy()
        for i in range(len(new_text)):
            new_text[i] = ' '.join(new_text[i])
        
        # new_text = self.sentence_tokenized_string_list_to_preprocessed_sentence_tokenized_string_list(new_text)
        new_text = self.sentence_tokenized_string_list_to_string(new_text)
        
        return new_text

    def word_tokenized_string_list_to_preprocessed_string(self, text):
        new_text = text.copy()
        new_text = self.word_tokenized_string_list_to_string(new_text)
        new_text = self.string_to_preprocessed_string(new_text)
        
        return new_text






















