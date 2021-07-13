# Text-Preprocesser
A simple preprocessing tool for natural language processing projects.

## How to Use

- Download TextPreprocesser.py
- import to another file: from TextPreprocesser import TextPreprocesser
- Create instance: TP = TextPreprocesser()
- Run the file. Load the required packages with a pip until there is no error.
- Review the examples below.

## Parameters

- lang: which language will be preprocessed English or Turkish.<br/>If you say what's the difference, the stopwords and the letter i, the difference.<br/>Example: TextPreprocesser(lang="turkish") default is english 
- lower: Will all characters be converted to lowercase? True or False <br/>Example: TextPreprocesser(lower=False) default is True. 
- digits: Will Numbers be deleted? True or False. <br/>Example: TextPreprocesser(digits=False) default is True. 
- link: Will links be deleted? True or False.  <br/>Example: TextPreprocesser(link=False) default is True. 
- punc: Will punctuation be deleted? True or False.  <br/>Example: TextPreprocesser(punc=False) default is True.
- stem: Will stemming be done? True or False.  <br/>Example: TextPreprocesser(stem=False) default is True.
- stop_words: Will stop words be deleted? True or False. <br/>Example: TextPreprocesser(stop_words=False) default is True.
- min_length_count: Deletes shorter than word length. int. <br/>Example: TextPreprocesser(min_length_count=1) default is 2.

## Examples

```python
from TextPreprocesser import TextPreprocesser

TP = TextPreprocesser(lang="english", lower=True, digits=True, link=True, punc=True,
                 stem=True, stop_words=True, min_length_count=2)

test = "I am not an Athenian or a Greek, but a citizen of the world. -Diogenes"

result = TP.string_to_preprocessed_string(test)

print(result)
# athenian greek citizen world diogen

TP2 = TextPreprocesser(lang="english", lower=True, digits=True, link=True, punc=True,
                 stem=False, stop_words=False, min_length_count=0)

test2 = "I am not an Athenian or a Greek, but a citizen of the world. -Diogenes ref:https://www.brainyquote.com/quotes/diogenes_107020"

result2 = TP2.string_to_preprocessed_string(test)

print(result2)
# i am not an athenian or a greek  but a citizen of the world   diogenes
```
### You can also convert between data structres.

```python
from TextPreprocesser import TextPreprocesser

TP = TextPreprocesser(lang="english", lower=True, digits=True, link=True, punc=True,
                 stem=False, stop_words=False, min_length_count=2)

text = """
Lorem Ipsum is simply dummy text of the printing and typesetting industry.
Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,
when an unknown printer took a galley of type and scrambled it to make a type specimen book.
It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged.
It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages,
and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
"""

tokenized = TP.string_to_word_tokenized_string_list(text)

print(tokenized)
#[['Lorem', 'Ipsum', 'is', 'simply', 'dummy', 'text', 'of', 'the', 'printing', 'and', 'typesetting', 'industry', '.'],
#['Lorem', 'Ipsum', 'has', 'been', 'the', 'industry', "'s", 'standard', 'dummy', 'text', 'ever', 'since', 'the', '1500s', ',', 'when', 'an', 'unknown', 'printer', 'took', 'a', 'galley', 'of', 'type', 'and', 'scrambled', 'it', 'to', 'make', 'a', 'type', 'specimen', 'book', '.'],
#['It', 'has', 'survived', 'not', 'only', 'five', 'centuries', ',', 'but', 'also', 'the', 'leap', 'into', 'electronic', 'typesetting', ',', 'remaining', 'essentially', 'unchanged', '.'],
#['It', 'was', 'popularised', 'in', 'the', '1960s', 'with', 'the', 'release', 'of', 'Letraset', 'sheets', 'containing', 'Lorem', 'Ipsum', 'passages', ',', 'and', 'more', 'recently', 'with', 'desktop', 'publishing', 'software', 'like', 'Aldus', 'PageMaker', 'including', 'versions', 'of', 'Lorem', 'Ipsum', '.']]

processed_sent_tokenized = TP.string_to_preprocessed_sentence_tokenized_string_list(text)

print(processed_sent_tokenized)
#['lorem ipsum simply dummy text the printing and typesetting industry', 
# 'lorem ipsum has been the industry standard dummy text ever since the when unknown printer took galley type and scrambled make type specimen book', 
# 'has survived not only five centuries but also the leap into electronic typesetting remaining essentially unchanged', 
# 'was popularised the with the release letraset sheets containing lorem ipsum passages and more recently with desktop publishing software like aldus pagemaker including versions lorem ipsum']

list_of_text = [ "Why is this so unbearable?" ,
"Why can’t I endure it?” You’ll be embarrassed to answer.",
"Then remind yourself that past and future have no power over you.",
"Only the present—and even that can be minimized.",
"Just mark off its limits." ]

plain_text = TP.sentence_tokenized_string_list_to_string(list_of_text)

print(plain_text)

#Why is this so unbearable? Why can’t I endure it?” You’ll be embarrassed to answer. Then remind yourself that past and future have no power over you. Only the present—and even that can be minimized. Just mark off its limits.

print(TP.string_to_sentence_tokenized_string_list(plain_text))

# ['Why is this so unbearable?', 'Why can’t I endure it?” You’ll be embarrassed to answer.', 'Then remind yourself that past and future have no power over you.', 'Only the present—and even that can be minimized.', 'Just mark off its limits.']

print(TP.sentence_tokenized_string_list_to_preprocessed_word_tokenized_string_list(list_of_text))
# [['why', 'this', 'unbearable'], ['why', 'can', 'endure', 'you', 'embarrassed', 'answer'], ['then', 'remind', 'yourself', 'that', 'past', 'and', 'future', 'have', 'power', 'over', 'you'], ['only', 'the', 'present', 'and', 'even', 'that', 'can', 'minimized'], ['just', 'mark', 'off', 'its', 'limits']]

```
### All Functions
```python


from TextPreprocesser import TextPreprocesser

TP = TextPreprocesser()

#Available Functions
TP.string_to_preprocessed_string()
TP.string_to_sentence_tokenized_string_list()
TP.string_to_preprocessed_sentence_tokenized_string_list()
TP.string_to_word_tokenized_string_list()
TP.string_to_preprocessed_word_tokenized_string_list()
TP.sentence_tokenized_string_list_to_preprocessed_word_tokenized_string_list()
TP.sentence_tokenized_string_list_to_word_tokenized_string_list()
TP.sentence_tokenized_string_list_to_string()
TP.sentence_tokenized_string_list_to_preprocessed_sentence_tokenized_string_list()
TP.sentence_tokenized_string_list_to_preprocessed_string()
TP.word_tokenized_string_list_to_preprocessed_word_tokenized_string_list()
TP.word_tokenized_string_list_to_sentence_tokenized_string_list()
TP.word_tokenized_string_list_to_preprocesed_sentence_tokenized_string_list()
TP.word_tokenized_string_list_to_string()
TP.word_tokenized_string_list_to_preprocessed_string()

#Or Single Operations
TP.lower_turkish()
TP.lower_text()
TP.remove_digits()
TP.remove_link()
TP.remove_punc()
TP.stem_word()
TP.remove_short_words()
TP.remove_stop_words()
TP.stem_string()
TP.tokenize_word()
TP.tokenize_sentence()
TP.remove_blank() # Converts multiple spaces into single space.
```
