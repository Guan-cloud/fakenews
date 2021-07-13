# import nltk
# nltk.download('stopwords')
from TextPreprocesser import TextPreprocesser

TP = TextPreprocesser(lang="english", lower=True, digits=True, link=True, punc=True,
                 stem=True, stop_words=True, min_length_count=2)

test ="*Una gran noticia! Vacuna contra el virus Corona lista. Capaz de curar al paciente dentro de las 3 horas posteriores a la inyección. Felicitación a los científicos estadounidenses.*"

result = TP.string_to_preprocessed_string(test)

print(result)