# src/text_preprocessing.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK wordnet...")
    nltk.download('wordnet')
    print("Downloading NLTK omw-1.4...")
    nltk.download('omw-1.4')


# Inicializar lematizador e stopwords globalmente neste módulo
# para que não sejam inicializados toda vez que a função preprocess_text for chamada.
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Função para pré-processar um texto:
    - Converte para minúsculas
    - Remove caracteres não-alfabéticos (exceto espaços)
    - Tokeniza
    - Remove stop words
    - Lematiza
    """
    text = text.lower() # Converter para minúsculas
    # Remove caracteres que não sejam letras (a-z) ou espaços (\s)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text) # Tokenização

    # Remover stop words e lematizar
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    return ' '.join(processed_tokens)


if __name__ == '__main__':
    print("Testando a função preprocess_text:")
    sample_comment = "Hello, this is a Sample comment! It's very TOXIC and annoying. @user #tag"
    processed_comment = preprocess_text(sample_comment)
    print(f"Original: {sample_comment}")
    print(f"Processado: {processed_comment}")