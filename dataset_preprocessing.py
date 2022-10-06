
#preprocess the text data in the dataframe by removing the punctuations and converting the text to lowercase
#expanding the contractions, removing the stopwords, and lemmatizing the text, removing words with numbers and removing extra spaces
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

#remove punctuations
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

#convert text to lowercase
def to_lowercase(text):
    return text.lower()


#remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)

#lemmatize text
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(w) for w in word_tokens]
    return ' '.join(lemmatized_text)

#expand contractions in text


def expand_contractions(text):
    return contractions.fix(text)

#remove words with numbers
def remove_words_with_numbers(text):
    return re.sub(r'\w*\d\w*', '', text)

#remove extra spaces
def remove_extra_spaces(text):
    return re.sub(' +', ' ', text)


#apply all the above functions to the dataframe inside the preprocess_text function
def preprocess_text(df, text_field):
    df[text_field] = df[text_field].apply(lambda x: remove_punctuations(x))
    df[text_field] = df[text_field].apply(lambda x: to_lowercase(x))
    df[text_field] = df[text_field].apply(lambda x: expand_contractions(x))
    df[text_field] = df[text_field].apply(lambda x: remove_stopwords(x))
    df[text_field] = df[text_field].apply(lambda x: lemmatize_text(x))
    df[text_field] = df[text_field].apply(lambda x: remove_words_with_numbers(x))
    df[text_field] = df[text_field].apply(lambda x: remove_extra_spaces(x))
    return df


#visualize first 10 elements of the dataset
#print(df.head(10))
