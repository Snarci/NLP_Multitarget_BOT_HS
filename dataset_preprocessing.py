
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
#remove emoji from the text
def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                              "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

#apply all the above functions to the dataframe inside the preprocess_text function
def preprocess_text(df):
    df = df.apply(lambda x: remove_punctuations(x))
    df = df.apply(lambda x: to_lowercase(x))
    df = df.apply(lambda x: expand_contractions(x))
    #df = df.apply(lambda x: remove_stopwords(x))
    df = df.apply(lambda x: lemmatize_text(x))
    df = df.apply(lambda x: remove_words_with_numbers(x))
    df = df.apply(lambda x: remove_extra_spaces(x))
    df = df.apply(lambda x: remove_emoji(x))
    return df

def preprocess_text_test(text):
    text = remove_punctuations(text)
    text = to_lowercase(text)
    text = expand_contractions(text)
    #text = remove_stopwords(text)
    text = lemmatize_text(text)
    text = remove_words_with_numbers(text)
    text = remove_extra_spaces(text)
    text = remove_emoji(text)
    return text



#visualize first 10 elements of the dataset
#print(df.head(10))
