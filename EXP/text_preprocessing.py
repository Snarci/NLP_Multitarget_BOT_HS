#functions to preprocess text data

import re
import string
import nltk
import contractions

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# function to lowercase a string
def lowercase(text):
    text_lower = text.lower()
    return text_lower

# function to remove punctuation from a string
def remove_punctuation(text):
    text_nopunct = ''.join([char for char in text if char not in string.punctuation])
    return text_nopunct

# function to remove stopwords from a string
def remove_stopwords(text):
    stopword = nltk.corpus.stopwords.words('english')
    text_nostop = ' '.join([word for word in text.split() if word not in stopword])
    return text_nostop

# function to lemmatize a string
def lemmatize(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    text_lemmatized = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text_lemmatized

# function to remove extra whitespaces from a string
def remove_extra_spaces(text):
    return re.sub(' +', ' ', text)

# function to remove hashtags from a string
def remove_hashtags(text):
    text_nohashtag = re.sub(r'#\w+', '', text)
    return text_nohashtag

# function to remove word with numbers from a string
def remove_words_with_numbers(text):
    return re.sub(r'\w*\d\w*', '', text)

# function to remove urls from a string
def remove_urls(text):
    text_nourl = re.sub(r'http\S+', '', text)
    return text_nourl

# function to remove emojis from a string
def remove_emojis(text):
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
    text_noemoji = emoji_pattern.sub(r'', text)
    return text_noemoji



# function to expand contractions in a string
def expand_contractions(text):
    return contractions.fix(text)

# funtion to preprocess a string
def preprocess_string(text):
    text = lowercase(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = expand_contractions(text)
    text = lemmatize(text)
    text = remove_extra_spaces(text)
    text = remove_hashtags(text)
    text = remove_words_with_numbers(text)
    text = remove_urls(text)
    text = remove_emojis(text)
    return text