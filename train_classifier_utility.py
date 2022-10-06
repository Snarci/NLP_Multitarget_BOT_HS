import dataset_utility
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

need_balancing = True

#get dataset as a dataframe

def get_splitted_dataset_and_XY(df, label_name, corpus_name):
    #get the Labels column from df
    df_label = df[label_name]

    #get class distribution
    dataset_utility.get_label_frequency(df_label, verbose=True)

    #get the Corpus column from df
    df_corpus = df[corpus_name]

    #Applay the TF-IDF vectorizer to the corpus column
    vectorizer = TfidfVectorizer()

    #fit the vectorizer to the corpus column
    vectorizer.fit(df_corpus)

    #save the vectorizer for future use
    pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))
    #transform the corpus column
    df_corpus_tfidf = vectorizer.transform(df_corpus)

    #split the corpus column into train and test data
    X_train, X_test, y_train, y_test = train_test_split(df_corpus_tfidf, df_label, test_size=0.2, random_state=42)

    return  X_train, X_test, y_train, y_test

def train_classifier(X_train, y_train, classifier, need_balancing=True):
    #train the classifier
    #fit classifier using random undersampling boosting
    if need_balancing:
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        print("Nuova shape del training set: {}".format(X_resampled.shape))
        print("Vecchia shape del training set: {}".format(X_train.shape))
        classifier.fit(X_resampled, y_resampled)
    else:
        classifier.fit(X_train, y_train)
    train_classifier = classifier
    return train_classifier

def extract_stats(X_test, y_test, classifier):
    y_pred = classifier.predict(X_test)

    #get the report of the classifier
    print("Classification report:")
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

    #print accuracy of the classifier
    print("Classification accuracy:")
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_pred))

    #print Macro F1 score of the classifier
    print("Macro F1 score:")
    from sklearn.metrics import f1_score
    print(f1_score(y_test, y_pred, average='macro'))

    #get the confusion matrix of the classifier
    print("Classification CM:")
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix)

    #print auroc score of the classifier
    print("Classification AUROC:")
    from sklearn.metrics import roc_auc_score

    #enc_y_test=pd.Series(y_test, dtype="category").cat.codes.values

    #print(roc_auc_score(y_test,classifier.predict_proba(X_test), multi_class='ovr')) 

def save_classifier(classifier, filename):
    #save the classifier with pickle
    pickle.dump(classifier, open(filename, 'wb'))

def load_classifier(filename):
    #load the classifier with pickle
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def train_and_save_classifier(dataset_name, label_name, corpus_name, need_balancing=True):
    df = dataset_utility.get_dataset(name=dataset_name)
    X_train, X_test, y_train, y_test = get_splitted_dataset_and_XY(df, label_name, corpus_name)
    classifier = SVC(kernel='linear', C=1.0, random_state=42, class_weight='balanced',probability=True)
    #train the classifier
    classifier = train_classifier(X_train, y_train, classifier, need_balancing)
    #save the classifier
    filename= "./trained_classifiers/trained_classifier_"+label_name+".sav"
    save_classifier(classifier,filename)
    #extract stats
    classifier_loaded = load_classifier(filename)
    print("Extraendo stats per: " + label_name)
    extract_stats(X_test, y_test, classifier_loaded)
    return classifier

def agglomerate_columns(df, column_names):
    df_new = df[column_names]
    #print(df_new.head(15))
    #sum all the dataframe columns into one column
    df_new = df_new.sum(axis=1)
    #set to one al the rows with value greater than 1
    df_new[df_new > 1] = 1
    #print(df_new.head(15)) 
    return df_new

def get_splitted_dataset_and_XY_agglomerate(df, label_names, corpus_name):
    #get the Labels column from df
    df_label = agglomerate_columns(df, label_names)

    #get class distribution
    dataset_utility.get_label_frequency(df_label, verbose=True)

    #get the Corpus column from df
    df_corpus = df[corpus_name]

    #Applay the TF-IDF vectorizer to the corpus column
    vectorizer = TfidfVectorizer()

    #fit the vectorizer to the corpus column
    vectorizer.fit(df_corpus)
    #save the vectorizer for future use
    pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))
    
    #transform the corpus column
    df_corpus_tfidf = vectorizer.transform(df_corpus)

    #split the corpus column into train and test data
    X_train, X_test, y_train, y_test = train_test_split(df_corpus_tfidf, df_label, test_size=0.2, random_state=42)
    return  X_train, X_test, y_train, y_test

def train_and_save_classifier_agglomerate(dataset_name, label_name, corpus_name,agglomeration_name, need_balancing=True):
    df = dataset_utility.get_dataset(name=dataset_name)
    X_train, X_test, y_train, y_test = get_splitted_dataset_and_XY_agglomerate(df, label_name, corpus_name)
    classifier = SVC(kernel='linear', C=1.0, random_state=42, class_weight='balanced',probability=True)
    #train the classifier
    classifier = train_classifier(X_train, y_train, classifier, need_balancing)
    #save the classifier
    filename= "./trained_classifiers/trained_classifier_"+agglomeration_name+".sav"
    save_classifier(classifier,filename)
    #extract stats
    classifier_loaded = load_classifier(filename)
    print("Extraendo stats per: " + agglomeration_name)
    extract_stats(X_test, y_test, classifier_loaded)
    return classifier


def loop_train_and_save_classifier_for_all_classes():
    possible_targets=['Labels','None', 'African', 'Asian', 'Women', 
                'Caucasian', 'Jewish', 'Homosexual',
                'Islam', 'Hispanic', 'Arab', 'Refugee',
                'Economic', 'Other', 'Disability', 'Men',
                'Indian', 'Christian', 'Hindu', 'Indigenous',
                'Buddhism']
    remaining_targets=[ 'African', 'Asian', 'Women', 
                'Caucasian', 'Jewish', 'Homosexual',
                'Islam', 'Hispanic', 'Arab', 'Refugee',
                'Economic', 'Other', 'Disability', 'Men',
                'Indian', 'Christian', 'Hindu', 'Indigenous',
                'Buddhism']            
    #iterate for each possible target
    for i in range(0,len(possible_targets)): 
        train_and_save_classifier("expanded.csv", remaining_targets[i], "Corpus", need_balancing=True)

def apply_tfidf_on_text(text):
    #Applay the TF-IDF vectorizer to the corpus column
    vectorizer = TfidfVectorizer()
    #fit the vectorizer to the corpus column
    vectorizer.fit(text)
    #transform the corpus column
    text_tfidf = vectorizer.transform(text)
    return text_tfidf
         
