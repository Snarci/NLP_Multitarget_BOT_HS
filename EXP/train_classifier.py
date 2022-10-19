import pandas as pd
import pickle
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

# if the dataset is unbalanced, we can use this flag to balance it
need_balancing = True

save_path = './Save/'

# function to split the dataset in train and test
def split_dataset(dataset_path, label_name ,test_size = 0.2):
    df = pd.read_csv(dataset_path)
    # apply TF-IDF to the corpus column
    tfidf = TfidfVectorizer()
    #save the vectorizer for future use
    pickle.dump(tfidf, open("tfidf.pickle", "wb"))
    corpus = tfidf.fit_transform(df['corpus'])
    # split the dataset in train and test
    X_train, X_test, y_train, y_test = train_test_split(corpus, df[label_name], test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# function to balance the dataset using RandomUnderSampler
def balance_dataset(X_train, y_train):
    rus = RandomUnderSampler(random_state=42)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    return X_train, y_train

# function to train the model
def train_model(dataset_path, label_name, test_size = 0.2, need_balancing = True):
    df = pd.read_csv(dataset_path)
    # split the dataset in train and test
    X_train, X_test, y_train, y_test = split_dataset(dataset_path, label_name, test_size)
    # declear the classifier
    classifier = SVC(kernel='linear', C=1, random_state=42, class_weight='balanced',probability=True)
    # if the dataset is unbalanced, we can use this flag to balance it using RandomUnderSampler
    if need_balancing:
        X_train, y_train = balance_dataset(X_train, y_train)
    # train the model
    classifier.fit(X_train, y_train)
    return classifier, X_test, y_test

# function to save the model
def save_model(model, model_name):
    pickle.dump(model, open(save_path+model_name+'.sav', 'wb'))

# function to train and save the model
def train_and_save_classifier(dataset_path, label_name, test_size = 0.2, need_balancing = True):
    model, X_test, y_test = train_model(dataset_path, label_name, test_size, need_balancing)
    save_model(model, label_name)
    test_model(model, X_test, y_test, label_name)

# function to load the model
def load_model(model_name):
    model = pickle.load(open(save_path+model_name+'.sav', 'rb'))
    return model

# function to train one model for each label
def train_and_save_classifiers(dataset_path, labels, test_size = 0.2, need_balancing = True):
    for label in labels:
        train_and_save_classifier(dataset_path, label, test_size, need_balancing)


# function to test the model and save result in csv file
def test_model(model, X_test, y_test, label_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    # calculate how many different labels can be predicted
    labels = list(set(y_test))
    print(labels)
    # save the result in csv file
    if(len(labels) == 2):
        df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred, 'y_pred_proba1': y_pred_proba[:,0],'y_pred_proba2': y_pred_proba[:,1]})
    else:
        df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred, 'y_pred_proba1': y_pred_proba[:,0],'y_pred_proba2': y_pred_proba[:,1],'y_pred_proba3': y_pred_proba[:,2]})
    df.to_csv(save_path+label_name+'_result.csv', index=False)
    # print the accuracy score
    print("accuracy score: ", accuracy_score(y_test, y_pred))
    # print the f1 score
    if(len(labels) == 2):
        print("f1 score: ", f1_score(y_test, y_pred))
    # save accuracy score, f1 score and auroc score in csv file
    if(len(labels) == 2):
        df = pd.DataFrame({'accuracy_score': [accuracy_score(y_test, y_pred)], 'f1_score': [f1_score(y_test, y_pred, average='weighted')]})
    else:
        df = pd.DataFrame({'accuracy_score': [accuracy_score(y_test, y_pred)], 'f1_score': [f1_score(y_test, y_pred, average='weighted')]})
    df.to_csv(save_path+label_name+'_score.csv', index=False)
    # make the consusion matrix based on the result
    cm = confusion_matrix(y_test, y_pred)
    # save the confusion matrix in csv file
    df = pd.DataFrame(cm)
    df.to_csv(save_path+label_name+'_confusion_matrix.csv', index=False) 

    