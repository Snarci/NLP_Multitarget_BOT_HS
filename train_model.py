
from gettext import npgettext
import dataset_preprocessing
import dataset_manipolation
import numpy as np
#df = dataset_manipolation.get_cleaned_dataset(need_mean=0,testing_reduction=0,need_save=1)
import pandas as pd 

#import dataset preprocessed_mean_data.csv  
df = pd.read_csv('preprocessed_mean_data.csv')

#use a subset of the data for training

df = df.head(5000)

#show the first 5 rows of the dataframe.
print(df.head())

#replace nan with  empty string in the dataframe.
df = df.replace(np.nan, '', regex=True)

#print the number of nan values in the dataframe.
print(df.isna().sum())


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Text'])

# cast X to dataframe
X = pd.DataFrame(X.toarray())

#show the shape of the vectorized text column
print(X.shape)


#get the target column and cast it to a dataframe
Y = df['HD']
Y = pd.DataFrame(Y)

#split the data into training and testing sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, df, test_size=0.2, random_state=42)
#drop the text column from the training and testing sets labels.
y_train = y_train.drop(['Text'], axis=1)
y_test = y_test.drop(['Text'], axis=1)

#show the shape of the training and testing sets 
print(X_train.shape)
print(X_test.shape)

from sklearn.ensemble import RandomForestClassifier
#import knn classifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

#print first 5 rows of Y 
print(y_train.head())


from imblearn.under_sampling import RandomUnderSampler



from sklearn.metrics import classification_report
from imblearn.combine import SMOTEENN
#for each column in y_train, train a random forest classifier and print the accuracy score for each class.
for column in y_train.columns:
    rus = RandomUnderSampler(random_state=42)
    print("with RUS") 
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train[column])
    #print the sizes of the resampled sets.
    print("Nuova shape del training set: {}".format(X_resampled.shape))
    print("Vecchia shape del training set: {}".format(X_train.shape))
    #print number of elements per class in the training set.
    print("Nuovi elementi per classe: {}".format(y_resampled.value_counts()))
    clf = SVC(kernel='linear', C=1, random_state=42, class_weight='balanced')
    clf.fit(X_resampled, y_resampled)
    print(column)
    #print classification report for the classifier
    print(classification_report(y_test[column], clf.predict(X_test))) 
    print('\n')


#do the random undersampling on the training set.
 
# 0.24      0.70      0.36        93













'''
from imblearn.over_sampling import SMOTE
#for each column in y_train, train a random forest classifier and print the accuracy score for each class.
for column in y_train.columns:
    #SMOTE the data
    print("with SMOTE") 
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train[column])
    #train the random forest classifier
    print("Nuova shape del training set: {}".format(X_resampled.shape))
    print("Vecchia shape del training set: {}".format(X_train.shape))
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_resampled, y_resampled)
    print(column)
    #print classification report for the classifier
    print(classification_report(y_test[column], clf.predict(X_test))) 
    print('\n')
'''



