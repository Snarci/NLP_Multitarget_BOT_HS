from cProfile import label
import pandas as pd
import json
from collections import Counter
import text_preprocessing

# function to read a json file and convert it to a csv file
def json_to_csv(json_path = 'hatexplain.json', csv_path = 'hatexplain.csv'):
    df = pd.read_json(json_path,orient = 'index')
    df.to_csv(csv_path, index = False)

# function to read a csv file and delete a column of it by its name
def delete_column(df, column_name = 'Unnamed: 0'):
    df.drop(column_name, axis = 1, inplace = True)
    return df

# function to read a string and make an object from it using majority voting
def string_to_objJson(string):
    string = string.replace("'", '"')
    res = json.loads(string)
    labels = []
    target = []
    for i in range(0,3):
        labels.append(res[i]['label'])
        target.append(res[i]['target'])

    countLabels = Counter(labels)
    for key, value in countLabels.items():
        if value > 1:
            label = key
            break
    else:
        label = 'None'

    target = [item for sublist in target for item in sublist]
    countTarget = Counter(target)
    targets = []
    for key, value in countTarget.items():
        if value > 1:
            targets.append(key)

    if len(targets) == 0:
        obj = {'label': label, 'target': 'None'}
    else:
        #create a json obj using the label and targets
        obj = {"label": label, "target": targets}
    return obj

# function that given a array of string merge them in a single string
def merge_tokens(tokens):
    tokens = tokens.replace('[', '')
    tokens = tokens.replace(']', '')
    tokens = tokens.replace("'", '')
    tokens = tokens.split(',')
    tokens = ' '.join(tokens)
    return tokens

# function to eliminate specific rows if a value of a specific column is equal to a specific value
def eliminate_rows(df, column_name, value):
    df = df[df[column_name] != value]
    return df

# function that take in input a function and a dataframe
# and return a new dataframe with the function applied to specific column
def apply_function_to_column(df, column_name, function):
    df[column_name] = df[column_name].apply(function)
    return df
    


# function to read a csv file and modify it's structure
def restructure_dataset(csv_path = 'hatexplain.csv'):
    df = pd.read_csv(csv_path)
    #delete 'rationales' column
    df = delete_column(df, 'rationales')
    # transform annotators column in a json object
    df = apply_function_to_column(df,'annotators',(lambda x : string_to_objJson(x)))
    #create two new columns label and target
    df['label'] = df['annotators'].apply(lambda x : x['label'])
    df['target'] = df['annotators'].apply(lambda x : x['target'])
    #delete the old column
    df.drop('annotators', axis = 1, inplace = True)
    #delete post_id column
    df = delete_column(df, 'post_id')
    #trasmorm in a single string the tokens column
    df['corpus'] = df['post_tokens'].apply(lambda x : merge_tokens(x))
    df = df[['label', 'target', 'corpus']]
    #eliminate rows with label = 'None'
    df = eliminate_rows(df, 'label', 'None')
    #eliminate rows with target = 'None'
    df = eliminate_rows(df, 'target', 'None')
    # Preprocessing corpus
    df = apply_function_to_column(df, 'corpus', (lambda x : text_preprocessing.preprocess_string(x)))
    # Save the new dataset
    df.to_csv('hatexplainV2.csv', index = False)



