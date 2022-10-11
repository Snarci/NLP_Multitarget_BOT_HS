from cProfile import label
import pandas as pd
import json
from collections import Counter

# function to read a json file and convert it to a csv file
def json_to_csv(json_path = 'hatexplain.json', csv_path = 'hatexplain.csv'):
    df = pd.read_json(json_path,orient = 'index')
    df.to_csv(csv_path, index = False)

# function to read a csv file and delete a column of it by its name
def delete_column(csv_path = 'hatexplain.csv', column_name = 'Unnamed: 0'):
    df = pd.read_csv(csv_path)
    df.drop(column_name, axis = 1, inplace = True)
    df.to_csv(csv_path, index = False)

# function to read a string and make an object from it
def string_to_columns(string):
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
        targets.append('None')

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

# function to read a csv file and modify its column values for each row
def modify_columns(csv_path = 'hatexplain.csv', column_name = 'annotators'):
    df = pd.read_csv(csv_path)
    df[column_name] = df[column_name].apply(lambda x : string_to_columns(x))
    #create two new columns label and target
    df['label'] = df[column_name].apply(lambda x : x['label'])
    df['target'] = df[column_name].apply(lambda x : x['target'])
    #delete the old column
    df.drop(column_name, axis = 1, inplace = True)
    #delete post_id column
    df.drop('post_id', axis = 1, inplace = True)
    #trasmorm in a single string the tokens column
    df['corpus'] = df['post_tokens'].apply(lambda x : merge_tokens(x))
    df = df[['label', 'target', 'corpus']]
    df.to_csv('hatexplainV2.csv', index = False)



