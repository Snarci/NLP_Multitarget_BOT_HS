from cProfile import label
import pandas as pd
import json
from collections import Counter
import text_preprocessing

# arrays of possible values for the target_group
Race = ['African', 'Asian', 'Caucasian', 'Hispanic', 'Arab','Indian']  #['African','Arab','Asian','Caucasian','Hispanic']
Religion = ['Buddhism', 'Christian', 'Hindu', 'Islam', 'Jewish']
Gender = ['Men', 'Women']
Sexual_Orientation = ['Homosexual'] #['Heterosexual','Gay']
Miscellaneous =  ['None','Other','Refugee', 'Indigenous','Economic','Disability']
#Economic = ['Economic']
#Disability = ['Disability']

# array of possible values of the target_group column
target_groups = ['Race', 'Religion', 'Gender', 'Sexual Orientation', 'Miscellaneous']

# array of possible values of the target column
target_values = ['African', 'Asian', 'Caucasian', 'Jewish', 'Hispanic', 'Arab', 'Refugee','Indian', 'Indigenous','Buddhism', 'Christian', 'Hindu', 'Islam',
                 'Jewish','Men', 'Women','Homosexual','None','Other','Economic','Disability']

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
        label = 'Uncertainty'

    target = [item for sublist in target for item in sublist]
    countTarget = Counter(target)
    targets = []
    for key, value in countTarget.items():
        if value > 1:
            targets.append(key)

    if len(targets) == 0:
        obj = {'label': label, 'target': 'Uncertainty'}
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

# function to assign a target group based on the target column chek if the target is in a specific group
def assign_target_group(targets):
    targetGroup = []
    for target in targets:
        if target in Race:
            targetGroup.append('Race')
        if target in Religion:
            targetGroup.append('Religion')
        if target in Gender:
            targetGroup.append('Gender')
        if target in Sexual_Orientation:
            targetGroup.append('Sexual Orientation')
        if target in Miscellaneous:
            targetGroup.append('Miscellaneous')
        #if target in Economic:
        #    targetGroup.append('Economic')
        #if target in Disability:
        #    targetGroup.append('Disability')
    if len(targetGroup) == 0:
        print(targets)
    return targetGroup


# function to read a df and return a df with a new comlumn that contains the target group based on the target column
def create_target_group(df):
    df['target_group'] = df['target'].apply(lambda x : assign_target_group(x))
    return df

#function to create a number of columns equal to the number of possible values and set to 1 the column that contains the value
def create_binary_columns(df, name_column, possible_values_array):
    for value in possible_values_array:
        df[value] = df[name_column].apply(lambda x: 1 if value in x else 0)
    return df

# function to count how many records in a specific columns have a specific value and return an json object with the value and the percentage
def count_value(df, column_name, value):
    # count the number of records that have the value in the column
    print(column_name)
    count = len(df[df[column_name] == value])
    print(count)
    percentage = count / len(df)
    print(percentage)
    return {'label': column_name, 'value': value, 'percentage': percentage, 'count': count, 'total': len(df)}

#function to count value for a set of columns
def calculate_distribution(df, column_names, value):
    res = []
    for column_name in column_names:
        res.append(count_value(df, column_name, value))
    return res

#function to save a array of json object in a json file
def save_json(array, path):
    with open(path, 'w') as outfile:
        json.dump(array, outfile)


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
    #eliminate rows with label = 'Uncertainty'
    df = eliminate_rows(df, 'label', 'Uncertainty')
    #eliminate rows with target = 'Uncertainty'
    df = eliminate_rows(df, 'target', 'Uncertainty')
    # Preprocessing corpus
    df = apply_function_to_column(df, 'corpus', (lambda x : text_preprocessing.preprocess_string(x)))
    #create a new column target_group
    df = create_target_group(df)
    df = create_binary_columns(df, 'target_group', target_groups)
    #df = create_binary_columns(df, 'target', target_values)
    #Print the distribution of each target
    save_json(calculate_distribution(df, target_groups, 1), 'distribution.json')
    # Save the new dataset
    df.to_csv('hatexplainV2.csv', index = False)
