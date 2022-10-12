
from cmath import nan
from operator import concat
import pandas as pd
import numpy as np
import json
import dataset_preprocessing

def get_dataset(name='visualization.csv'):
    df = pd.read_csv(name, encoding='utf-8')
    return df

def get_labels_and_targets(df,df_corpus_preprocessed, verbose=True):
    #get the dataframe column of annotators

    df_annotators = df['annotators']


    #split the column using } as separator into multiple rows

    df_annotators = df_annotators.str.split('}', expand=True)

    #add '}' to the end of each row

    df_annotators = df_annotators+ '}'

    #get first row value and split it using "'label'" as separator into multiple rows
    #iterate for each row and get the first row value and split it using "'label'" as separator into multiple rows

    #estrazione target e labels
    keeper_label = []
    keeper_target = []
    keeper_corpus = []
    number_of_deled = 0
    #create empty dataframe to store short_df
    
    for i in range(0,len(df_annotators)):
        flag_eliminazione = 0
        current_row=df_annotators.iloc[i].values
        keeper_inner_label = ["","",""]
        keeper_inner_target = ["","",""]
        for j in range(0,3):
            current_sub_row=current_row[j]
            current_sub_row=current_sub_row[1:]
            current_sub_row = current_sub_row.replace("'", '"')
            current_sub_row = json.loads(current_sub_row)
            keeper_inner_label[j]=current_sub_row["label"]
            keeper_inner_target[j]=current_sub_row["target"]
            #append the array to the keeper array
        #check if the lements in keeper_inner_label are equal to each other
        if keeper_inner_label[0]!=keeper_inner_label[1] and keeper_inner_label[0]!=keeper_inner_label[2] and keeper_inner_label[1]!=keeper_inner_label[2]:
            #print("Tutti diversi per"+ 'keeper_inner_label'+ "riga"+str(i)+"eliminata")
            flag_eliminazione = 1
        elif keeper_inner_target[0]!=keeper_inner_target[1] and keeper_inner_target[0]!=keeper_inner_target[2] and keeper_inner_target[1]!=keeper_inner_target[2]:
            flag_eliminazione = 1
        else:
            keeper_target.append(keeper_inner_target)
            keeper_label.append(keeper_inner_label)
            #append the array to the keeper array
            keeper_corpus.append(df_corpus_preprocessed.iloc[i])
            #remove row of index i from df
        if flag_eliminazione == 1:
            flag_eliminazione = 0
            #add i to the dopping_index array

            number_of_deled = number_of_deled + 1

    if verbose:
        print("eliminate"+str(number_of_deled)+"righe")
        print("numero di righe"+str(len(keeper_target )))
        print("numero di rige di partenza"+str(len(df)))
    #cast keeper_corpus to a dataframe
    keeper_corpus = pd.DataFrame(keeper_corpus)
    return keeper_label, keeper_target, keeper_corpus

def mode_labels_and_targets(labels,targets, verbose=True):
    df_label = pd.DataFrame(labels)
    if verbose:
        print("Dataframe labels:")
        print(df_label)

    #get mode of each row for the array labels
    df_label = df_label.mode(axis=1)
    if verbose:
        print("Mode of labels:")
        print(df_label)


    df_target = pd.DataFrame(targets)
    if verbose:
        print("Dataframe targets:")
        print(df_target)

    #get mode of each row for the array targets
    df_target = df_target.mode(axis=1)
    if verbose:
        print("Mode of targets:")
        print(df_target)

    return df_label, df_target

def get_unique_values_labels_and_targets(df_label,df_target, verbose=True): 
    #get all possible target values from the dataframe
    df_target_values = df_target.values
    df_target_values = df_target_values.flatten()

    target_values=[]
    for i in range(0,len(df_target_values)):
        #add all the elements to a np array
        #check if df_target_values[i] has a len 
        #print(df_target_values[i])
        for j in range(0,len(df_target_values[i])):
            if df_target_values[i][j] not in target_values:
                target_values.append(df_target_values[i][j])
    if verbose:
        print("Possible target values:")
        print(target_values)



    #get all possible label values from the dataframe
    df_label_values = df_label.values
    df_label_values = df_label_values.flatten()

    label_values=[]
    for i in range(0,len(df_label_values)):
        #add all the elements to a np array
        if df_label_values[i] not in label_values:
            label_values.append(df_label_values[i])
    if verbose:
        print("Possible label values:")
        print(label_values)

    return label_values, target_values

def get_label_frequency(df_label, verbose=True):
    #get the frequency of each label value
    df_label_frequency = df_label.value_counts()
    if verbose:
        print("Frequency of labels:")
        print(df_label_frequency)
    return df_label_frequency
        
def extract_corpus(df, verbose=True):
    #print(df.head(15))
    #get the last column of df
    print(df.shape)
    df_corpus = df["post_tokens"]
    #print(df_corpus.head(15))
    if verbose:
        print("Dataframe corpus:")
        print(df_corpus)
    return df_corpus

def merge_copus_tokens(df_corpus, verbose=True):
    #for each row in the dataframe, merge the tokens in the row
    for i in range(0,len(df_corpus)):
        #remove "[", "]" and "'" from the row
        tmp = df_corpus.iloc[i].replace('[', '')
        tmp = tmp.replace(']', '')
        tmp = tmp.replace("'", '')
        tmp = tmp.split(',')
        #merge the tokens in the row
        tmp = ' '.join(tmp)
        #replace the row with the merged tokens
        df_corpus.iloc[i] = tmp
    return df_corpus

def preprocess_corpus(df_corpus, verbose=True):
    df_corpus=dataset_preprocessing.preprocess_text(df_corpus)
    return df_corpus

def merge_labels_targets_corpus(df_labels,df_targets,df_corpus, verbose=True):
    #merge the labels, targets and corpus in a single dataframe using array of column names
    df_merged = pd.concat([df_labels,df_targets,df_corpus], axis=1) 
    df_merged.columns = ['Labels', 'Targets', 'Corpus']
    if verbose:
        print("Dataframe merged:")
        print(df_merged)
    return df_merged    

def save_dataframe(df, filename, verbose=True):
    #save the dataframe in a csv file
    df.to_csv(filename, index=False)
    if verbose:
        print("Dataframe saved in "+filename)

#function to eliminate rows with none values in a specific label from the dataframe
def eliminate_none_values(df, label):
    #get the index of the rows with none values in the label
    indexNames = df[ df[label] == 'None' ].index
    # Delete these row indexes from dataFrame
    df.drop(indexNames , inplace=True)
    return df
#function to eliminate rows with none values in an array a specific cell from the dataframe


def expand_targets(df,targets_names, verbose=True):
    #create a amtrix of size len(targets_names) x le
    filler = np.zeros((len(df),len(targets_names)), dtype=int)
    #cast filler to a dataframe

    filler = pd.DataFrame(filler)
    #concat the dataframe with the filler matrix
    df = pd.concat([df,filler], axis=1)
    df.columns = ['Labels', 'Targets', 'Corpus']+ targets_names
    #count the number of times each target value occurs in the dataframe and add a column for each target value
    for i in range(0,len(df["Targets"])):  #
        for j in range(0,len(targets_names)):
            #if the target name is in the wow 
            if df["Targets"].iloc[i] is not nan:
                if targets_names[j] in df["Targets"].iloc[i]:
                    #update dataframe with 1 in the target column
                    df.iloc[i,j+3] = 1
            else:
                #remove the dataframe row 
                df.drop(i, inplace=True)

    if verbose:
        print("Dataframe targets expanded:")
        print(df)

    return df
    
