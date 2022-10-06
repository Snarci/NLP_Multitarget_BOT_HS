#read dataset ghc_train.tsv as a dataframe and show first 10 rows

from email import header
import pandas as pd
import dataset_preprocessing

def get_cleaned_dataset(need_mean=0,testing_reduction=0,need_save=0):
    if need_mean:
        df = pd.read_csv('GabHateCorpus_annotations.tsv', sep='\t')
        #print(df.head(10))

        #replace nans with integer zeroes in the dataset
        df = df.fillna(int(0))

        #remove the column "Annotator" from the dataset
        df.drop(['Annotator'], axis=1, inplace=True)

        #group by the column ID and do majority voting on the other columns
        df = df.groupby('ID').aggregate(lambda x: x.value_counts().index[0])

        #save the dataset to a new file csv format with the name "mean_data.csv"
        #df.to_csv('mean_data.csv', index=False)
    else:
        df = pd.read_csv('mean_data.csv')
        if testing_reduction:
            df = df.head(10)
        df = dataset_preprocessing.preprocess_text(df, 'Text')
        if need_save:
            df.to_csv('preprocessed_mean_data.csv', index=False)
    return df
