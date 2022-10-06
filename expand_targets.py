import dataset_utility


#read dataset preprocessed.csv as a dataframe
possible_targets=['None', 'African', 'Asian', 'Women', 
                'Caucasian', 'Jewish', 'Homosexual',
                'Islam', 'Hispanic', 'Arab', 'Refugee',
                'Economic', 'Other', 'Disability', 'Men',
                'Indian', 'Christian', 'Hindu', 'Indigenous',
                'Buddhism']

df = dataset_utility.get_dataset(name='preprocessed.csv')

#get the target column from df

expanded = dataset_utility.expand_targets(df, possible_targets)

#save the expanded dataframe to expanded.csv
dataset_utility.save_dataframe(expanded, 'expanded.csv', verbose=True)

