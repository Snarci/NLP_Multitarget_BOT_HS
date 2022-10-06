import dataset_preprocessing
import pandas as pd



df = pd.read_csv('preprocessed_mean_data.csv')

labels=['Hate','HD','CV','VO','REL','RAE','SXO','GEN','IDL','NAT','POL','MPH','EX','IM']
print("The distribution of the classes in the dataset is:")
for label in labels:
    print(label+": "+str(df[label].value_counts()))

print("\n")


