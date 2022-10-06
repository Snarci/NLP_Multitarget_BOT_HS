import  dataset_utility


main_verbose = True
need_exctract_labels_and_targets = True
need_exctract_corpus = True
need_save_dataset = True

df = dataset_utility.get_dataset(name='visualization.csv')
if need_exctract_corpus:

    df_corpus = dataset_utility.extract_corpus(df, verbose=main_verbose)

    df_corpus = dataset_utility.merge_copus_tokens(df_corpus, verbose=main_verbose)

    df_corpus_preprocessed = dataset_utility.preprocess_corpus(df_corpus, verbose=main_verbose)

if need_exctract_labels_and_targets:
    #print(df.head(15))
    [labels,targets,df_corpus_preprocessed] = dataset_utility.get_labels_and_targets(df,df_corpus_preprocessed, verbose=main_verbose)

    [df_label,df_target] = dataset_utility.mode_labels_and_targets(labels,targets, verbose=main_verbose)

    [label_values, target_values] = dataset_utility.get_unique_values_labels_and_targets(df_label,df_target, verbose=main_verbose)

    label_freqeuncy = dataset_utility.get_label_frequency(df_label, verbose=True)



if need_save_dataset:
    
    #print shapes of the dataframes
    print("Shape of df_label:")
    print(df_label.shape)
    print("Shape of df_target:")
    print(df_target.shape)
    print("Shape of df_corpus:")
    print(df_corpus.shape)
    print("Shape of df_corpus_preprocessed:")
    print(df_corpus_preprocessed.shape)
    df_merged = dataset_utility.merge_labels_targets_corpus(df_label,df_target,df_corpus_preprocessed, verbose=True)
    
    dataset_utility.save_dataframe(df_merged, 'preprocessed.csv', verbose=True)
