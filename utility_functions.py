from dataset_preprocessing import preprocess_text_test
import train_classifier_utility as tcu
import os
import pandas as pd
import pickle

def classify_text(text, label_classifier,list_target_classifiers,ths_for_bot, verbose=True):
    response = ""
    #preprocess the text
    
    text = preprocess_text_test(text)
    print(text+ " after preprocessing")
    #apply TF-IDF on the text
    vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
    text = vectorizer.transform([text])
    #classify the text label
    label = label_classifier.predict(text)
    label_probability = label_classifier.predict_proba(text)
    normal_probability = label_probability[0][1]
    hate_probability = label_probability[0][0]
    offensive_probability = label_probability[0][2]
    list_label_probabilities = []
    list_target_classifiers_classes = ['Disability', 'Economic', 'Etnicity', 'Homosexual', 'Other', 'Religion', 'Sex']
    if verbose:
        print(label_probability)
        print(label)
        print("Normal probability is: "+str(normal_probability))
        print("Hate probability is: "+str(hate_probability))
        print("Offensive probability is: "+str(offensive_probability))
    #print(list_target_classifiers_classes)   
    #print("the labels is: "+str(label[0]))    
    if label == "normal" or (hate_probability<=ths_for_bot and offensive_probability<=ths_for_bot):
        response = "This is a normal message."
    else:
        #for each classifier in the list of classifiers list_target_classifiers
        for classifier in list_target_classifiers:
            #classify the text label probability
            label_probability = classifier.predict_proba(text)
            label_probability = label_probability[0][1]
            #print(label_probability)
            #append the label probability to the list of label probabilities
            list_label_probabilities.append(label_probability)
        
        #print(list_label_probabilities)
        #get the index of the maximum value in the list of label probabilities
        max_index = list_label_probabilities.index(max(list_label_probabilities))
        #print("Index of the maximum value: " + str(max_index))
        #get the label of the maximum value in the list of label probabilities
        #print("Label max value: " +list_target_classifiers_classes[max_index])
        response = "the message is: "+str(label[0]) + " towards the "+str(list_target_classifiers_classes[max_index]) + " target"
    return response

def load_all_classfiers_in_folder(folder_name):
    list_classifiers = []
    for filename in os.listdir(folder_name):
        if filename.endswith(".sav") and filename != "trained_classifier_Labels.sav":
            list_classifiers.append(tcu.load_classifier("./"+folder_name+"/"+filename))
    return list_classifiers

'''
#load the classifier
print("Loading classifier...")
targets_classifiers=load_all_classfiers_in_folder("trained_classifiers")
print("loaded classifiers 1")
label_classifier =tcu.load_classifier("./trained_classifiers/trained_classifier_Labels.sav")
print("loaded classifiers 2")
#classify the text
text = "sex be so good a bitch be slow stroking and cry"
classify_text(text, label_classifier, targets_classifiers, verbose=True)
'''