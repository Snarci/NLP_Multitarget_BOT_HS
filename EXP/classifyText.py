from unicodedata import name
import dataset_preprocessing
import pickle


def classify_text(text, label_classifier,list_target_classifiers, treshold_bot = 0.4,verbose=True ):
    text = dataset_preprocessing.preprocess_text_test(text)
    tfidf = pickle.load(open("tfidf.pickle", "rb"))
    text = tfidf.transform([text])

    label = label_classifier.predict(text)
    label_probability = label_classifier.predict_proba(text)
    #normal_probability = label_probability[0][1]
    hate_probability = label_probability[0][0]
    offensive_probability = label_probability[0][2]

    if((label != "normal") or (hate_probability>treshold_bot and offensive_probability>treshold_bot)):
        list_label_probabilities = []
        for classifier in list_target_classifiers:
            label_probability = classifier.predict_proba(text)
            label_probability = label_probability[0][1]
            list_label_probabilities.append(label_probability)
        max_index = list_label_probabilities.index(max(list_label_probabilities))
        return True, "the message is: "+str(label[0]) + " towards the "+str(list_target_classifiers[max_index]) + " target"
    else: 
        return False , "This is a normal message."