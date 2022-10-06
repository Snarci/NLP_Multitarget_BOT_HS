import train_classifier_utility as tcu

#tcu.loop_train_and_save_classifier_for_all_classes()

etnicity_targets =   [ 'African', 'Asian', 'Caucasian', 'Jewish', 'Hispanic', 'Arab', 'Refugee','Indian', 'Indigenous']     

sex_targets = ['Women', 'Men']

religion_targets = [ 'Jewish','Islam','Christian', 'Hindu', 'Buddhism']

homosexuality_targets = ['Homosexual']

economic_targets = ['Economic']

other_targets = ['Other']

disability_targets = ['Disability']

aa =["Etnicity","Sex","Religion","Homosexual","Economic","Other","Disability"]
tcu.train_and_save_classifier_agglomerate("expanded.csv", etnicity_targets, "Corpus","Etnicity", need_balancing=True)
tcu.train_and_save_classifier_agglomerate("expanded.csv", sex_targets, "Corpus","Sex", need_balancing=True)
tcu.train_and_save_classifier_agglomerate("expanded.csv", religion_targets, "Corpus","Religion", need_balancing=True)
tcu.train_and_save_classifier_agglomerate("expanded.csv", homosexuality_targets, "Corpus","Homosexual", need_balancing=True)
tcu.train_and_save_classifier_agglomerate("expanded.csv", economic_targets, "Corpus","Economic", need_balancing=True)
tcu.train_and_save_classifier_agglomerate("expanded.csv", other_targets, "Corpus","Other", need_balancing=True)
tcu.train_and_save_classifier_agglomerate("expanded.csv", disability_targets, "Corpus","Disability", need_balancing=True)

