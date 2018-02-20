# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:05:42 2018

@author: V
"""


# 1. Prepare Problem ##########################################################
    # a) Load libraries _____ _____ _____ _____ ____ _____ _____ _____ _____ __
import pandas as pd
import sys as os
import time
import winsound
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pickle import dump
from pickle import load
#from pandas.tools.plotting import scatter_matrix

def outputXML(user):
    f = open(user[1] + ".xml", 'w')
    f.write("<user\n\tid=\"" + user[1] + "\"\n")
    f.write("age_group=\"" + user[2] + "\"\n")
    f.write("gender=\"" + user[3] + "\"\n")
    f.write("extrovert=\"" + user[4] + "\"\n")
    f.write("neurotic=\"" + user[5] + "\"\n")
    f.write("agreeable=\"" + user[6] + "\"\n")
    f.write("conscientious=\"" + user[7] + "\"\n")
    f.write("open=\"" + user[8] + "\"\n")
    f.write("/>")
    f.close()

def convert_age_to_class(n):

    if n <= 24:
        return "A"
    elif n <= 34:
        return "B"
    elif n <= 49:
        return "C"
    else:
        return "D"



def main():

    # b) Load dataset: Conver .csv into DataFrames _____ _____ _____ _____ ____

    argv = os.argv
    opts = {}
    while(argv):
        if argv[0][0] == '-':
            opts[argv[0]] = argv[1]
        argv = argv[1:]

    relation_path = opts['-r']
    profile_path = opts['-p']
    relation_df = pd.read_csv(relation_path)
    profile_df = pd.read_csv(profile_path)


    # 2. Summarize Data #######################################################

        # a) Descriptive statistics _____ _____ _____ _____ _____ _____ _____ _

    #print(profile_df.describe())
    #print(relation_df.describe())


        # b) Data visualizations _____ _____ _____ _____ _____ _____ _____ ____

    #pd.set_option('display.width', 100)
    #pd.set_option('precision', 3)
    #correlations = profile_df.corr(method='pearson')
    #print(correlations)
    #profile_df.hist()
    #pyplot.show()

    # 3. Prepare Data #########################################################

        # a) Data Cleaning
        # b) Feature Selection
        # c) Data Transforms

    #####

    #userid_col = relation_df[['userid']]
    #row_counter = 1
    #num_users = 1
    #userid_dict = {}
    #
    ##put all userids' in a dictionary
    #for index, row in userid_col.iterrows():
    #    l = row.tolist()
    #    userid = l[0].strip()
    #    if (userid not in userid_dict):
    #        userid_dict[userid] = ""
    #        num_users += 1
    #
    #    row_counter += 1
    #    if (row_counter < -2000): #15change
    #        break
    #
    #print("Here now")
    #
    ##combine all likeids' associated with a userid
    ##make this the value of the userid in the dictionary
    #for index, row in relation_df.iterrows():#45change
    #
    #    row_list = row.tolist()
    #    userid = str(row_list[1])
    #    user_vals = userid_dict[userid]
    #    userid_dict[userid] = user_vals + " " + str(row_list[2])
    #
    #t_df = pd.DataFrame.from_dict(userid_dict, orient='index')
    #t_df = t_df.reset_index() ## remember to reassign when calling a function
    #t_df.columns = ["userid", "likes"]
    #
    #merge_df = pd.merge(t_df, profile_df, on="userid") #55change
    #merge_df.to_csv("merged_test.csv", sep=',')

    #####

    merge_df = pd.read_csv('merged.csv', sep=',')


    # 4. Evaluate Algorithms###################################################

    # a) Split-out validation dataset _____ _____ _____ _____ _____ _____ _____
    X = merge_df['likes']
    y_gender = merge_df['gender']
    y_age = merge_df['age']
    y_age = y_age.apply(convert_age_to_class)
#    y_age.to_csv("age_classified.csv", sep=',')
    valida_size = 0.20
    seed = 7
#
#    #split-out gender
    X_train_gender, X_validation_gender, y_train_gender, y_validation_gender = train_test_split(X, y_gender, test_size=valida_size, random_state=seed)
#
#    #split-out age
    X_train, X_validation, y_train_age, y_validation_age = train_test_split(X, y_age, test_size=valida_size, random_state=seed)

    # b) Test options and evaluation metric _____ _____ _____ _____ _____ _____
#    count_vect1 = CountVectorizer()
#    X_train = count_vect1.fit_transform(X_train)
#    count_vect2 = CountVectorizer(vocabulary=count_vect1.vocabulary_)
#    X_validation = count_vect2.fit_transform(X_validation)
#
#    count_vect3 = CountVectorizer()
#    X = count_vect3.fit_transform(X)
#    count_vect4 = CountVectorizer(vocabulary=count_vect3.vocabulary_)


    # c) Spot Check Algorithms _____ _____ _____ _____ _____ _____ _____ _____

# start
#    models_gender = []
#    models_gender.append(('multiNB', MultinomialNB()))
#    models_gender.append(('bernoulliNB', BernoulliNB()))
#    models_gender.append(('kNN', KNeighborsClassifier()))
#    models_gender.append(('LogReg', LogisticRegression()))
#    models_gender.append(('SGD', linear_model.SGDClassifier(max_iter=10, learning_rate='optimal', random_state=seed)))
#
#    print('Comparing Algorithms for gender:')
#    results_gender = []
#    names_gender  = []
#    for name, model in models_gender:
#        kfold = KFold(n_splits=10, random_state = seed)
#        cv_results = cross_val_score(model, X_train, y_train_gender, cv=kfold, scoring='accuracy')
#        results_gender.append(cv_results)
#        names_gender.append(name)
#        print("%s -> accuracy:%f w/std_dev:%f" % (name, cv_results.mean(),cv_results.std()))
#
#
#    # d) Compare Algorithms _____ _____ _____ _____ _____ _____ _____ _____ ___
#    fig_gender = pyplot.figure()
#    fig_gender.suptitle('Algorithm Comparison')
#    ax = fig_gender.add_subplot(111)
#    pyplot.boxplot(results_gender)
#    ax.set_xticklabels(names_gender)
#    pyplot.show()
#
#    #repeat c & d for age
#    models_age= []
#    models_age.append(('multiNB', MultinomialNB()))
#    models_age.append(('bernoulliNB', BernoulliNB()))
#    models_age.append(('kNN', KNeighborsClassifier()))
#    models_age.append(('LogReg', LogisticRegression()))
#    models_age.append(('SGD', linear_model.SGDClassifier(max_iter=10, learning_rate='optimal', random_state=seed)))
#
#    print()
#    print('Comparing Algorithms for age:')
#    results_age = []
#    names_age = []
#    for name, model in models_age:
#        kfold = KFold(n_splits=10, random_state = seed)
#        cv_results = cross_val_score(model, X_train, y_train_age, cv=kfold, scoring='accuracy')
#        results_age.append(cv_results)
#        names_age.append(name)
#        print("%s -> accuracy:%f w/std_dev:%f" % (name, cv_results.mean(),cv_results.std()))
#
#    fig_age = pyplot.figure()
#    fig_age.suptitle('Algorithm Comparison')
#    ax = fig_age.add_subplot(111)
#    pyplot.boxplot(results_age)
#    ax.set_xticklabels(names_age)
#    pyplot.show()
# end

    # 5. Improve Accuracy #####################################################
    # a) Algorithm Tuning

    # b) Ensembles gender _____ _____ _____ _____ _____ _____ _____ _____ _____
    lr_gender = LogisticRegression(random_state=seed)
    sgd_gender = linear_model.SGDClassifier(max_iter=10, learning_rate='optimal', random_state=seed)
    mNB_gender = MultinomialNB()
    ensemble_gender = VotingClassifier(estimators=[('lr', lr_gender),('sgd',sgd_gender),('mNB',mNB_gender)],voting='hard')

    # b) Ensembles age
    lr_age = LogisticRegression(random_state=seed)
    sgd_age = linear_model.SGDClassifier(max_iter=10, learning_rate='optimal', random_state=seed)
    bNB_age = BernoulliNB()
    ensemble_age = VotingClassifier(estimators=[('lr', lr_age),('sgd',sgd_age),('bNB',bNB_age)],voting='hard')

    # 6. Finalize Model #######################################################
    # a) Predictions on validation dataset_____ _____ _____ _____ _____ _____ _

##start
#    ensemble_gender = ensemble_gender.fit(X_train, y_train_gender)
#    gender_ensemble_results = ensemble_gender.predict(X_validation)
#    print("gender ensemble results:")
#    print(accuracy_score(y_validation_gender, gender_ensemble_results))
#    print(confusion_matrix(y_validation_gender, gender_ensemble_results))
#    print(classification_report(y_validation_gender, gender_ensemble_results))
#
#    print()
#    print("age ensemble results:")
#    ensemble_age = ensemble_age.fit(X_train, y_train_age)
#    age_ensemble_results = ensemble_age.predict(X_validation)
#    print(accuracy_score(y_validation_age, age_ensemble_results))
#    print(confusion_matrix(y_validation_age, age_ensemble_results))
#    print(classification_report(y_validation_age, age_ensemble_results))


    # b) Create standalone model on entire training dataset _____ _____ _____ _

#    count_vect1 = CountVectorizer()
#    X_train = count_vect1.fit_transform(X_train)
#    count_vect2 = CountVectorizer(vocabulary=count_vect1.vocabulary_)
#    X_validation = count_vect2.fit_transform(X_validation)
##end

##start
    count_vect3 = CountVectorizer()
    X = count_vect3.fit_transform(X)
    count_vect4 = CountVectorizer(vocabulary=count_vect3.vocabulary_)
    X_validation = count_vect4.fit_transform(X_validation)

    ensemble_gender = ensemble_gender.fit(X, y_gender)
    print("Here")
    gender_ensemble_results = ensemble_gender.predict(X_validation)
    print("gender ensemble results:")
    print(accuracy_score(y_validation_gender, gender_ensemble_results))
    print(confusion_matrix(y_validation_gender, gender_ensemble_results))
    print(classification_report(y_validation_gender, gender_ensemble_results))

    print()
    print("age ensemble results:")
    ensemble_age = ensemble_age.fit(X, y_age)
    age_ensemble_results = ensemble_age.predict(X_validation)
    print(accuracy_score(y_validation_age, age_ensemble_results))
    print(confusion_matrix(y_validation_age, age_ensemble_results))
    print(classification_report(y_validation_age, age_ensemble_results))
##end

    # c) Serialize the model for later use _____ _____ _____ _____ _____ _____
    filename_gender = 'gender_ensemble_clf.sav'
    dump(ensemble_gender, open(filename_gender, 'wb'))
    filename_age = 'age_ensemble_clf.sav'
    dump(ensemble_age, open(filename_age, 'wb'))
    print()
    print()
    print("Done")

main()