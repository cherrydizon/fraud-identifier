#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
#from tester import test_classifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages',
                 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi', 'from_poi_to_this_person_fraction',
                 'from_this_person_to_poi_fraction'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "size of data set: "
print len(data_dict)

### Task 2: Remove outliers
data_dict.pop("TOTAL")
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#Create new features
for key in data_dict:
    if (data_dict[key]['to_messages'] != 0 and
        data_dict[key]['to_messages'] != 'NaN' and
        data_dict[key]['from_poi_to_this_person'] != 'NaN'): 
            data_dict[key]['from_poi_to_this_person_fraction'] = data_dict[key]['from_poi_to_this_person']/float(data_dict[key]['to_messages'])
    else:
        data_dict[key]['from_poi_to_this_person_fraction'] = 'NaN'
    
    if (data_dict[key]['from_messages'] != 0 and
        data_dict[key]['from_messages'] != 'NaN' and
        data_dict[key]['from_this_person_to_poi'] != 'NaN'): 
            data_dict[key]['from_this_person_to_poi_fraction'] = data_dict[key]['from_this_person_to_poi']/float(data_dict[key]['from_messages'])
    else:
        data_dict[key]['from_this_person_to_poi_fraction'] = 'NaN'

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


from sklearn.feature_selection import SelectKBest, f_classif


from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html   
scaler = MinMaxScaler()
###---1st Algorithm---###

kbest = SelectKBest(f_classif)
pipeline = Pipeline([('scaler', scaler), ('kbest', kbest), ('abc', AdaBoostClassifier(base_estimator = DecisionTreeClassifier(random_state = 22), random_state = 22))])
param_grid = {'kbest__k' : [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
              'abc__n_estimators' : [5, 50, 100],
              'abc__learning_rate' : [0.5, 1.0, 2.0],
              'abc__algorithm' : ['SAMME', 'SAMME.R'],
              }
          
grid_search = GridSearchCV(pipeline, param_grid = param_grid, scoring = 'f1', cv = StratifiedKFold(labels, random_state = 22))
grid_search.fit(features, labels)
print "The best parameters for the grid:"
print(grid_search.best_params_)
print grid_search.grid_scores_

### The code below is for checking the performance of the algorithm. 
### It can be commented out since tester.py is assumed to be run independently, not in this file.

#get the best estimator
clf = grid_search.best_estimator_
dump_classifier_and_data(clf, my_dataset, features_list) ###(this part is commented out since 3rd algorithm is chosen as the final model)
#run the test_classifier locally
print "performance after tuning:"
test_classifier(clf, my_dataset, features_list, folds = 1000) ###(this part is commented out since tester.py will be run independently when checking)

#performance before tuning
pipeline = Pipeline([('scaler', scaler), ('kbest', SelectKBest(f_classif, k=8)), ('abc', AdaBoostClassifier(base_estimator = DecisionTreeClassifier(random_state = 22), random_state = 22))])
param_grid = {}
print "performance before tuning:"          
grid_search = GridSearchCV(pipeline, param_grid = param_grid, scoring = 'f1', cv = StratifiedKFold(labels, random_state = 22))
grid_search.fit(features, labels)
clf = grid_search.best_estimator_
dump_classifier_and_data(clf, my_dataset, features_list) ###(this part is commented out since 3rd algorithm is chosen as the final model)
#run the test_classifier locally
test_classifier(clf, my_dataset, features_list, folds = 1000) ###(this part is commented out since tester.py will be run independently when checking)


###---2nd Algorithm--###
from sklearn.svm import SVC
kbest = SelectKBest(f_classif)
pipeline = Pipeline([('scaler', scaler), ('kbest', kbest), ('clf', SVC(random_state = 22))])
param_grid = {'kbest__k' : [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
              'clf__kernel' : ['linear', 'poly', 'rbf'],
              'clf__degree' : [1, 3, 5],
              'clf__C' : [0.5, 1.0, 2.0],
              'clf__decision_function_shape' : ['ovo', 'ovr', 'None']}
grid_search = GridSearchCV(pipeline, param_grid = param_grid, scoring = 'f1', cv = StratifiedKFold(labels, random_state = 22))
grid_search.fit(features, labels)
print "The best parameters for the grid:"
print(grid_search.best_params_)
print grid_search.grid_scores_

### The code below is for checking the performance of the algorithm. 
### It can be commented out since tester.py is assumed to be run independently, not in this file.

#get the best estimator
clf = grid_search.best_estimator_
dump_classifier_and_data(clf, my_dataset, features_list) ###(this part is commented out since 2nd algorithm is chosen as the final model)
#run the test_classifier locally
print "performance after tuning:"
test_classifier(clf, my_dataset, features_list, folds = 1000) ###(this part is commented out since tester.py will be run independently when checking)

#performance before tuning
pipeline = Pipeline([('scaler', scaler), ('kbest', SelectKBest(f_classif,k=18)), ('clf', SVC(random_state = 22))])
param_grid = {}
print "performance before tuning:"          
grid_search = GridSearchCV(pipeline, param_grid = param_grid, scoring = 'f1', cv = StratifiedKFold(labels, random_state = 22))
grid_search.fit(features, labels)
clf = grid_search.best_estimator_
dump_classifier_and_data(clf, my_dataset, features_list) ###(this part is commented out since 3rd algorithm is chosen as the final model)
#run the test_classifier locally
test_classifier(clf, my_dataset, features_list, folds = 1000) ###(this part is commented out since tester.py will be run independently when checking)


###---3rd Algorithm (FINAL ANALYSIS)---###
kbest = SelectKBest(f_classif)
pipeline = Pipeline([('kbest', kbest), ('clf', GaussianNB())])
param_grid = {'kbest__k': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
             }
          
grid_search = GridSearchCV(pipeline, param_grid = param_grid, scoring = 'f1', cv = StratifiedKFold(labels, random_state = 22))
grid_search.fit(features, labels)
print "The best parameters for the grid:"
print(grid_search.best_params_)
features_selected =  grid_search.best_estimator_.named_steps['kbest'].get_support()
features_selected_list = [x for x, y in zip(features_list[1:], features_selected) if y]
best_k = grid_search.best_params_['kbest__k']
SKB_k = SelectKBest(f_classif, k=best_k)
SKB_k.fit_transform(features_train, labels_train)
feature_scores = SKB_k.scores_
features_scores_selected = [feature_scores[i] for i in SKB_k.get_support(indices=True)]

for feature_name in features_selected_list:
    print feature_name

print features_scores_selected
#get the best estimator
clf = grid_search.best_estimator_
dump_classifier_and_data(clf, my_dataset, features_list)
test_classifier(clf, my_dataset, features_list, folds = 1000) ###(this part can be commented out since tester.py will be run independently when checking)

