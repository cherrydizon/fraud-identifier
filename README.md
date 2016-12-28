# fraud-identifier
##Project Overview
The goal of this project is to build a person of interest identifier based on financial and email data made public as a result of the Enron company scandal.The Enron scandal, revealed in October 2001, led to the bankruptcy of the Enron Corporation, an American energy company due to widespread corporate fraud.  A significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. Machine learning will be used to select the factors that affect a person being a point of interest, and to be able to identify fraud (in this case, persons of interest) in an automated way.

There are 146 data points (i.e. people) in the dataset.  There are 18 POIs in the data, which is remarkably fewer than the non-POIs. There are 21 features available (including the POI label). There were some features that have many missing values like ‘loan advances’, ‘director_fees‘, etc. When plotting the salary and bonus of the data points, there was one outlier that popped out to me immediately. By inspecting the enron61702insiderpay.pdf, I discovered that the dictionary key of this data point is “TOTAL”. I removed this outlier because it is a spreadsheet quirk. There were also some outliers with remarkably higher bonus and salary. I didn’t remove them because they are valid data points (data that corresponds to real persons) and they can be used as data in building a POI identifier.

##Selecting Features
I used SelectKBest function to select my features. I used GridSearchCV to determine what value of k would give the best performance to my model. It turns out that 13 is the best value for parameter k. The table below shows the features used in my model and the corresponding feature scores.

Feature Name |	Feature Scores
-------------|----------------
salary | 	15.81 
total_payments |	8.96 
loan_advances |	7.04 
bonus |	30.65 
deferred_income |	8.49 
total_stock_value |	10.81 
expenses |	4.31 
exercised_stock_options |	9.96 
long_term_incentive	| 7.53 
 restricted_stock |	8.05 
from_poi_to_this_person |	4.93 
shared_receipt_with_poi |	10.67 
from_this_person_to_poi_fraction |	13.79

I did not use feature scaling in my final model since I used GaussianNB as my classifier. Naive Bayes already do feature scaling by design so feature scaling is not needed. I created 'from_poi_to_this_person_fraction' and 'from_this_person_to_poi_fraction' as new features. This feature seemed relevant in order to know how frequent is the interaction of a certain person to a POI. A fraction is needed since it is important to know how frequent the email interaction is compared to the total number of emails of a certain person. I ended up using ‘from_this_person_to_poi_fraction’ in my final model since it is included in the best 13 features.

##Algorithms Used
I used Gaussian Naive Bayes as my final algorithm. I also tried using Adaboost and SVC. The table below shows the performance of the three algorithms (these values were taken by running tester.py after feature selection and parameter tuning).

 Algorithm |	Accuracy | Precision | Recall | f1 
-----------|-----------|-----------|--------|----
Gaussian Naive Bayes |	0.85 |	0.43 |	0.33	| 0.38
Adaboost |	0.81	| 0.29	| 0.27 |	0.28
SVC |	0.86	| 0.18	| 0.02 |	0.04

##Parameter Tuning
Parameter tuning is setting the parameters of an algorithm to optimal values in order to get the best performance of the algorithm. If parameter tuning is not done, it may result to bad performance like overfitting etc. I used GridSearchCV to search for the best parameter values and the best estimator. I did parameter tuning on Adaboost and SVC. The parameters I tuned and the performance before and after tuning are shown in the table. For Adaboost, the performance before and after tuning were the same while in SVC, the performance improved after tuning (these values were taken through StratifiedKFold cross-validation).


Algorithm	| Tuned Parameters |	f1 before tuning |	f1 after tuning
----------|------------------|-------------------|-----------------
Adaboost |	n_estimators, learning_rate, algorithm |	0.491 |	0.491
SVC	| kernel, degree, C, decision_function_shape	| 0.095	| 0.000

##Validation
Validation is assessing the performance of the algorithm. A classic mistake is testing and training the model in the same dataset. It will give high performance metric values and it will give you the illusion that your model is doing well when in fact it is not. I validated my algorithms using cross validation (StratifiedKFold) in my analysis. Cross validation involves partitioning data into subsets, performing the analysis on one subset, and validating the analysis on the other subset. Multiple rounds of cross-validation are performed using different partitions, and the validation results are averaged over the rounds.

##Evaluation
The performance of my model is shown below.

Evaluation |	Metric Value
-----------|----------------
precision|	0.43
recall|	0.33

In terms of this project, precision is the probability that given a result of POI, it is indeed a POI. Meanwhile, recall is the probability that a POI is correctly identified. Good precision and bad recall means that whenever a POI gets flagged, I know with a lot of confidence that it is very likely to be a real POI and not a false alarm. However, I sometimes miss real POIs since I am effectively reluctant to pull the trigger on edge cases. Bad precision and good recall means that nearly every time a POI shows up in my test set, I am able to identify him or her. On the other hand, I sometimes get some false positives where non-POIs get flagged.

###Files
- `poi_id.py`: Person of interest identifier
- `final_project_dataset.pkl`: The dataset for the project

###References
- [Comparing supervised learning algorithms](http://www.dataschool.io/comparing-supervised-learning-algorithms/)
- [Feature Scaling](http://dshincd.github.io/blog/feautre-scaling/)
- [Cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))
- Udacity Forums







