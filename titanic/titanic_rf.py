import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Read in data, requires pipe-delimited csv
train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

# save labels
train_labels = train_set['Survived']

# remove labels and other uneeded columns from training feature set 
train_features = train_set.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1)

# remove unused columns from test set
test_features = test_set.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

# change cabin column to bool
train_features['Cabin'] = train_features['Cabin'].notnull().astype('int')
test_features['Cabin'] = test_features['Cabin'].notnull().astype('int')

# Take the log of the Fare column
train_features['Fare'] = train_features['Fare'].replace(0.0, 0.01)
train_features['Fare'] = np.log(train_features['Fare'])

test_features['Fare'] = test_features['Fare'].fillna(test_features['Fare'].median())
test_features['Fare'] = test_features['Fare'].replace(0.0, 0.01)
test_features['Fare'] = np.log(test_features['Fare'])

# fill missing ages with average age
train_features['Age'] = train_features['Age'].fillna(train_features['Age'].median())
test_features['Age'] = test_features['Age'].fillna(test_features['Age'].median())

#replacing the missing values in the Embarked feature with S
train_features['Embarked'] = train_features['Embarked'].fillna('S')
test_features['Embarked'] = test_features['Embarked'].fillna('S')

# one hot encode 
train_features = pd.get_dummies(train_features, columns=['Pclass', 'Sex', 'Embarked'])
test_features = pd.get_dummies(test_features, columns=['Pclass', 'Sex', 'Embarked'])

# save column names
column_names = list(train_features.columns)

save_features = test_features
# Convert to numpy array
train_features = np.array(train_features)
test_features = np.array(test_features)

# Split the data into training and testing sets
split_train_features, split_test_features, split_train_labels, split_test_labels = train_test_split(
    train_features, train_labels, test_size=0.4, random_state=42)

pipe = make_pipeline(StandardScaler(), LogisticRegression())
# pipe.fit(split_train_features, split_train_labels)
pipe.fit(train_features, train_labels)

# predictions = pipe.predict(split_test_features)
predictions = pipe.predict(test_features)

# Get numerical feature importances
# importances = list(pipe.feature_importances_)

# List of tuples with variable and importance
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(column_names, importances)]

# Sort the feature importances by most important first
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

print('-------------------------------------')
print('Logistic Regression')
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
print('{:.2f}%'.format(pipe.score(split_test_features, split_test_labels) * 100))
print('-------------------------------------')

foo = pd.DataFrame({'PassengerId': test_set['PassengerId'], 'Survived': predictions})
print(foo.tail())

foo.to_csv('log_reg_submission.csv', index=False)

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)

sc = StandardScaler()
X_train = sc.fit_transform(train_features)
X_test = sc.transform(test_features)

pd_x_train = pd.DataFrame(X_train, columns=column_names)

split_X_train = sc.fit_transform(split_train_features)
split_X_test = sc.transform(split_test_features)

# Train the model on training data
rf.fit(X_train, train_labels)
# rf.fit(X_train, split_train_labels)

predictions = rf.predict(test_features)
# predictions = rf.predict(split_test_features)

pipe = make_pipeline(StandardScaler(), RandomForestClassifier())
# pipe.fit(split_train_features, split_train_labels)
pipe.fit(train_features, train_labels)
pipe_predictions = pipe.predict(test_features)

print('-------------------------------------')
print('Random Forest Classifier')
# importances = list(rf.feature_importances_)
# print('Mean Absolute Error:', metrics.mean_absolute_error(split_test_labels, predictions))
# print('Mean Squared Error:', metrics.mean_squared_error(split_test_labels, predictions))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(split_test_labels, predictions)))
# print('accuracy: {:.2f}%'.format(round(metrics.accuracy_score(split_test_labels, predictions) * 100)))

# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(column_names, importances)]
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
print('-------------------------------------')

foo = pd.DataFrame({'PassengerId': test_set['PassengerId'], 'Survived': pipe_predictions})
print(foo.tail())

foo.to_csv('rf_pipe_submission.csv', index=False)

gbk = GradientBoostingClassifier()
gbk.fit(X_train, train_labels)
predictions = gbk.predict(test_features)

foo = pd.DataFrame({'PassengerId': test_set['PassengerId'], 'Survived': predictions})
print(foo.tail())

foo.to_csv('gbk_submission.csv', index=False)