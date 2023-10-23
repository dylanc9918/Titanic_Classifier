import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
from sklearn.metrics import classification_report


# read in the dataframe from csv file
df_train_all = pd.read_csv(r"C:\Users\Admin\Downloads\train.csv")

df_test = pd.read_csv(r"C:\Users\Admin\Downloads\test.csv")


# create label array of if the passenger survived or not and drop this from df_train
labels = np.array(df_train_all["Survived"])
df_train = df_train_all.drop("Survived", axis=1)

# view categorical variables
sns.countplot(x=pd.qcut(df_train_all["Age"], 10),
              hue='Survived', data=df_train_all)


# feature engineering for categorical variables
encoder = ce.OrdinalEncoder(df_train.columns)

# fit and transform
df_train_encode = encoder.fit_transform(df_train)
df_test_encode = encoder.fit_transform(df_test)


# fill in missing values in age column with most common age
age_value = df_train_encode["Age"].value_counts().index[0]

df_train_encode["Age"].fillna(age_value, inplace=True)
df_test_encode["Age"].fillna(age_value, inplace=True)

# fill in missing values in fare column with most common fare
emb_val = df_train_encode["Embarked"].value_counts().index[0]
df_train_encode["Embarked"].fillna(emb_val, inplace=True)
df_test_encode["Embarked"].fillna(emb_val, inplace=True)

# fill in missing values in fare column with most common fare
fare_val = df_train_encode["Fare"].value_counts().index[0]
df_train_encode["Fare"].fillna(emb_val, inplace=True)
df_test_encode["Fare"].fillna(emb_val, inplace=True)


features = ["Sex", "Age", "Pclass", "Fare"]

# saving final training set as an array
df_train_encode = df_train_encode[features]
df_test_encode = df_test_encode[features]

# creating random forest model
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    min_samples_split=5,
    max_leaf_nodes=25,
    max_features="auto")

# fit to train data
rf.fit(df_train_encode, labels)


# prediciting the test data
predictions = rf.predict(df_test_encode)


# saving it to an output file
output_rf = pd.DataFrame(
    {'PassengerId': df_test.PassengerId, 'Survived': predictions.flatten()})


# Save the submission file
output_rf.to_csv('output_rf.csv', index=False)
