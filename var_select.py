import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
from sklearn.metrics import classification_report


# read in the dataframe from csv file
df_train_all = pd.read_csv(r"C:\Users\Admin\Downloads\train.csv")

df_test_all = pd.read_csv(r"C:\Users\Admin\Downloads\test.csv")


# view categorical variables
sns.countplot(x=pd.qcut(df_train_all["Age"], 10),
              hue='Survived', data=df_train_all)
