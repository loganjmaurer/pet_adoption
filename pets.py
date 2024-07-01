import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#read data
df = pd.read_csv("pet_adoption_data.csv")
print(df.head())

#convert to numeric
lr = preprocessing.LabelEncoder()
df['PetType'] = lr.fit_transform(df['PetType'])
df['Breed'] = lr.fit_transform(df['Breed'])
df['Color'] = lr.fit_transform(df['Color'])
df['Size'] = lr.fit_transform(df['Size'])

#exploratory analysis / summary statistics and correlations
df.describe()
correlations = df.corr()
sns.heatmap(correlations, xticklabels=correlations.columns, yticklabels=correlations.columns, annot=True)
#plt.savefig("corrMat.png")

#test/train split
X= df.drop('AdoptionLikelihood',axis=1)
y=df['AdoptionLikelihood']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#scale data 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#fit models
statMod = LogisticRegression()
MLMod = GaussianNB()

statMod.fit(X_train, y_train)
MLMod.fit(X_train, y_train)

#obtain predictions
statPred = statMod.predict(X_test)
MLPred = MLMod.predict(X_test)

#show confusion matrices
print(classification_report(y_test,statPred))
print(classification_report(y_test, MLPred))