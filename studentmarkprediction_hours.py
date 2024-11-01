
 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

 
from google.colab import drive
drive.mount('/content/drive')

 
df = pd.read_csv('/content/drive/MyDrive/StudentMarkPrediction_hours/StudentMarksDataset.csv')

 
df.head()

 
df.info()

 
df.shape

 
df.isnull().sum()

 
df['StudyHrs'].fillna(df['StudyHrs'].mean(), inplace=True)

 
df.isnull().sum()

 
df.duplicated().sum()

 
df.head()

 
df.nunique()

 
df.describe()

 
plt.scatter(x =df.StudyHrs, y = df.Marks)
plt.xlabel("Students Study Hours")
plt.ylabel("Students marks")
plt.title("Scatter Plot of Students Study Hours vs Students marks")
plt.show()

 
df_corr = df.corr()                          
plt.figure(figsize=(10,8))
sns.heatmap(df_corr, annot=True, cmap='coolwarm')   
plt.show()

 
for i in df.select_dtypes(include='number').columns:
  sns.histplot(data=df,x=i)
  plt.show()

 
#Boxplot to identify the outliers
#using numerical cols
for i in df.select_dtypes(include='number').columns:
  sns.boxplot(data=df,x=i)
  plt.show()

 
#finding outlier using z-score
import scipy.stats as stas
z_scores = stas.zscore(df)     
threshold = 3
print("Size before removing outliers",df.shape)
outlier_df = df[(z_scores>threshold).any(axis=1)]
df = df[(z_scores<=threshold).all(axis=1)]
print("Size after removing outliers",df.shape)

 
X = df.drop('Marks',axis=1)
y = df['Marks']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=2)

 
print(X.shape, X_train.shape, X_test.shape)

 

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

scalar.fit(df)

df_scaled = pd.DataFrame(scalar.transform(df), columns=df.columns)

 
from sklearn.linear_model import LinearRegression

 
model = LinearRegression()

 
model.fit(X_train, y_train)

 
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error

 
y_pred_train =model.predict(X_train)
print("Mean squared error:",mean_squared_error(y_train,y_pred_train))
print('\n')
print("Mean Absolute error:",mean_absolute_error(y_train,y_pred_train))
print('\n')
print("R-squared:",r2_score(y_train,y_pred_train))
print('\n')

 
y_pred =model.predict(X_test)
print("Mean squared error:",mean_squared_error(y_test,y_pred))
print('\n')
print("Mean Absolute error:",mean_absolute_error(y_test,y_pred))
print('\n')
print("R-squared:",r2_score(y_test,y_pred))
print('\n')

 
plt.scatter(X_test, y_test)
plt.plot(X_train, model.predict(X_train), color = "r")

 
#Marks Prediction based upon study hours
model.predict([[9]])[0]

 
with open('StudentMarkPrediction_model.pkl', 'wb') as file:
    pickle.dump(model, file)


