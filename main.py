import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


insurance_dataset = pd.read_csv('insurance.csv')
insurance_dataset.head()

insurance_dataset.shape
insurance_dataset.info()

insurance_dataset.isnull().sum()
insurance_dataset.describe()

sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['age'])
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(x="sex", data=insurance_dataset)
plt.title("Sex distribution")
plt.show()

insurance_dataset["sex"].value_counts()

plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['bmi'])
plt.title('BMI Distribution')
plt.show()
insurance_dataset["bmi"].value_counts()

plt.figure(figsize=(6,6))
sns.countplot(x="children", data=insurance_dataset)
plt.title("Children distribution")
plt.show()

insurance_dataset["children"].value_counts()

plt.figure(figsize=(6,6))
sns.countplot(x="smoker", data=insurance_dataset)
plt.title("Smoker distribution")
plt.show()
insurance_dataset["smoker"].value_counts()

plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['charges'])
plt.title('Charges Distribution')
plt.show()
insurance_dataset["charges"].value_counts()

plt.figure(figsize=(6,6))
sns.countplot(x="region", data=insurance_dataset)
plt.title("Region distribution")
plt.show()
insurance_dataset["region"].value_counts()

insurance_dataset.replace({"sex":{"male":0, "female":1}},inplace=True)
insurance_dataset.replace({"smoker":{"yes":0, "no":1}},inplace=True)
insurance_dataset.replace({"region":{"southeast":0, "southwest":1, "northeast":2, "northwest":3}},inplace=True)

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

training_data_prediction = lin_reg_model.predict(X_train)
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared value:', r2_train)

test_data_prediction = lin_reg_model.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared value:', r2_test)

input_data = (19,0,24.5,0,1,0)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = lin_reg_model.predict(input_data_reshaped)
print("The insurance cost is USD", prediction[0])
