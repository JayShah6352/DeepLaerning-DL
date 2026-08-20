import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
##preprocessing==start
df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# print(df[:5])
# print(df.columns)
df=df.drop(['customerID'],axis=1)
##df.drop('customerID',axis=columns,inplace=True)
# print(df.columns)
# print(df.shape)#(7043, 20)
# print(df['TotalCharges'].isnull())
# print(df[pd.to_numeric(df['TotalCharges'],errors='coerce').isnull()].shape)##(11,20)
# print(df[df.TotalCharges == ' '])
df1=(df[df.TotalCharges != ' '])
# print(df1.shape)##(7032,20)
# print(df1.TotalCharges)
df2=pd.DataFrame()
df2['TotalCharges']=pd.to_numeric(df1['TotalCharges'])+10
# print(df2.TotalCharges)
print(df1[df1['Churn'] == 'No'])
# print(df1.Churn)
tenure_churm_no=df1[df1['Churn'] == 'No'].tenure
# print(tenure_churm_no)
tenure_churm_yes=df1[df1['Churn'] == 'Yes'].tenure
# print(tenure_churm_yes)
# plt.hist(tenure_churm_no['tenure'], label='No')
# plt.hist(tenure_churm_yes['tenure'], label='Yes')
# plt.xlabel('Tenure')
# plt.ylabel('Number of Customers')
# plt.title('Customer Churn Distribution')
# # plt.show()
# plt.hist([tenure_churm_yes,tenure_churm_no],rwidth=0.95,color=['green','red'],label=['Churn=Yes','Churn=No'])
# plt.legend()
# # plt.show()
# tenure_churm_mc_no=df1[df1['Churn'] == 'No'].MonthlyCharges
# print(tenure_churm_mc_no)
# tenure_churm_mc_yes=df1[df1['Churn'] == 'Yes'].MonthlyCharges
# print(tenure_churm_mc_yes)
# plt.xlabel('Tenure')
# plt.ylabel('Number of Customers')
# plt.title('Customer Churn Distribution')
# # plt.show()
# plt.hist([tenure_churm_mc_yes,tenure_churm_mc_no],rwidth=0.95,color=['green','red'],label=['Churn=Yes','Churn=No'])
# # plt.legend()
# # plt.show()

def print_unique_col_value(df):
    for colum in df:
        if df[colum].dtype== 'object':
            print(f'{colum}:{df[colum].unique()}')
# print_unique_col_value(df1)
df1.replace({'No internet service':'No','No phone service':'No'},inplace=True)
# print_unique_col_value(df1)
df1['gender'].replace({'Female':1,'Male':0},inplace=True)
# print_unique_col_value(df1)
columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']

for col in columns:
    df1[col] = df1[col].replace({'Yes': 1, 'No': 0})
# print_unique_col_value(df1)

# for col in df1:
    # print(f'{col}: {df1[col].unique()}')
# print(df1.gender.unique())
# print(df1.InternetService)
df2=pd.get_dummies(data=df1,columns=['InternetService','Contract','PaymentMethod'])
# print(df2.columns)
# print(df2[['InternetService_DSL','InternetService_Fiber optic','InternetService_No']])
# print(df2.head())
# print(df2.tenure)
# print(df2.MonthlyCharges)
# print(df2.TotalCharges)
# print(df2.dtypes)
# print(df2[['tenure','MonthlyCharges','TotalCharges']])
scaler = MinMaxScaler()
df2[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df2[['tenure', 'MonthlyCharges', 'TotalCharges']])
# print(df2[['tenure', 'MonthlyCharges', 'TotalCharges']])
##preprocessing==end

###Training model=start
X=df2.drop('Churn',axis='columns')
y=df2['Churn']
# print(X.shape)(7032,26)
# print(y.shape)(7032),
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
# print(len(X_train.columns))#26
# print(X_train.shape)##(5625,26)
# print(X_test.shape)##(1407,26)
# print(y_train.shape)##(5625,)
# print(y_test.shape)##(1407,)

model=tf.keras.Sequential([
    tf.keras.layers.Dense(26,input_shape=(26,),activation='relu'),
    tf.keras.layers.Dense(15,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=5)
model.evaluate(X_test,y_test)
y_test_pred = model.predict(X_test)
# print (y_test_pred[:10])
# print(y_test[: 10])
y_pred = []
for i in y_test_pred:
    if i >= 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

print(y_pred[:10])
print(y_test[:10])
