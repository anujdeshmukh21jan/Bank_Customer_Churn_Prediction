from django.shortcuts import render
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

__model =None
# Create your views here.
def index(request):
    return render(request,"index.html")

def predictData(request):
    Country = request.POST["country"]
    c_score = request.POST['c_score']
    gender = request.POST["gender"]
    age = request.POST["age"]
    tenure = request.POST["tenure"]
    balance = request.POST["balance"]
    numofproducts = request.POST["numofproducts"]
    HasCrCard = request.POST["HasCrCard"]
    IsActiveMember = request.POST["IsActiveMember"]
    EstimatedSalary = request.POST["EstimatedSalary"]

    # print(Country,c_score,gender,age,tenure,balance,numofproducts,HasCrCard,IsActiveMember,EstimatedSalary)

    # xgbClassifier_loaded = joblib.load('xgb_classifier_model.pkl')
    global __model
    predict=None
    if __model is None:
        with open('./ML_model/xgb_classifier_model.pkl','rb') as f:
            __model = joblib.load(f)

    df = create_df(Country,c_score,gender,age,tenure,balance,numofproducts,HasCrCard,IsActiveMember,EstimatedSalary)
    scaled_df=feature_scale(df)
    print(scaled_df['remainder__Age'][0])

    print(gender,type(gender))
    gender=float(gender)
    print(gender,type(gender))
    HasCrCard=float(HasCrCard)
    IsActiveMember=float(IsActiveMember)


    if Country==0:
        predict=__model.predict([[1,0,0,scaled_df['remainder__CreditScore'][0],gender,scaled_df['remainder__Age'][0],scaled_df['remainder__Tenure'][0], scaled_df['remainder__Balance'][0],  scaled_df['remainder__NumOfProducts'][0],  HasCrCard,IsActiveMember,scaled_df['remainder__EstimatedSalary'][0]   ]])
    if Country==1:
        predict=__model.predict([[0,1,0,  scaled_df['remainder__CreditScore'][0],gender,scaled_df['remainder__Age'][0],scaled_df['remainder__Tenure'][0], scaled_df['remainder__Balance'][0],  scaled_df['remainder__NumOfProducts'][0],  HasCrCard,IsActiveMember,scaled_df['remainder__EstimatedSalary'][0] ]])
    else:
        predict=__model.predict([[0,0,1,  scaled_df['remainder__CreditScore'][0],gender,scaled_df['remainder__Age'][0],scaled_df['remainder__Tenure'][0], scaled_df['remainder__Balance'][0],  scaled_df['remainder__NumOfProducts'][0],  HasCrCard,IsActiveMember,scaled_df['remainder__EstimatedSalary'][0] ]])

    print(predict)
    return render(request,"index.html",{"prediction":predict})


def create_df(Country,c_score,gender,age,tenure,balance,numofproducts,HasCrCard,IsActiveMember,EstimatedSalary):
    
    data = {
        'Country': [Country],
        'remainder__CreditScore': [c_score],
        'Gender': [gender],
        'remainder__Age': [age],
        'remainder__Tenure': [tenure],
        'remainder__Balance': [balance],
        'remainder__NumOfProducts': [numofproducts],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'remainder__EstimatedSalary': [EstimatedSalary]
    }

    df = pd.DataFrame(data)
    print(df)
    return df

def feature_scale(df):
    # scaler = MinMaxScaler()
    scaler_loaded = joblib.load('./ML_model/scaler.pkl')
    cols_to_scale = ['remainder__CreditScore', 'remainder__Age', 'remainder__Tenure', 'remainder__Balance', 'remainder__NumOfProducts', 'remainder__EstimatedSalary']
    df[cols_to_scale] = scaler_loaded.transform(df[cols_to_scale])
    print(df)
    return df