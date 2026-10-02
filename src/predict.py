#Import the required libraries
import joblib
import pandas as pd

#Load the transformer
transformer = joblib.load('models/transformer.pkl')

#Load the trained model
model = joblib.load('models/xgb_model.pkl')

# Extract exact features names
# expected_features = model.get_booster().feature_names
# print(expected_features)

#Defining a function to accept input for prediction
def prepare_input(customer_data: dict):
    #Example customer_data = {"tenure": 12, 
    # "MonthlyCharges": 100, 
    # "TotalCharges": 500, and more}
    
    #Convert dictionary to dataframe
    df = pd.DataFrame([customer_data])

    return df

def predict_churn(df):
    #Calling prepare_input function
    df = prepare_input(df)

    #Running the transformer on the input data
    df_array = transformer.transform(df)

    #Get the expected features from the model
    feature_names = transformer.get_feature_names_out()
    
    df_encoded = pd.DataFrame(df_array, columns=feature_names)

    #Making the prediction
    prediction = model.predict(df_encoded) #Predicting 0 or 1 (not churn or churn)
    
    #Getting the probability
    probability = model.predict_proba(df_encoded)[0][1] #Getting the probability of churn (class 1)

    #Getting the risk level
    risk_level = get_risk_level(probability)

    #Getting the recommended action
    recom_action = get_recommendation(risk_level)
    return {
        "Churn_prediction": int(prediction[0]),
        "Churn_probability": float(probability),
        "Risk_level": risk_level,
        "Recommended_action": recom_action
    }

#Defining a function that accepts the probability and classifies it based on risk
def get_risk_level(probability: float):
    if probability >= 0.7:
        return "HIGH RISK"
    if probability >= 0.4:
        return "MEDIUM RISK"
    if probability < 0.4:
        return "LOW RISK"

#Defining a function that gives a recommended action based on the risk
def get_recommendation(risk_level: str):
    if risk_level == "HIGH RISK":
        return (
            "Immediate proactive outreach required. "
        "Offer a 12-month contract lock in with a 15% discount or premium tech support."
        )
    elif risk_level == "MEDIUM RISK":
        return (
            "Target for engagement enhancement. "
            "Encourage enrollment in automatic payments and offer free trials of add-on services"
        )
    elif risk_level == "LOW RISK":
        return (
            "Maintain standard workflow. "
            "Candidate for referral incentives and plan upgrade promotions."
        )