from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_churn, get_risk_level, get_recommendation, prepare_input
from enum import Enum
from typing import Literal

class GenderEnum(str, Enum):
    Male  = "Male"
    Female = "Female"

class PartnerEnum(str, Enum):
    Yes = "Yes"
    No = "No"

class DependentsEnum(str, Enum):
    Yes = "Yes"
    No = "No"

class PhoneServiceEnum(str, Enum):
    Yes = "Yes"
    No = "No"

class MultipleLinesEnum(str, Enum):
    Yes = "Yes"
    No = "No"
    no_phone_service = "No phone service"

class InternetServiceEnum(str, Enum):
    DSL = "DSL"
    FiberOptic = "Fiber optic"
    No = "No"

class OnlineSecurityEnum(str, Enum):
    Yes = "Yes"
    No = "No"
    no_internet_service = "No internet service"

class OnlineBackupEnum(str, Enum):
    Yes = "Yes"
    No = "No"
    no_internet_service = "No internet service"

class DeviceProtectionEnum(str, Enum):
    Yes = "Yes"
    No = "No"
    no_internet_service = "No internet service"

class TechSupportEnum(str, Enum):
    Yes = "Yes"
    No = "No"
    no_internet_service = "No internet service"

class StreamingTVEnum(str, Enum):
    Yes = "Yes"
    No = "No"
    no_internet_service = "No internet service"

class StreamingMoviesEnum(str, Enum):
    Yes = "Yes"
    No = "No"
    no_internet_service = "No internet service"

class ContractEnum(str, Enum):
    MonthToMonth = "Month-to-month"
    OneYear = "One year"
    TwoYear = "Two year"

class PaperlessBillingEnum(str, Enum):
    Yes = "Yes"
    No = "No"

class PaymentMethodEnum(str, Enum):
    ElectronicCheck = "Electronic check"
    MailedCheck = "Mailed check"
    BankTransfer = "Bank transfer (automatic)"
    CreditCard = "Credit card (automatic)"


app = FastAPI(
    title="Telco Churn Prediction API",
    description="This API predicts the likelihood of a customer churning based on their data.",
    version="1.0.0"
)

#Return API name and status
@app.get("/")
def get_api_status():
    return {
        "api_name": "Telco Churn Prediction API",
        "status": "online"
    }

@app.get("/health")
def get_api_health():
    return {
        "status": "healthy"
    }

class CustomerData(BaseModel):
    gender: GenderEnum
    SeniorCitizen: Literal[0, 1]
    Partner: PartnerEnum
    Dependents: DependentsEnum
    tenure: int
    PhoneService: PhoneServiceEnum
    MultipleLines: MultipleLinesEnum
    InternetService: InternetServiceEnum
    OnlineSecurity: OnlineSecurityEnum
    OnlineBackup: OnlineBackupEnum
    DeviceProtection: DeviceProtectionEnum
    TechSupport: TechSupportEnum
    StreamingTV: StreamingTVEnum
    StreamingMovies: StreamingMoviesEnum
    Contract: ContractEnum
    PaperlessBilling: PaperlessBillingEnum
    PaymentMethod: PaymentMethodEnum
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(customer_data: CustomerData):
    #Convert pydantic model into a dictionary
    customer_data_dict = customer_data.model_dump()

    #Running the dataframe conversion function
    df = prepare_input(customer_data_dict)

    #Running the churn predictor function
    results = predict_churn(df)

    #Getting the risk level
    risk = get_risk_level(results['probability'])

    #Getting the recommended action
    recommended_action = get_recommendation(risk)

    #Return the prediction and risk level
    return {
        "Churn_prediction": results['prediction'],
        "Churn_probability": round(results['probability'], 3),
        "Risk_level": risk,
        "Recommended_action": recommended_action
    }
