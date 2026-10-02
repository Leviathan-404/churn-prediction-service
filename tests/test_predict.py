from predict import predict_churn, get_risk_level
import pytest

@pytest.fixture
def sample_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 70.35
    }

@pytest.fixture
def churn_results(sample_customer):
    # Test to check if the predict_churn function
    # returns a dictionary with the expected keys and types
    results = predict_churn(sample_customer)
    return results

def test_to_predict_churn(churn_results):
    assert isinstance(churn_results, dict)

def test_probability(churn_results):
    #Testing to ensure range is between 0 and 1
    assert 0<= churn_results['Churn_probability'] <= 1

def test_keys(churn_results):
    #Test to ensure all keys are presents in the results dictionary
    assert "Churn_prediction" in churn_results
    assert "Churn_probability" in churn_results
    assert "Risk_level" in churn_results
    assert "Recommended_action" in churn_results

def test_risk_levels():
    assert get_risk_level(0.8) == "HIGH RISK"
    assert get_risk_level(0.5) == "MEDIUM RISK"
    assert get_risk_level(0.2) == "LOW RISK"