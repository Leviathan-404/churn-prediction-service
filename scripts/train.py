#Import the required libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv', engine='python')

#Changing the TotalCharges column into float
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

#Removing unnecessary column
df.drop(columns='customerID', inplace=True)

#Creating a separate dataframe for the encoded Churn column
df_churn = df[['Churn']]
df_churn = pd.get_dummies(df_churn, columns=['Churn'], dtype=int)
df_churn.reset_index(drop=True, inplace=True)
df_churn.head()

#Dropping the Churn column from the original dataframe to avoid data leakage
df.drop(columns='Churn', inplace=True)

#Encoding the rest of the categorical columns
cat_cols = df.select_dtypes(include='object').columns

#Creating a transformer that applies OrdinalEncoder to text or categorical columns
transformer = make_column_transformer(
    (OrdinalEncoder(), cat_cols),
    remainder='passthrough' #Keep all numerical columns untouched
)

#Transforming the data and saving it back as a dataframe
transformed_array = transformer.fit_transform(df)

##Extracting the exact features names from the transformer
feature_names = transformer.get_feature_names_out()

#Creating a new dataframe with the transformed data and the feature names
df_encoded = pd.DataFrame(transformed_array, columns=feature_names)
df_encoded.head()

#Splitting the data into features and target variable
X = df_encoded
y = df_churn['Churn_Yes']

#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initializing the XGBoost classifier
xgb_model = XGBClassifier(
    n_estimators=100,
    scale_pos_weight=2.78, #Setting scale_pos_weight to heavily penalize missing actual churners
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False
)

#Using StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Computing cross-validation scores
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='accuracy')

#Printing the cross-validation scores and their mean
print("Scores per fold: ", cv_scores)

print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

#Print the standard deviation of the cross-validation scores
print(f"Standard Deviation of CV Accuracy: {cv_scores.std():.4f}")

#Training the model on the training data
xgb_model.fit(X_train, y_train)

#Saving the transformer
joblib.dump(transformer, 'models/transformer.pkl')
print("Transformer artifact successfully saved to models/transformer.pkl")
#Saving the trained model
joblib.dump(xgb_model, 'models/xgb_model.pkl')
print("Model artifact successfully saved to models/xgb_model.pkl")

#Making predictions on the test set
y_pred = xgb_model.predict(X_test) #Generates the predicted labels for the test set
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1] #Generates the predicted probabilities for the positive class
#Printing the classification report
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

#Printing ROC AUC Score
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

#Creating a function to present the confusion matrix pictorially
def plot_confusion_matrix(y,y_predict):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y,y_predict)
    ax = plt.subplot()
    sns.heatmap(cm,
                annot=True,
                ax = ax,
                fmt='d',
                xticklabels = ['No Churn', 'Churn'],
                yticklabels = ['No Churn', 'Churn']
                ) #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

    plt.show()
plot_confusion_matrix(y_test, y_pred)