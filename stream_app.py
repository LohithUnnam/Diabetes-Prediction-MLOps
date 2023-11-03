import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset

# Load the data
df = pd.read_csv("C:\\Users\\HP\\Desktop\\diabetes_predictor\\data\\diabetes_preprocessed.csv")

for column in df.columns:
    df[column] = df[column].astype('int')

# Define target and prediction columns
target = "Diabetes_binary"
prediction = "prediction"

# Define feature columns
numerical_features = ['BMI', 'MentHlth', 'PhysHlth']
categorical_features = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
                        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                        'DiffWalk', 'Sex', 'Age', 'Education', 'Income']

# Sample data for the app
current_data = df.sample(400)

# Model selection in the sidebar
selected_model = st.sidebar.selectbox("Select a Model", ["Random Forest", "Logistic Regression", "SVM"])

# Train the selected model
if selected_model == "Random Forest":
    regressor = RandomForestClassifier(random_state=1, n_estimators=50)
elif selected_model == "Logistic Regression":
    regressor = LogisticRegression(random_state=1)
elif selected_model == "SVM":
    regressor = SVC(probability=True, random_state=1)

regressor.fit(current_data[numerical_features + categorical_features], current_data[target])

# Make probabilistic predictions (predict_proba instead of predict)
current_prob_predictions = regressor.predict_proba(current_data[numerical_features + categorical_features])

# Add probabilistic predictions to the DataFrame
current_data['prediction_probability'] = current_prob_predictions[:, 1]  # Assuming class 1 is positive

# Set up ColumnMapping with the new prediction column
column_mapping = ColumnMapping()
column_mapping.target = target
column_mapping.prediction = 'prediction_probability'  # Update the prediction column name
column_mapping.numerical_features = numerical_features
column_mapping.categorical_features = categorical_features

st.title("Diabetes Binary Classification Report")

# Show the updated classification performance report
classification_performance = Report(metrics=[ClassificationPreset()])
classification_performance.run(current_data=current_data, reference_data=None, column_mapping=column_mapping)
st.write("Classification Report:")
html_report = classification_performance.get_html()
st.components.v1.html(
    f'<div style="width: 100%; max-width: 1200px;">{html_report}</div>',
    height=1000,
    scrolling=True
)
