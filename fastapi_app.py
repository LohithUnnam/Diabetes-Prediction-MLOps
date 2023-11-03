from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel, Field

app = FastAPI()

available_models = {
    "rfc": joblib.load("C:\\Users\\HP\\Desktop\\diabetes_predictor\\saved_model\\rfc_model.pkl")
}

# Load the pre-trained StandardScaler used for specific columns
scaler = joblib.load("C:\\Users\\HP\\Desktop\\diabetes_predictor\\saved_model\\scaler_model.pkl")

class InputData(BaseModel):
    HighBP: int = Field(
        ...,
        description="Indicates whether the individual has high blood pressure.",
        ge=0,  # Greater than or equal to 0
        le=1,  # Less than or equal to 1
    )
    HighChol: int = Field(
        ...,
        description="Indicates whether the individual has high cholesterol.",
        ge=0,  # Greater than or equal to 0
        le=1,  # Less than or equal to 1
    )
    CholCheck: int = Field(
        ...,
        description="Indicates whether the individual has had a cholesterol check in the past 5 years.",
        ge=0,  # Greater than or equal to 0
        le=1,  # Less than or equal to 1
    )
    BMI: float = Field(
        ...,
        description="Body Mass Index, a measure of a person's body fat based on height and weight.",
        ge=0.0,  # Greater than or equal to 0.0
        le=100.0,  # Less than or equal to 100.0 (adjust the upper limit as needed)
    )
    Smoker: int = Field(
        ...,
        description="Asks whether the individual has ever smoked at least 100 cigarettes in their lifetime.",
        ge=0,  # Greater than or equal to 0
        le=1,  # Less than or equal to 1
    )
    Stroke: int = Field(
        ...,
        description="Indicates whether the individual has ever been told that they had a stroke.",
        ge=0,  # Greater than or equal to 0
        le=1,  # Less than or equal to 1
    )
    HeartDiseaseorAttack: int = Field(
        ...,
        description="Indicates whether the individual has coronary heart disease (CHD) or myocardial infarction (MI).",
        ge=0,  # Greater than or equal to 0
        le=1,  # Less than or equal to 1
    )
    PhysActivity: int = Field(
        ...,
        description="Indicates whether the individual engaged in physical activity in the past 30 days (excluding job-related activity).",
        ge=0,  # Greater than or equal to 0
        le=1,  # Less than or equal to 1
    )
    Fruits: int = Field(
        ...,
        description="Indicates whether the individual consumes fruit one or more times per day.",
        ge=0,  # Greater than or equal to 0
        le=1,  # Less than or equal to 1
    )
    Veggies: int = Field(
        ...,
        description="Indicates whether the individual consumes vegetables one or more times per day.",
        ge=0,  # Greater than or equal to 0
        le=1,  # Less than or equal to 1
    )
    HvyAlcoholConsump: int = Field(
        ...,
        description="Indicates whether the individual is classified as a heavy alcohol consumer.",
        ge=0,  # Greater than or equal to 0
        le=1,  # Less than or equal to 1
    )
    AnyHealthcare: int = Field(
        ...,
        description="Indicates whether the individual has any kind of health care coverage, including health insurance.",
        ge=0,  # Greater than or equal to 0
        le=1,  # Less than or equal to 1
    )
    NoDocbcCost: int = Field(
        ...,
        description="Indicates whether there was a time in the past 12 months when the individual needed to see a doctor but could not due to cost.",
        ge=0,  # Greater than or equal to 0
        le=1,  # Less than or equal to 1
    )
    GenHlth: int = Field(
        ...,
        description="Asks the individual to rate their general health on a scale of 1-5, with 1 being excellent and 5 being poor.",
        ge=1,  # Greater than or equal to 1
        le=5,  # Less than or equal to 5
    )
    MentHlth: int = Field(
        ...,
        description="Represents the number of days of poor mental health the individual experienced in the past month on a scale of 1-30 days.",
        ge=1,  # Greater than or equal to 1
        le=30,  # Less than or equal to 30
    )
    PhysHlth: int = Field(
        ...,
        description="Represents the number of days of physical illness or injury the individual experienced in the past month on a scale of 1-30 days.",
        ge=1,  # Greater than or equal to 1
        le=30,  # Less than or equal to 30
    )
    DiffWalk: int = Field(
        ...,
        description="Indicates whether the individual has serious difficulty walking or climbing stairs.",
        ge=0,  # Greater than or equal to 0
        le=1,  # Less than or equal to 1
    )
    Sex: int = Field(
        ...,
        description="Gender of the individual.",
        ge=0,  # Greater than or equal to 0
        le=1,  # Less than or equal to 1
    )
    Age: int = Field(
        ...,
        description="Age category, with 13 levels (_AGEG5YR see codebook) ranging from 1 (18-24) to 13 (80 or older).",
        ge=1,  # Greater than or equal to 1
        le=13,  # Less than or equal to 13
    )
    Education: int = Field(
        ...,
        description="Education level (EDUCA see codebook) on a scale of 1-6, with 1 indicating never attended school or only kindergarten and higher numbers indicating higher education levels.",
        ge=1,  # Greater than or equal to 1
        le=6,  # Less than or equal to 6
    )
    Income: int = Field(
        ...,
        description="Income level (INCOME2 see codebook) on a scale of 1-8, with 1 indicating less than $10,000 and 8 indicating $75,000 or more.",
        ge=1,  # Greater than or equal to 1
        le=8,  # Less than or equal to 8
    )

class ModelSelection(BaseModel):
    model_name: str = Field(
        ...,
        description="Specify the name or identifier of the model you want to use for prediction.",
    )

@app.get("/")
def index():
    return {"Hi":"Welcome Everyone"}

@app.post("/predict/")
async def predict(data: InputData, model_selection: ModelSelection):
    try:
        model_name = model_selection.model_name

        if model_name not in available_models:
            raise HTTPException(status_code=400, detail="Invalid model name")

        # Create a copy of the input data
        data_dict = data.__dict__

        # Create a list of values for the columns that need scaling
        columns_to_scale = ["BMI", "MentHlth", "PhysHlth"]
        scaled_values = [data_dict[col] for col in columns_to_scale]

        # Apply scaling to the selected columns using the loaded StandardScaler
        scaled_values = scaler.transform([scaled_values])[0]

        # Replace the scaled values in the data dictionary
        for col, scaled_value in zip(columns_to_scale, scaled_values):
            data_dict[col] = scaled_value

        # Perform prediction using the selected model
        prediction = available_models[model_name].predict([list(data_dict.values())])

        return {"Diabetes_binary_prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))