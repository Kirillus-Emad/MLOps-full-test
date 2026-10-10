import mlflow
from mlflow.tracking import MlflowClient


# ----------------------------------------------
# MLflow setup
# ----------------------------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

registered_model_name = "ProfitPredictionModel"
alias = "champion"


# ----------------------------------------------
# Get champion model version
# ----------------------------------------------

model_version = client.get_model_version_by_alias(
    name=registered_model_name,
    alias=alias
)

print("=" * 50)
print("MODEL INFORMATION")
print("=" * 50)

print(f"Registered model : {model_version.name}")
print(f"Version          : {model_version.version}")
print(f"Alias            : {alias}")


# ----------------------------------------------
# Print model metadata
# ----------------------------------------------

print("\n" + "=" * 50)
print("MODEL METADATA")
print("=" * 50)

for key, value in model_version.tags.items():
    print(f"{key:15}: {value}")


# ----------------------------------------------
# Load model
# ----------------------------------------------

model_uri = f"models:/{registered_model_name}@{alias}"

model = mlflow.pyfunc.load_model(model_uri)

print("\nModel loaded successfully!")


# ----------------------------------------------
# Make prediction
# ----------------------------------------------

# Four numeric input features
X_new = [
    [100, 50, 20, 10]
]

prediction = model.predict(X_new)

print("\n" + "=" * 50)
print("PREDICTION")
print("=" * 50)

print(f"Input      : {X_new[0]}")
print(f"Prediction : {prediction[0]}")