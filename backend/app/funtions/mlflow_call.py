import mlflow

def call_model_at_stage(registered_model_name, stage, data):
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{registered_model_name}/{stage}"
    )
    # Evaluate the model
    predictions = model.predict(data)
    return predictions