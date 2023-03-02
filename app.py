import json
from fastapi import FastAPI, File
from uvicorn import run as app_run
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from document_classifier.pipeline.training_pipeline import TrainPipeline
from document_classifier.pipeline.prediction_pipeline import ModelPredictor
from document_classifier.constant import *

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/train")
async def training():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    

@app.post("/predict")
async def prediction(image_file: bytes = File(description="A file read as bytes")):
    try:
        prediction_pipeline = ModelPredictor()

        result = prediction_pipeline.initiate_model_predictor(image_file)

        json_str = json.dumps(result, indent=4, default=str)
        
        return Response(content=json_str, media_type='application/json')

    except Exception as e:
        JSONResponse(content = f"Error Occurred! {e}", status_code=500)
    




if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)