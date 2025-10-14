import os
import logging
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import ImageStat

from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer
import text_recognizer.util as util

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Initialize FastAPI app and model
app = FastAPI(title="Text Recognizer API")
model = ParagraphTextRecognizer()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define request model
class ImageData(BaseModel):
    image: str  # base64 string

@app.get("/")
# Health check endpoint
async def index():
    return {"message": "Text Recognizer API is running."}

@app.post("/v1/predict")
# Prediction endpoint
async def predict(image_data: ImageData):
    try:
        image = util.read_b64_image(image_data.image, grayscale=True)
        return await _predict_and_log(image)
    except Exception as e:
        logging.exception("Error during POST prediction")
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/v1/predict")
async def predict_get(image_url: str = Query(..., description="URL of the image to be processed")):
    try:
        logging.info(f"Fetching image from URL: {image_url}")
        image = util.read_image_pil(image_url, grayscale=True)
        return await _predict_and_log(image)
    except Exception as e:
        logging.exception("Error during GET prediction")
        return JSONResponse(status_code=400, content={"error": str(e)})

async def _predict_and_log(image):
    if image is None:
        logging.error("Invalid image input.")
        return JSONResponse(status_code=400, content={"error": "Invalid image input."})

    # Check if the image is blank
    stat = ImageStat.Stat(image)
    if sum(stat.mean) < 5:  # Threshold for blank image
        logging.warning("Received a blank image.")
        return {"predicted_text": "", "warning": "The provided image appears to be blank."}

    try:
        predicted_text = model.predict(image)
        image_stat = ImageStat.Stat(image)
        logging.info(f"METRIC image_mean_intensity {image_stat.mean[0]}")
        logging.info(f"METRIC image_area {image.size[0] * image.size[1]}")
        logging.info(f"METRIC pred_length {len(predicted_text)}")
        logging.info(f"pred {predicted_text}")

        return JSONResponse({"pred": str(predicted_text)})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return JSONResponse(status_code=500, content={"error": "An error occurred during prediction."})
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server.app:app", host="0.0.0.0", port=8000)

    