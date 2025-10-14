from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import ImageStat
from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer
import text_recognizer.util as util

app = FastAPI(title="Text Recognizer API")
model = ParagraphTextRecognizer()


class ImageRequest(BaseModel):
    image_url: str


@app.post("/predict")
async def predict(req: ImageRequest):
    """Provide main prediction API"""
    image = _load_image(req.image_url)
    if isinstance(image, str):
        # Error message returned from _load_image
        raise HTTPException(status_code=400, detail=image)

    pred = model.predict(image)
    image_stat = ImageStat.Stat(image)

    # Print metrics (can be used in CloudWatch / logs)
    print("METRIC image_mean_intensity {}".format(image_stat.mean[0]))
    print("METRIC image_area {}".format(image.size[0] * image.size[1]))
    print("METRIC pred_length {}".format(len(pred)))
    print("INFO pred {}".format(pred))

    return {"pred": str(pred)}


def _load_image(image_url: str):
    if not image_url:
        return "no image_url provided"
    print("INFO url {}".format(image_url))
    try:
        return util.read_image_pil(image_url, grayscale=True)
    except Exception as e:
        return f"Failed to load image: {e}"