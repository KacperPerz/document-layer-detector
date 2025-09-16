from fastapi import APIRouter, File, UploadFile, HTTPException
from ..core.model import predict, get_label_name
import numpy as np
import cv2
from pydantic import BaseModel
from typing import List

router = APIRouter()

class BoundingBox(BaseModel):
    x_1: float
    y_1: float
    x_2: float
    y_2: float
    type: str
    score: float

class DetectionResponse(BaseModel):
    detections: List[BoundingBox]


@router.post("/detect/", response_model=DetectionResponse)
async def detect_layout(file: UploadFile = File(...)):
    """
    Accepts an image file and returns the detected layout elements.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    layout = predict(img)

    results = [
        BoundingBox(
            x_1=block.block.x_1,
            y_1=block.block.y_1,
            x_2=block.block.x_2,
            y_2=block.block.y_2,
            type=get_label_name(block.type),
            score=block.score
        ) for block in layout
    ]

    return DetectionResponse(detections=results)
