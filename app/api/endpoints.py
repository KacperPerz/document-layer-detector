from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from ..core.model import predict, get_label_name
from ..utils.drawing import draw_detections
import numpy as np
import cv2
from pydantic import BaseModel
from typing import List
import io

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


@router.post("/detect/")
async def detect_layout(
    file: UploadFile = File(...),
    format: str = Query("json", enum=["json", "image", "both"])  # response format
):
    """
    Accepts an image file and returns detected elements.
    format=json|image|both
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    layout = predict(img)

    results = [
        {
            "x_1": block.block.x_1,
            "y_1": block.block.y_1,
            "x_2": block.block.x_2,
            "y_2": block.block.y_2,
            "type": get_label_name(block.type),
            "score": block.score,
        } for block in layout
    ]

    if format == "json":
        return JSONResponse({"detections": results})

    annotated = draw_detections(img, layout)
    ok, buf = cv2.imencode(".png", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")

    if format == "image":
        return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")

    # both: return multipart? simplest: return json with base64 image
    import base64
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return JSONResponse({"detections": results, "image_base64": b64})
