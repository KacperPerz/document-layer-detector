import layoutparser as lp
import torch

model = None
LABEL_MAP = {
    73: "Book"
}

def get_label_name(label_id):
    try:
        return LABEL_MAP.get(int(label_id), str(label_id))
    except Exception:
        return str(label_id)

def get_model():
    """
    Loads and returns the layout detection model.
    Initializes the model on the first call, attempting to use MPS (Apple Silicon GPU) if available.
    Uses a standard COCO model from the Detectron2 model zoo.
    """
    global model
    if model is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        model = lp.Detectron2LayoutModel(
            config_path='/app/model_weights/config.yml',
            model_path='/app/model_weights/model_final.pkl',
            label_map=LABEL_MAP,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8, "MODEL.DEVICE", device]
        )
    return model

def predict(image):
    """
    Performs layout prediction on the given image.
    """
    model = get_model()
    return model.detect(image)
