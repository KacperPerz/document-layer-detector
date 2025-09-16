import layoutparser as lp
import torch

model = None

def get_model():
    """
    Loads and returns the layout detection model.
    Initializes the model on the first call, attempting to use MPS (Apple Silicon GPU) if available.
    """
    global model
    if model is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        model = lp.Detectron2LayoutModel(
            config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8, "MODEL.DEVICE", device]
        )
    return model

def predict(image):
    """
    Performs layout prediction on the given image.
    """
    model = get_model()
    return model.detect(image)
