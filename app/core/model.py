import layoutparser as lp

model = None

def get_model():
    """
    Loads and returns the layout detection model.
    Initializes the model on the first call.
    """
    global model
    if model is None:
        model = lp.Detectron2LayoutModel(
            config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
        )
    return model

def predict(image):
    """
    Performs layout prediction on the given image.
    """
    model = get_model()
    return model.detect(image)
