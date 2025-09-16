import layoutparser as lp
import torch

model = None

# PubLayNet document layout labels
PUBLAYNET_LABELS = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}

def get_label_name(label_id):
    try:
        return PUBLAYNET_LABELS.get(int(label_id), str(label_id))
    except Exception:
        return str(label_id)

def get_model():
    """
    Loads and returns the PubLayNet layout detection model (document layout).
    Uses MPS when available, otherwise CPU.
    """
    global model
    if model is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        # Prefer a local, pre-downloaded weight to avoid runtime download failures in Docker
        local_weights = '/app/model_weights/publaynet_frcnn_r50_fpn_3x.pth'
        try:
            model = lp.Detectron2LayoutModel(
                config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                model_path=local_weights,
                label_map=PUBLAYNET_LABELS,
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5, "MODEL.DEVICE", device]
            )
        except Exception:
            # Fallback to remote if local file missing; layoutparser will attempt to fetch
            model = lp.Detectron2LayoutModel(
                config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                label_map=PUBLAYNET_LABELS,
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5, "MODEL.DEVICE", device]
            )
    return model

def predict(image):
    model = get_model()
    return model.detect(image)
