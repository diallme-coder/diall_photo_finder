"""Face engine. Real engine = OpenCV YuNet (detect) + SFace (embed). Mock engine is for tests only."""
import os
import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
YUNET = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
SFACE = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
MATCH_THRESHOLD = 0.363  # SFace cosine threshold recommended by OpenCV


class EngineUnavailable(RuntimeError):
    pass


class OpenCVEngine:
    name = "opencv-sface"
    threshold = MATCH_THRESHOLD

    def __init__(self):
        if not (os.path.exists(YUNET) and os.path.exists(SFACE)):
            raise EngineUnavailable("Models missing. Run: python scripts/download_models.py")
        self.det = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), 0.8, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(SFACE, "")

    def faces(self, img):
        """Return list of (box[x,y,w,h], embedding float32[128])."""
        h, w = img.shape[:2]
        scale = min(1.0, 1280 / max(h, w))
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, dets = self.det.detect(img)
        out = []
        if dets is None:
            return out
        for d in dets:
            aligned = self.rec.alignCrop(img, d)
            emb = self.rec.feature(aligned).flatten().astype(np.float32)
            box = [int(v / scale) for v in d[:4]]
            out.append((box, emb))
        return out


class MockEngine:
    """TEST ONLY: treats the whole image as one 'face'. Not real recognition."""
    name = "mock"
    threshold = 0.9

    def faces(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        v = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).flatten()
        v -= v.mean()
        return [([0, 0, img.shape[1], img.shape[0]], v)]


def get_engine():
    if os.environ.get("FACE_ENGINE") == "mock":
        return MockEngine()
    return OpenCVEngine()


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
