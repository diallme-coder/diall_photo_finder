"""Face engine. Real engine = OpenCV YuNet (detect) + SFace (embed). Mock engine is for tests only."""
import os
import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
YUNET = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
SFACE = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
MATCH_THRESHOLD = 0.363  # SFace cosine threshold recommended by OpenCV (used for indexing/grouping)
CLAIM_THRESHOLD = 0.5    # stricter: needed to link a tag to a member account


class EngineUnavailable(RuntimeError):
    pass


def yaw_from_landmarks(det):
    """Head yaw estimate from a YuNet detection row: (nose_x - eye_midpoint_x) / eye_distance.
    ~0 facing the camera; positive = nose toward image right. Landmarks are det[4:14]."""
    re_x, le_x, nose_x = float(det[4]), float(det[6]), float(det[8])
    dist = abs(le_x - re_x)
    if dist < 1e-6:
        return 0.0
    return (nose_x - (re_x + le_x) / 2.0) / dist


class OpenCVEngine:
    name = "opencv-sface"
    threshold = MATCH_THRESHOLD
    claim_threshold = CLAIM_THRESHOLD

    def __init__(self):
        if not (os.path.exists(YUNET) and os.path.exists(SFACE)):
            raise EngineUnavailable("Models missing. Run: python scripts/download_models.py")
        self.det = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), 0.8, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(SFACE, "")

    def faces(self, img):
        """Return list of (box[x,y,w,h], embedding float32[128], yaw)."""
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
            out.append((box, emb, yaw_from_landmarks(d)))
        return out


class MockEngine:
    """TEST ONLY: treats the whole image as one 'face'. Not real recognition.
    The embedding ignores the 8px left/right margins; 'yaw' is read from their brightness
    difference ((right - left) / 200) so tests can simulate head poses deterministically."""
    name = "mock"
    threshold = 0.9
    claim_threshold = 0.95

    def faces(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        yaw = float((g[:, -8:].mean() - g[:, :8].mean()) / 200.0)
        e = g.copy()
        e[:, :8] = e[:, 8:9]
        e[:, -8:] = e[:, -9:-8]
        v = cv2.resize(e, (8, 8), interpolation=cv2.INTER_AREA).flatten()
        v -= v.mean()
        return [([0, 0, img.shape[1], img.shape[0]], v, yaw)]


def get_engine():
    if os.environ.get("FACE_ENGINE") == "mock":
        return MockEngine()
    return OpenCVEngine()


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
