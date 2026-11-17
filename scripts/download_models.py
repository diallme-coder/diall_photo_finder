"""Download the OpenCV Zoo face models (Apache-2.0) into ./models. Run once on a machine with internet."""
import os, urllib.request
BASE = "https://github.com/opencv/opencv_zoo/raw/main/models/"
FILES = {
    "face_detection_yunet_2023mar.onnx": "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
dest = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(dest, exist_ok=True)
for name, path in FILES.items():
    out = os.path.join(dest, name)
    if os.path.exists(out) and os.path.getsize(out) > 10000:
        print("have", name); continue
    print("downloading", name)
    urllib.request.urlretrieve(BASE + path, out)
print("done")
