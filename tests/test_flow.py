import base64, io, os, sys, tempfile
os.environ["FACE_ENGINE"] = "mock"
os.environ["DATA_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2, numpy as np
import app as A

def person(seed):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    return cv2.resize(small, (160, 160), interpolation=cv2.INTER_NEAREST)

def jpg(img): return cv2.imencode(".jpg", img)[1].tobytes()
def noisy(img, s=8):
    return np.clip(img.astype(int) + np.random.default_rng().normal(0, s, img.shape), 0, 255).astype(np.uint8)
def durl(img): return "data:image/jpeg;base64," + base64.b64encode(jpg(img)).decode()

class C:
    def __init__(s): s.c = A.app.test_client(); s.t = None
    def tok(s):
        s.c.get("/"); 
        with s.c.session_transaction() as ss: return ss["csrf"]
    def post(s, url, **kw):
        kw.setdefault("headers", {})["X-CSRF"] = s.tok(); return s.c.post(url, **kw)
    def signup(s, u, role):
        return s.post("/register", data={"username": u, "password": "password123", "role": role})

def selfie(cl, img, consent=True):
    ch = cl.post("/api/consent").get_json()["challenge"]
    return cl.post("/api/find-me", json={"challenge": ch, "frames": [durl(noisy(img)) for _ in range(3)]})

def test_full_flow():
    ph, alice, bob, eve = C(), C(), C(), C()
    ph.signup("pro", "photographer"); alice.signup("alice", "member"); bob.signup("bob", "member"); eve.signup("eve", "member")
    A_, B_ = person(1), person(2)
    # upload without consent rejected
    r = ph.post("/upload", data={"file": (io.BytesIO(jpg(A_)), "a.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 400
    ids = []
    for img in (noisy(A_), noisy(A_), noisy(B_)):
        r = ph.post("/upload", data={"file": (io.BytesIO(jpg(img)), "x.jpg"), "consent": "yes"}, content_type="multipart/form-data")
        assert r.status_code == 200 and r.get_json()["faces"] == 1; ids.append(r.get_json()["id"])
    with A.app.app_context():
        tags = A.db().execute("SELECT tag FROM tags").fetchall()
        assert len(tags) == 2 and all(t["tag"].startswith("anon-") for t in tags)
    # no consent -> blocked
    assert alice.post("/api/find-me", json={"challenge": "x", "frames": []}).status_code == 403
    r = selfie(alice, A_); j = r.get_json(); assert j["matched"] and j["count"] == 2
    page = alice.c.get("/dashboard").get_data(as_text=True)
    assert f"/media/{ids[0]}" in page and f"/media/{ids[1]}" in page and f"/media/{ids[2]}" not in page
    assert alice.c.get(f"/media/{ids[2]}").status_code == 404      # not her photo
    assert bob.c.get(f"/media/{ids[0]}").status_code == 404
    assert ph.c.get(f"/media/{ids[0]}").status_code == 200
    # Eve claiming Alice's face with Alice's photo shown to camera: tag already claimed -> nothing revealed
    assert selfie(eve, A_).get_json()["matched"] is False
    # Bob finds his own
    assert selfie(bob, B_).get_json()["count"] == 1
    # static image replayed as 3 identical frames is rejected
    ch = bob.post("/api/consent").get_json()["challenge"]
    r = bob.post("/api/find-me", json={"challenge": ch, "frames": [durl(B_)] * 3}); assert r.status_code == 400
    # different people in frames rejected
    ch = bob.post("/api/consent").get_json()["challenge"]
    r = bob.post("/api/find-me", json={"challenge": ch, "frames": [durl(B_), durl(A_), durl(B_)]}); assert r.status_code == 400
    # wrong role
    assert alice.post("/upload", data={"file": (io.BytesIO(jpg(A_)), "a.jpg"), "consent": "yes"}, content_type="multipart/form-data").status_code == 403
    # csrf
    assert alice.c.post("/api/consent").status_code == 400
    # delete my data -> opt-out: future upload of Alice's face creates no row
    assert alice.post("/api/delete-my-face-data").get_json()["ok"]
    r = ph.post("/upload", data={"file": (io.BytesIO(jpg(noisy(A_))), "x.jpg"), "consent": "yes"}, content_type="multipart/form-data")
    assert r.get_json()["faces"] == 0
    assert "/media/" not in alice.c.get("/dashboard").get_data(as_text=True).split("<h2>My photos</h2>")[1].split("results")[1]

def test_video_upload():
    import subprocess
    ph = C(); ph.signup("vid", "photographer")
    d = tempfile.mkdtemp(); cv2.imwrite(d + "/f.png", person(9))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-loop", "1", "-i", d + "/f.png", "-t", "2", "-r", "10",
                    "-pix_fmt", "yuv420p", d + "/v.mp4"], check=True)
    r = ph.post("/upload", data={"file": (open(d + "/v.mp4", "rb"), "v.mp4"), "consent": "yes"}, content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["faces"] >= 2
