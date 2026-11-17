import base64, io, json, os, subprocess, sys, tempfile, threading
from test_flow import C, A, person, noisy, durl, jpg, pose, YAW, selfie, gray_img, cv2, np
import faces

ROOT = os.path.join(os.path.dirname(__file__), "..")


def run_py(code, env=None):
    e = {k: v for k, v in os.environ.items() if k not in ("SECRET_KEY", "DEV", "DATA_DIR")}
    e.update({"FACE_ENGINE": "mock", "DATA_DIR": tempfile.mkdtemp()}); e.update(env or {})
    return subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {ROOT!r})\n" + code],
                          capture_output=True, text=True, env=e, cwd=tempfile.mkdtemp())


def member(name):
    m = C(); m.signup(name, "member"); return m


def challenge(m):
    return m.post("/api/consent").get_json()


def frames_for(ch, imgs_or_img, yaws=None):
    yaws = yaws or [YAW[p] for p in ch["poses"]]
    imgs = imgs_or_img if isinstance(imgs_or_img, list) else [imgs_or_img] * 3
    return [durl(pose(noisy(im), y)) for im, y in zip(imgs, yaws)]


def upload(ph, img, name="x.jpg", data=None):
    return ph.post("/upload", data={"file": (io.BytesIO(data or jpg(img)), name), "consent": "yes"},
                   content_type="multipart/form-data")


# ---------------------------------------------------------------- H1 liveness
def test_poses_are_random_server_side_permutations():
    m = member("liv1"); orders = set()
    for _ in range(25):
        ch = challenge(m); assert sorted(ch["poses"]) == ["left", "right", "straight"]; orders.add(tuple(ch["poses"]))
    assert len(orders) > 1


def test_static_photo_rejected_whatever_the_pose():
    ph = C(); ph.signup("pro_h1", "photographer"); A_ = person(31); upload(ph, A_)
    m = member("liv2")
    for y in (0.0, 0.5, -0.5):                       # same photo, same yaw in all 3 noisy frames
        ch = challenge(m)
        r = m.post("/api/find-me", json={"challenge": ch["challenge"], "frames": frames_for(ch, A_, [y] * 3)})
        assert r.status_code == 400


def test_frames_in_wrong_order_rejected_and_right_order_accepted():
    m = member("liv3"); A_ = person(32)
    ch = challenge(m)
    wrong = [YAW[p] for p in ch["poses"]]; wrong = wrong[1:] + wrong[:1]
    r = m.post("/api/find-me", json={"challenge": ch["challenge"], "frames": frames_for(ch, A_, wrong)})
    assert r.status_code == 400
    assert selfie(m, A_).status_code == 200


def test_yaw_from_landmarks():
    # YuNet row: x,y,w,h, re(x,y), le(x,y), nose(x,y), rmouth(x,y), lmouth(x,y), score
    def det(nose_x): return [0, 0, 100, 100, 30, 40, 70, 40, nose_x, 60, 35, 80, 65, 80, 0.9]
    assert abs(faces.yaw_from_landmarks(det(50))) < 1e-6
    assert faces.yaw_from_landmarks(det(60)) == 0.25 and faces.yaw_from_landmarks(det(40)) == -0.25


# ---------------------------------------------------------------- H2 / M4 / M5 claims
def test_dispute_logged_when_best_tag_claimed_by_someone_else():
    ph = C(); ph.signup("pro_h2", "photographer"); A_ = person(41); upload(ph, A_)
    owner, eve = member("own_h2"), member("eve_h2")
    assert selfie(owner, A_).get_json()["matched"]
    assert selfie(eve, A_).get_json() == {"ok": True, "matched": False, "photos": []}
    with A.app.app_context():
        rows = A.db().execute("SELECT d.tag, d.claimant_id, d.ts, t.user_id FROM claim_disputes d JOIN tags t ON t.tag=d.tag").fetchall()
        eve_id = A.db().execute("SELECT id FROM users WHERE username='eve_h2'").fetchone()["id"]
    assert any(r["claimant_id"] == eve_id and r["user_id"] != eve_id and r["ts"] for r in rows)


def _vec(seed):
    v = np.random.default_rng(seed).uniform(-60, 60, 64); return v - v.mean()


def _orth(v, seed, norm):
    e = np.random.default_rng(seed).uniform(-1, 1, 64); e -= e.mean(); e -= v * (e @ v) / (v @ v)
    return e * norm / np.linalg.norm(e)


def test_claim_threshold_stricter_than_index_threshold():
    ph = C(); ph.signup("pro_m5", "photographer"); v = _vec(51)
    upload(ph, None, "a.png", cv2.imencode(".png", gray_img(v))[1].tobytes())
    m = member("m5claim"); probe = v + _orth(v, 52, 0.43 * np.linalg.norm(v))   # cos ~0.92: > 0.9 index, < 0.95 claim
    assert faces.cosine(v, probe) > 0.9
    assert selfie(m, gray_img(probe)).get_json()["matched"] is False


def test_all_frame_pairs_must_be_same_person():
    m = member("m5b"); v = _vec(53); d = _orth(v, 54, 0.28 * np.linalg.norm(v))
    ch = challenge(m)   # frame0 close to both others, but frames 1 and 2 are cos~0.855 apart
    r = m.post("/api/find-me", json={"challenge": ch["challenge"], "frames":
               frames_for(ch, [gray_img(v), gray_img(v + d), gray_img(v - d)])})
    assert r.status_code == 400


def test_poisoned_tag_with_close_centroid_but_far_rows_is_not_claimable():
    ph = C(); ph.signup("pro_m4", "photographer"); upload(ph, person(61))
    v = _vec(62); nv = np.linalg.norm(v)
    with A.app.app_context():
        d = A.db(); pid = d.execute("SELECT id FROM photos ORDER BY id DESC LIMIT 1").fetchone()["id"]
        d.execute("INSERT INTO tags(tag,created) VALUES('anon-poison',1)")
        for k in range(4):   # each row cos~0.857 to v, centroid cos~0.958 to v
            d.execute("INSERT INTO face_rows(photo_id,tag,emb) VALUES(?,?,?)",
                      (pid, "anon-poison", A.blob(v + _orth(v, 100 + k, 0.6 * nv))))
        d.commit()
    m = member("m4claim")
    assert selfie(m, gray_img(v)).get_json()["matched"] is False


# ---------------------------------------------------------------- M1 rate limits
def test_search_slot_is_atomic():
    with A.app.app_context():
        pass
    results = []
    def go(): results.append(A.take_search_slot(987654))
    ts = [threading.Thread(target=go) for _ in range(30)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert results.count(True) == A.SEARCH_LIMIT_PER_HOUR and results.count(False) == 20


def test_ip_rate_limit_login_register_findme():
    A.IP_LIMITS["login"] = (3, 900); A.IP_LIMITS["register"] = (2, 3600); A.IP_LIMITS["find-me"] = (2, 3600)
    c = C()
    codes = [c.post("/login", data={"username": "nobody", "password": "x"}).status_code for _ in range(4)]
    assert codes == [401, 401, 401, 429]
    c2 = C()
    codes = [c2.post("/register", data={"username": f"rl{i}x", "password": "password123", "role": "member"}).status_code
             for i in range(3)]
    assert codes[-1] == 429 and codes[0] != 429
    m = C(); A._ip_hits.clear(); m.signup("rlm", "member"); A._ip_hits.clear()
    codes = [m.post("/api/find-me", json={"challenge": "x", "frames": []}).status_code for _ in range(3)]
    assert codes[-1] == 429


# ---------------------------------------------------------------- M2 image limits
def _findme(m, frames, ch=None):
    ch = ch or challenge(m)
    return m.post("/api/find-me", json={"challenge": ch["challenge"], "frames": frames})


def test_bad_frames_are_400_not_500():
    m = member("img1"); good = durl(person(1)); jp = jpg(person(1))
    bmp = base64.b64encode(cv2.imencode(".bmp", person(1))[1].tobytes()).decode()
    cases = [[1, 2, 3], [None] * 3, [{"a": 1}] * 3,
             ["data:text/html;base64," + base64.b64encode(jp).decode()] * 3,
             ["data:image/jpeg;base64,@@@notbase64@@@"] * 3,
             ["data:image/jpeg;base64," + bmp] * 3,                       # wrong magic bytes
             ["data:image/gif;base64," + base64.b64encode(jp).decode()] * 3,
             ["data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xff" + b"0" * (3 * 1024 * 1024)).decode(), good, good]]
    for fr in cases:
        assert _findme(m, fr).status_code == 400, fr[0] if isinstance(fr[0], str) and len(fr[0]) < 80 else fr[0].__class__


def test_findme_body_limit_10mb():
    m = member("img2"); ch = challenge(m)
    big = "data:image/jpeg;base64," + "A" * (11 * 1024 * 1024)
    r = m.post("/api/find-me", data=json.dumps({"challenge": ch["challenge"], "frames": [big]}), content_type="application/json")
    assert r.status_code == 413


def test_upload_magic_bytes_checked():
    ph = C(); ph.signup("pro_m2", "photographer")
    r = upload(ph, None, "x.jpg", cv2.imencode(".bmp", person(3))[1].tobytes())   # decodable, but not jpeg/png/webp
    assert r.status_code == 400
    assert upload(ph, person(3)).status_code == 200


def test_opencv_pixel_limit_set_before_cv2_import():
    code = ("import app, cv2, numpy as np\n"
            "assert __import__('os').environ['OPENCV_IO_MAX_IMAGE_PIXELS']=='40000000'\n"
            "png=cv2.imencode('.png', np.zeros((7000,7000,3),np.uint8), [cv2.IMWRITE_PNG_COMPRESSION,1])[1]\n"
            "try:\n    cv2.imdecode(png, cv2.IMREAD_COLOR); print('decoded')\nexcept cv2.error:\n    print('ok')")
    r = run_py(code); assert "ok" in r.stdout, r.stderr[-500:]


def test_video_not_supported_any_more():
    assert not hasattr(A, "VID_EXT")
    assert "VideoCapture" not in open(os.path.join(ROOT, "app.py")).read()


# ---------------------------------------------------------------- L1 single-use challenges
def test_challenge_single_use_even_when_attempt_fails():
    m = member("ch1"); A_ = person(71); ch = challenge(m)
    bad = m.post("/api/find-me", json={"challenge": ch["challenge"], "frames": frames_for(ch, A_, [0, 0, 0])})
    assert bad.status_code == 400
    good = m.post("/api/find-me", json={"challenge": ch["challenge"], "frames": frames_for(ch, A_)})
    assert good.status_code == 400 and "expired" in good.get_json()["error"].lower()
    with A.app.app_context():
        assert A.db().execute("SELECT COUNT(*) c FROM challenges WHERE id=?", (ch["challenge"],)).fetchone()["c"] == 0


def test_challenge_not_usable_by_other_user():
    a, b = member("ch2a"), member("ch2b"); ch = challenge(a); A_ = person(72)
    assert b.post("/api/consent").status_code == 200
    r = b.post("/api/find-me", json={"challenge": ch["challenge"], "frames": frames_for(ch, A_)})
    assert r.status_code == 400


# ---------------------------------------------------------------- L2 headers / cookies
def test_security_headers_and_no_inline_script():
    c = C(); r = c.c.get("/"); h = r.headers
    csp = h["Content-Security-Policy"]
    for part in ("default-src 'self'", "img-src 'self' blob:", "media-src 'self' blob:", "script-src 'self'",
                 "object-src 'none'", "frame-ancestors 'none'", "base-uri 'none'"):
        assert part in csp
    assert h["Referrer-Policy"] == "no-referrer" and "max-age" in h["Strict-Transport-Security"]
    html = r.get_data(as_text=True)
    assert '<meta name="csrf"' in html and "<script>" not in html and "window.CSRF" not in html
    for js in ("find.js", "upload.js"):
        src = open(os.path.join(ROOT, "static", js)).read()
        assert "window.CSRF" not in src and 'meta[name=csrf]' in src.replace('"', "").replace("'", "") or "meta[name=csrf]" in src


def test_cookie_secure_unless_dev_and_session_lifetime():
    r = run_py("import app; print(app.app.config['SESSION_COOKIE_SECURE'], app.app.config['PERMANENT_SESSION_LIFETIME'].total_seconds())")
    assert r.stdout.split() == ["True", "43200.0"], r.stderr[-400:]
    r = run_py("import app; print(app.app.config['SESSION_COOKIE_SECURE'])", {"DEV": "1"})
    assert r.stdout.split() == ["False"]
    c = C(); c.signup("perm1", "member")
    r = c.c.post("/login", data={"username": "perm1", "password": "password123", "csrf": c.tok()})
    assert "Expires=" in r.headers["Set-Cookie"]


# ---------------------------------------------------------------- L3 secret key
def test_secret_key_file_0600_and_skipped_with_env():
    r = run_py("import app, os, stat; p=os.path.join(app.DATA,'secret.key'); print(oct(stat.S_IMODE(os.stat(p).st_mode)))")
    assert r.stdout.strip() == "0o600", r.stderr[-400:]
    r = run_py("import app, os; print(os.path.exists(os.path.join(app.DATA,'secret.key')))", {"SECRET_KEY": "abc"})
    assert r.stdout.strip() == "False", r.stderr[-400:]


# ---------------------------------------------------------------- L4 generic upload error
def test_upload_failure_is_generic_and_leaves_no_files():
    ph = C(); ph.signup("pro_l4", "photographer")
    class Boom:
        name = "boom"; threshold = 0.9; claim_threshold = 0.95
        def faces(self, img): raise RuntimeError("secret-internal-detail /srv/x")
    old = A._engine; A._engine = Boom()
    try:
        before = set(os.listdir(A.UPLOADS)); r = upload(ph, person(81))
    finally:
        A._engine = old
    assert r.status_code == 400 and "secret-internal-detail" not in r.get_data(as_text=True)
    assert set(os.listdir(A.UPLOADS)) == before


# ---------------------------------------------------------------- L7 usernames / login timing
def test_username_ascii_only():
    for bad in ("usér123", "ＡＢＣ１２３", "ab", "a" * 31, "bad name", "bad-name"):
        r = C().post("/register", data={"username": bad, "password": "password123", "role": "member"})
        assert r.status_code == 400, bad
    assert C().post("/register", data={"username": "ok_name_1", "password": "password123", "role": "member"}).status_code == 302


def test_login_hashes_even_when_user_missing():
    calls = []; real = A.check_password_hash
    A.check_password_hash = lambda h, p: (calls.append(1), real(h, p))[1]
    try:
        r = C().post("/login", data={"username": "nosuchuser", "password": "whatever1"})
    finally:
        A.check_password_hash = real
    assert r.status_code == 401 and calls == [1]


# ---------------------------------------------------------------- docs honesty / scope
def test_readme_and_home_are_honest_and_photo_only():
    readme = open(os.path.join(ROOT, "README.md")).read().lower()
    home = open(os.path.join(ROOT, "templates", "home.html")).read().lower()
    assert "replay" in readme and "opt-out" in readme and "unencrypted" in readme
    assert "videos" not in readme and "videos" not in home and "video/*" not in open(os.path.join(ROOT, "templates", "photographer.html")).read()
    for t in ("photographer.html", "member.html", "register.html"):
        txt = open(os.path.join(ROOT, "templates", t)).read().lower().replace('<video id="cam"', "")
        assert "video" not in txt.replace("</video>", ""), t
    assert 'accept="image/*"' in open(os.path.join(ROOT, "templates", "photographer.html")).read()


def test_oversized_pixel_frame_is_400():
    m = member("img3")
    png = base64.b64encode(cv2.imencode(".png", np.zeros((7000, 7000, 3), np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()).decode()
    r = _findme(m, ["data:image/png;base64," + png] * 1 + [durl(person(1))] * 2)
    assert r.status_code == 400
