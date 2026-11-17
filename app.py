"""Diall Photo Finder: photographers upload; faces are indexed under anonymous tags;
members find their own photos with a live selfie. See README for privacy design."""
import base64, os, secrets, sqlite3, time
import cv2
import numpy as np
from flask import (Flask, abort, g, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import faces

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(BASE, "data"))
UPLOADS = os.path.join(DATA, "uploads")
DB_PATH = os.path.join(DATA, "app.db")
IMG_EXT = {"jpg", "jpeg", "png", "webp"}
VID_EXT = {"mp4", "mov", "webm", "m4v"}
SEARCH_LIMIT_PER_HOUR = 10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024
os.makedirs(UPLOADS, exist_ok=True)
_key_file = os.path.join(DATA, "secret.key")
if not os.path.exists(_key_file):
    with open(_key_file, "w") as f:
        f.write(secrets.token_hex(32))
    os.chmod(_key_file, 0o600)
app.secret_key = os.environ.get("SECRET_KEY") or open(_key_file).read()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = faces.get_engine()
    return _engine


# ---------------------------------------------------------------- database
SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, pw_hash TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role IN('photographer','member')),
  bio TEXT DEFAULT '', created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS tags(          -- one per distinct face; random anonymous name
  tag TEXT PRIMARY KEY, user_id INTEGER REFERENCES users(id), created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS photos(
  id INTEGER PRIMARY KEY, owner_id INTEGER NOT NULL REFERENCES users(id),
  filename TEXT NOT NULL, thumb TEXT, kind TEXT NOT NULL, title TEXT DEFAULT '', created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS face_rows(
  id INTEGER PRIMARY KEY, photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  tag TEXT NOT NULL REFERENCES tags(tag), box TEXT, ts REAL, emb BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS consents(
  user_id INTEGER NOT NULL, kind TEXT NOT NULL, ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS search_log(user_id INTEGER NOT NULL, ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS opt_outs(     -- faces whose owners asked not to be indexed
  id INTEGER PRIMARY KEY, emb BLOB NOT NULL, ts REAL NOT NULL);
"""


def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys=ON")
    return g.db


@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d:
        d.close()


def init_db():
    c = sqlite3.connect(DB_PATH)
    c.executescript(SCHEMA)
    c.commit()
    c.close()


init_db()


# ---------------------------------------------------------------- helpers
def me():
    if "uid" not in session:
        return None
    return db().execute("SELECT * FROM users WHERE id=?", (session["uid"],)).fetchone()


def login_required(role=None):
    def deco(fn):
        def wrapper(*a, **kw):
            u = me()
            if not u:
                return redirect(url_for("login")) if request.method == "GET" else abort(401)
            if role and u["role"] != role:
                abort(403)
            return fn(u, *a, **kw)
        wrapper.__name__ = fn.__name__
        return wrapper
    return deco


@app.before_request
def csrf():
    if request.method == "POST":
        tok = request.headers.get("X-CSRF") or request.form.get("csrf")
        if not tok or tok != session.get("csrf"):
            abort(400, "bad csrf token")


@app.context_processor
def inject():
    session.setdefault("csrf", secrets.token_hex(16))
    return {"csrf": session["csrf"], "user": me()}


@app.after_request
def headers(r):
    r.headers["X-Content-Type-Options"] = "nosniff"
    r.headers["Cache-Control"] = "no-store"
    r.headers["Permissions-Policy"] = "camera=(self)"
    return r


def ext(name):
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def new_tag():
    return "anon-" + secrets.token_hex(4)


def blob(v):
    return np.asarray(v, dtype=np.float32).tobytes()


def unblob(b):
    return np.frombuffer(b, dtype=np.float32)


def tag_centroids():
    """tag -> mean embedding."""
    rows = db().execute("SELECT tag, emb FROM face_rows").fetchall()
    acc = {}
    for r in rows:
        acc.setdefault(r["tag"], []).append(unblob(r["emb"]))
    return {t: np.mean(v, axis=0) for t, v in acc.items()}


def best_match(emb, centroids):
    best, score = None, -1.0
    for t, c in centroids.items():
        s = faces.cosine(emb, c)
        if s > score:
            best, score = t, s
    return best, score


def is_opted_out(emb):
    th = engine().threshold
    for r in db().execute("SELECT emb FROM opt_outs"):
        if faces.cosine(emb, unblob(r["emb"])) >= th:
            return True
    return False


def index_faces(photo_id, found, centroids, ts=None):
    """Store each detected face under an existing or brand-new anonymous tag."""
    th, n = engine().threshold, 0
    for box, emb in found:
        if is_opted_out(emb):
            continue
        tag, score = best_match(emb, centroids)
        if tag is None or score < th:
            tag = new_tag()
            db().execute("INSERT INTO tags(tag,created) VALUES(?,?)", (tag, time.time()))
            centroids[tag] = emb
        db().execute("INSERT INTO face_rows(photo_id,tag,box,ts,emb) VALUES(?,?,?,?,?)",
                     (photo_id, tag, ",".join(map(str, box)), ts, blob(emb)))
        n += 1
    return n


def decode_data_url(s):
    raw = base64.b64decode(s.split(",", 1)[-1], validate=False)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        abort(400, "bad image")
    return img


# ---------------------------------------------------------------- pages / auth
@app.get("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        u = request.form.get("username", "").strip().lower()
        p = request.form.get("password", "")
        role = request.form.get("role")
        if not (3 <= len(u) <= 30 and u.replace("_", "").isalnum()) or len(p) < 8 or role not in ("photographer", "member"):
            return render_template("register.html", error="Username 3-30 letters/numbers, password 8+ chars."), 400
        try:
            cur = db().execute("INSERT INTO users(username,pw_hash,role,created) VALUES(?,?,?,?)",
                               (u, generate_password_hash(p), role, time.time()))
            db().commit()
        except sqlite3.IntegrityError:
            return render_template("register.html", error="Username taken."), 400
        session.clear()
        session["uid"] = cur.lastrowid
        return redirect(url_for("dashboard"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        row = db().execute("SELECT * FROM users WHERE username=?",
                           (request.form.get("username", "").strip().lower(),)).fetchone()
        if row and check_password_hash(row["pw_hash"], request.form.get("password", "")):
            session.clear()
            session["uid"] = row["id"]
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Wrong username or password."), 401
    return render_template("login.html")


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.get("/dashboard")
@login_required()
def dashboard(u):
    if u["role"] == "photographer":
        photos = db().execute(
            "SELECT p.*, (SELECT COUNT(DISTINCT tag) FROM face_rows WHERE photo_id=p.id) AS people "
            "FROM photos p WHERE owner_id=? ORDER BY created DESC", (u["id"],)).fetchall()
        return render_template("photographer.html", photos=photos)
    tags = db().execute("SELECT tag FROM tags WHERE user_id=?", (u["id"],)).fetchall()
    found = db().execute(
        "SELECT DISTINCT p.*, o.username AS photographer FROM photos p "
        "JOIN face_rows f ON f.photo_id=p.id JOIN tags t ON t.tag=f.tag "
        "JOIN users o ON o.id=p.owner_id WHERE t.user_id=? ORDER BY p.created DESC", (u["id"],)).fetchall()
    return render_template("member.html", photos=found, linked=bool(tags))


@app.get("/p/<username>")
def public_profile(username):
    owner = db().execute("SELECT * FROM users WHERE username=? AND role='photographer'", (username,)).fetchone()
    if not owner:
        abort(404)
    n = db().execute("SELECT COUNT(*) c FROM photos WHERE owner_id=?", (owner["id"],)).fetchone()["c"]
    return render_template("profile.html", owner=owner, count=n)


# ---------------------------------------------------------------- uploads (photographers)
@app.post("/upload")
@login_required("photographer")
def upload(u):
    if request.form.get("consent") != "yes":
        return jsonify(error="You must confirm you have the right to upload these and index faces."), 400
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="no file"), 400
    e = ext(f.filename)
    if e not in IMG_EXT | VID_EXT:
        return jsonify(error="unsupported type"), 400
    kind = "image" if e in IMG_EXT else "video"
    name = f"{secrets.token_hex(12)}.{e}"
    path = os.path.join(UPLOADS, name)
    f.save(path)
    try:
        found_total, thumb = 0, None
        cur = db().execute("INSERT INTO photos(owner_id,filename,kind,title,created) VALUES(?,?,?,?,?)",
                           (u["id"], name, kind, request.form.get("title", "")[:120], time.time()))
        pid = cur.lastrowid
        cents = tag_centroids()
        if kind == "image":
            img = cv2.imread(path)
            if img is None:
                raise ValueError("not a valid image")
            found_total = index_faces(pid, engine().faces(img), cents)
        else:
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            step, i = int(fps), 0  # sample ~1 frame per second
            ok, frame = cap.read()
            if not ok:
                raise ValueError("not a valid video")
            tname = name.rsplit(".", 1)[0] + "_t.jpg"
            cv2.imwrite(os.path.join(UPLOADS, tname), frame)
            thumb = tname
            while ok:
                if i % step == 0:
                    found_total += index_faces(pid, engine().faces(frame), cents, ts=i / fps)
                ok, frame = cap.read()
                i += 1
            cap.release()
        db().execute("UPDATE photos SET thumb=? WHERE id=?", (thumb, pid))
        db().commit()
    except Exception as ex:  # clean up on failure
        db().rollback()
        os.remove(path)
        return jsonify(error=str(ex)), 400
    return jsonify(ok=True, id=pid, faces=found_total)


@app.post("/photo/<int:pid>/delete")
@login_required("photographer")
def delete_photo(u, pid):
    row = db().execute("SELECT * FROM photos WHERE id=? AND owner_id=?", (pid, u["id"])).fetchone()
    if not row:
        abort(404)
    for fn in (row["filename"], row["thumb"]):
        if fn and os.path.exists(os.path.join(UPLOADS, fn)):
            os.remove(os.path.join(UPLOADS, fn))
    db().execute("DELETE FROM photos WHERE id=?", (pid,))
    db().commit()
    return redirect(url_for("dashboard"))


@app.get("/media/<int:pid>")
@login_required()
def media(u, pid):
    """Owner, or a member whose own face appears in it. Nobody else."""
    row = db().execute("SELECT * FROM photos WHERE id=?", (pid,)).fetchone()
    if not row:
        abort(404)
    allowed = row["owner_id"] == u["id"] or db().execute(
        "SELECT 1 FROM face_rows f JOIN tags t ON t.tag=f.tag WHERE f.photo_id=? AND t.user_id=? LIMIT 1",
        (pid, u["id"])).fetchone()
    if not allowed:
        abort(404)
    use_thumb = request.args.get("thumb") and row["thumb"]
    return send_from_directory(UPLOADS, row["thumb"] if use_thumb else row["filename"])


# ---------------------------------------------------------------- find me (members)
@app.post("/api/consent")
@login_required("member")
def give_consent(u):
    db().execute("INSERT INTO consents(user_id,kind,ts) VALUES(?,?,?)", (u["id"], "biometric", time.time()))
    db().commit()
    session["challenge"] = secrets.token_hex(8)
    session["challenge_ts"] = time.time()
    return jsonify(ok=True, challenge=session["challenge"], prompts=["Look straight", "Turn head left", "Turn head right"])


def has_consent(uid):
    return bool(db().execute("SELECT 1 FROM consents WHERE user_id=? AND kind='biometric'", (uid,)).fetchone())


@app.post("/api/find-me")
@login_required("member")
def find_me(u):
    """Body: {challenge, frames:[dataURL x3]} captured live by the camera.
    Basic liveness: fresh server challenge, 3 frames each with exactly one face, same person, frames not identical.
    This is a baseline; use a vendor liveness check before production."""
    if not has_consent(u["id"]):
        return jsonify(error="Consent required"), 403
    data = request.get_json(silent=True) or {}
    if data.get("challenge") != session.get("challenge") or time.time() - session.get("challenge_ts", 0) > 300:
        return jsonify(error="Session expired, start again"), 400
    now = time.time()
    recent = db().execute("SELECT COUNT(*) c FROM search_log WHERE user_id=? AND ts>?", (u["id"], now - 3600)).fetchone()["c"]
    if recent >= SEARCH_LIMIT_PER_HOUR:
        return jsonify(error="Too many searches, try later"), 429
    db().execute("INSERT INTO search_log VALUES(?,?)", (u["id"], now))
    db().commit()
    frames = data.get("frames") or []
    if len(frames) != 3:
        return jsonify(error="3 live frames required"), 400
    imgs = [decode_data_url(fr) for fr in frames]
    embs = []
    for im in imgs:
        fs = engine().faces(im)
        if len(fs) != 1:
            return jsonify(error="Each frame must show exactly one face"), 400
        embs.append(fs[0][1])
    th = engine().threshold
    if min(faces.cosine(embs[0], e) for e in embs[1:]) < th:
        return jsonify(error="Frames are not the same person"), 400
    diffs = [float(np.mean(cv2.absdiff(imgs[0], cv2.resize(im, imgs[0].shape[1::-1])))) for im in imgs[1:]]
    if max(diffs) < 1.0:
        return jsonify(error="Liveness check failed (no movement detected)"), 400
    probe = np.mean(embs, axis=0)
    cents = tag_centroids()
    tag, score = best_match(probe, cents)
    session.pop("challenge", None)
    mine = {r["tag"] for r in db().execute("SELECT tag FROM tags WHERE user_id=?", (u["id"],))}
    if tag is None or score < th:
        return jsonify(ok=True, matched=False, photos=[])
    owners = {t: o for t, o in db().execute("SELECT tag, user_id FROM tags").fetchall()}
    if owners[tag] not in (None, u["id"]):
        return jsonify(ok=True, matched=False, photos=[])  # best match already claimed by someone else: never reveal
    # One person can be split across several tags (poses, lighting): link every close, unclaimed tag.
    close = [t for t, c in cents.items() if faces.cosine(probe, c) >= th and owners[t] in (None, u["id"])]
    if mine and not (mine & set(close)):
        return jsonify(ok=True, matched=False, photos=[])  # a different face than the one already linked
    for t in close:
        db().execute("UPDATE tags SET user_id=? WHERE tag=?", (u["id"], t))
    db().commit()
    q = ",".join("?" * len(close))
    n = db().execute(f"SELECT COUNT(DISTINCT photo_id) c FROM face_rows WHERE tag IN ({q})", close).fetchone()["c"]
    return jsonify(ok=True, matched=True, count=n)


@app.post("/api/delete-my-face-data")
@login_required("member")
def delete_face_data(u):
    """Unlink the tag, remember its embedding as an opt-out so future uploads skip it, drop stored rows."""
    cents = tag_centroids()
    for t in db().execute("SELECT tag FROM tags WHERE user_id=?", (u["id"],)).fetchall():
        if t["tag"] in cents:
            db().execute("INSERT INTO opt_outs(emb,ts) VALUES(?,?)", (blob(cents[t["tag"]]), time.time()))
        db().execute("DELETE FROM face_rows WHERE tag=?", (t["tag"],))
        db().execute("DELETE FROM tags WHERE tag=?", (t["tag"],))
    db().execute("DELETE FROM consents WHERE user_id=?", (u["id"],))
    db().commit()
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
