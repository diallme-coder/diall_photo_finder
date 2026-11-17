"""Diall Photo Finder: photographers upload; faces are indexed under anonymous tags;
members find their own photos with a live selfie. See README for privacy design."""
import base64, binascii, contextlib, itertools, logging, os, re, secrets, sqlite3, threading, time
from datetime import timedelta
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "40000000")  # must be set before cv2 is imported
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
SEARCH_LIMIT_PER_HOUR = 10
IP_LIMITS = {"register": (20, 3600), "login": (20, 900), "find-me": (30, 3600)}  # (hits, window seconds) per IP
FRAME_MAX_BYTES = 3 * 1024 * 1024
FIND_ME_MAX_BODY = 10 * 1024 * 1024
CHALLENGE_TTL = 300
POSES = ("straight", "left", "right")
POSE_TEXT = {"straight": "Look straight at the camera", "left": "Turn your head left", "right": "Turn your head right"}
YAW_STRAIGHT_MAX, YAW_TURN_MIN = 0.12, 0.2   # yaw = (nose_x - eye_mid_x)/eye_dist; "left" = nose toward image right
TAG_ROW_FRACTION = 0.8                       # share of a tag's face rows a probe must match to claim it
USERNAME_RE = re.compile(r"[a-z0-9_]{3,30}", re.ASCII)

def dev_mode(argv=None, environ=None):
    """Plain-HTTP local development: `python app.py --dev` (any OS) or DEV=1."""
    import sys
    return "--dev" in (sys.argv if argv is None else argv) or (os.environ if environ is None else environ).get("DEV") == "1"


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
os.makedirs(UPLOADS, exist_ok=True)
_key_file = os.path.join(DATA, "secret.key")
if os.environ.get("SECRET_KEY"):
    app.secret_key = os.environ["SECRET_KEY"]
else:
    try:
        _fd = os.open(_key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(_fd, "w") as f:
            f.write(secrets.token_hex(32))
    except FileExistsError:
        pass
    with open(_key_file) as f:
        app.secret_key = f.read().strip()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  SESSION_COOKIE_SECURE=not dev_mode(),
                  PERMANENT_SESSION_LIFETIME=timedelta(hours=12))

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
CREATE TABLE IF NOT EXISTS claim_disputes(  -- someone matched a tag that another member already owns
  tag TEXT NOT NULL, claimant_id INTEGER NOT NULL, ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS challenges(      -- single-use liveness challenges (server-side pose order)
  id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, poses TEXT NOT NULL, ts REAL NOT NULL);
"""


def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, timeout=30)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys=ON")
    return g.db


@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d:
        d.close()


@contextlib.contextmanager
def write_txn():
    """One short write transaction: takes the write lock up front, commits on success, rolls back on error."""
    d = db()
    d.execute("BEGIN IMMEDIATE")
    try:
        yield d
        d.commit()
    except BaseException:
        d.rollback()
        raise


def init_db():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")      # readers never block the writer (and vice versa)
    c.executescript(SCHEMA)
    c.execute("DELETE FROM consents WHERE rowid NOT IN (SELECT MIN(rowid) FROM consents GROUP BY user_id, kind)")
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS consents_user_kind ON consents(user_id, kind)")
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
def body_limits():
    if request.path == "/api/find-me":
        request.max_content_length = FIND_ME_MAX_BODY


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
    r.headers["Content-Security-Policy"] = ("default-src 'self'; img-src 'self' blob:; media-src 'self' blob:; "
                                            "script-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
    r.headers["Referrer-Policy"] = "no-referrer"
    r.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return r


_ip_hits, _ip_lock = {}, threading.Lock()


def ip_limited(bucket):
    """True if this client IP exceeded the bucket's limit (in-memory, per process).
    Uses request.remote_addr; behind a reverse proxy configure werkzeug ProxyFix."""
    limit, window = IP_LIMITS[bucket]
    now, key = time.time(), (bucket, request.remote_addr)
    with _ip_lock:
        if len(_ip_hits) > 10000:
            for k in [k for k, v in _ip_hits.items() if not v or v[-1] < now - 3600]:
                del _ip_hits[k]
        hits = [t for t in _ip_hits.get(key, []) if t > now - window]
        if len(hits) >= limit:
            _ip_hits[key] = hits
            return True
        hits.append(now)
        _ip_hits[key] = hits
        return False


def take_search_slot(uid):
    """Atomically check the per-user hourly limit and record a search."""
    c = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    try:
        c.execute("BEGIN IMMEDIATE")
        now = time.time()
        n = c.execute("SELECT COUNT(*) FROM search_log WHERE user_id=? AND ts>?", (uid, now - 3600)).fetchone()[0]
        if n >= SEARCH_LIMIT_PER_HOUR:
            c.execute("ROLLBACK")
            return False
        c.execute("INSERT INTO search_log(user_id,ts) VALUES(?,?)", (uid, now))
        c.execute("COMMIT")
        return True
    except Exception:
        if c.in_transaction:
            c.execute("ROLLBACK")
        raise
    finally:
        c.close()


def ext(name):
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def new_tag():
    return "anon-" + secrets.token_hex(4)


def blob(v):
    return np.asarray(v, dtype=np.float32).tobytes()


def unblob(b):
    return np.frombuffer(b, dtype=np.float32)


class Centroids:
    """Per-tag running sum and count of face embeddings, plus the unit-length mean for fast matching.
    Matching is one matrix-vector product instead of a Python loop over tags."""

    def __init__(self, tags=(), sums=None, counts=()):
        self.tags = list(tags)
        self.index = {t: i for i, t in enumerate(self.tags)}
        self.sums = None if sums is None or not len(self.tags) else np.asarray(sums, dtype=np.float32)
        self.counts = list(counts)
        self.unit = None if self.sums is None else self.sums / (np.linalg.norm(self.sums, axis=1, keepdims=True) + 1e-9)

    def centroid(self, tag):
        i = self.index[tag]
        return self.sums[i] / self.counts[i]

    def scores(self, emb):
        if self.unit is None:
            return np.empty(0, dtype=np.float32)
        return self.unit @ (np.asarray(emb, dtype=np.float32) / (np.linalg.norm(emb) + 1e-9))

    def best(self, emb, allowed=None):
        s = self.scores(emb)
        if allowed is not None:
            s = np.where(allowed, s, -np.inf)
        if not len(s):
            return None, -1.0
        i = int(np.argmax(s))
        return (self.tags[i], float(s[i])) if np.isfinite(s[i]) else (None, -1.0)

    def add(self, tag, emb):
        emb = np.asarray(emb, dtype=np.float32)
        i = self.index.get(tag)
        if i is None:
            self.index[tag] = len(self.tags)
            self.tags.append(tag)
            self.counts.append(1)
            self.sums = emb[None, :].copy() if self.sums is None else np.vstack([self.sums, emb])
            self.unit = (self.sums / (np.linalg.norm(self.sums, axis=1, keepdims=True) + 1e-9)) if self.unit is None \
                else np.vstack([self.unit, emb / (np.linalg.norm(emb) + 1e-9)])
        else:
            self.sums[i] += emb
            self.counts[i] += 1
            self.unit[i] = self.sums[i] / (np.linalg.norm(self.sums[i]) + 1e-9)


def load_centroids(conn):
    """All tag centroids from one query and one numpy pass."""
    rows = conn.execute("SELECT tag, emb FROM face_rows").fetchall()
    if not rows:
        return Centroids()
    mat = np.frombuffer(b"".join(r["emb"] for r in rows), dtype=np.float32).reshape(len(rows), -1)
    uniq, inv = np.unique(np.array([r["tag"] for r in rows]), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.concatenate(([0], np.flatnonzero(np.diff(inv[order])) + 1))
    return Centroids(uniq.tolist(), np.add.reduceat(mat[order], starts, axis=0), np.bincount(inv).tolist())


def tag_rows_of(conn, tag):
    return [unblob(r["emb"]) for r in conn.execute("SELECT emb FROM face_rows WHERE tag=?", (tag,))]


def claimable(probe, rows, ct):
    """Probe must be within the claim threshold of the tag centroid AND of >=80% of the tag's own face rows
    (so a few planted look-alike rows cannot drag a centroid onto someone's face)."""
    if faces.cosine(probe, np.mean(rows, axis=0)) < ct:
        return False
    close = sum(faces.cosine(probe, r) >= ct for r in rows)
    return close / len(rows) >= TAG_ROW_FRACTION


def best_match(emb, centroids, allowed=None):
    return centroids.best(emb, allowed)


def load_opt_outs(conn):
    """Unit-length opt-out embeddings as one matrix (None if there are none). Load once per request."""
    rows = conn.execute("SELECT emb FROM opt_outs").fetchall()
    if not rows:
        return None
    m = np.frombuffer(b"".join(r["emb"] for r in rows), dtype=np.float32).reshape(len(rows), -1)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def is_opted_out(emb, opt_matrix):
    if opt_matrix is None:
        return False
    return bool(np.max(opt_matrix @ (np.asarray(emb, dtype=np.float32) / (np.linalg.norm(emb) + 1e-9))) >= engine().threshold)


def index_faces(d, photo_id, found, centroids, opt_matrix, ts=None):
    """Store each detected face under an existing or brand-new anonymous tag. Two faces of one photo never
    share a tag. Centroids are updated as we go. Must run inside the caller's write transaction."""
    th, n, used = engine().threshold, 0, set()
    for box, emb, _yaw in found:
        if is_opted_out(emb, opt_matrix):
            continue
        allowed = np.array([t not in used for t in centroids.tags]) if centroids.tags else None
        tag, score = best_match(emb, centroids, allowed)
        if tag is None or score < th:
            tag = new_tag()
            d.execute("INSERT INTO tags(tag,created) VALUES(?,?)", (tag, time.time()))
        centroids.add(tag, emb)
        used.add(tag)
        d.execute("INSERT INTO face_rows(photo_id,tag,box,ts,emb) VALUES(?,?,?,?,?)",
                  (photo_id, tag, ",".join(map(str, box)), ts, blob(emb)))
        n += 1
    return n


_DATA_URL = re.compile(r"data:image/(jpeg|png);base64,")
_MAGIC = {"jpeg": b"\xff\xd8\xff", "png": b"\x89PNG\r\n\x1a\n"}


def decode_data_url(s):
    m = _DATA_URL.match(s) if isinstance(s, str) else None
    if not m or len(s) - m.end() > FRAME_MAX_BYTES * 4 // 3 + 8:
        abort(400, "bad image")
    try:
        raw = base64.b64decode(s[m.end():], validate=True)
    except (binascii.Error, ValueError):
        abort(400, "bad image")
    if len(raw) > FRAME_MAX_BYTES or not raw.startswith(_MAGIC[m.group(1)]):
        abort(400, "bad image")
    try:
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:  # e.g. over OPENCV_IO_MAX_IMAGE_PIXELS
        img = None
    if img is None:
        abort(400, "bad image")
    return img


def image_magic_ok(head):
    return head.startswith(_MAGIC["jpeg"]) or head.startswith(_MAGIC["png"]) or (head[:4] == b"RIFF" and head[8:12] == b"WEBP")


DUMMY_HASH = generate_password_hash("not-a-real-password")


# ---------------------------------------------------------------- pages / auth
@app.get("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        if ip_limited("register"):
            return render_template("register.html", error="Too many attempts, try later."), 429
        u = request.form.get("username", "").strip().lower()
        p = request.form.get("password", "")
        role = request.form.get("role")
        if not USERNAME_RE.fullmatch(u) or len(p) < 8 or role not in ("photographer", "member"):
            return render_template("register.html", error="Username 3-30 characters: a-z, 0-9, underscore. Password 8+ chars."), 400
        try:
            cur = db().execute("INSERT INTO users(username,pw_hash,role,created) VALUES(?,?,?,?)",
                               (u, generate_password_hash(p), role, time.time()))
            db().commit()
        except sqlite3.IntegrityError:
            return render_template("register.html", error="Username taken."), 400
        session.clear()
        session.permanent = True
        session["uid"] = cur.lastrowid
        return redirect(url_for("dashboard"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if ip_limited("login"):
            return render_template("login.html", error="Too many attempts, try later."), 429
        row = db().execute("SELECT * FROM users WHERE username=?",
                           (request.form.get("username", "").strip().lower(),)).fetchone()
        pw = request.form.get("password", "")
        ok = check_password_hash(row["pw_hash"] if row else DUMMY_HASH, pw)  # hash even if user is missing
        if row and ok:
            session.clear()
            session.permanent = True
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
class UploadError(ValueError):
    """Safe-to-show message for the uploader."""


@app.post("/upload")
@login_required("photographer")
def upload(u):
    if request.form.get("consent") != "yes":
        return jsonify(error="You must confirm you have the right to upload these and index faces."), 400
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="no file"), 400
    e = ext(f.filename)
    if e not in IMG_EXT:
        return jsonify(error="unsupported type (photos only: jpg, png, webp)"), 400
    if not image_magic_ok(f.stream.read(16)):
        return jsonify(error="not a valid jpg, png or webp image"), 400
    f.stream.seek(0)
    name = f"{secrets.token_hex(12)}.{e}"
    path = os.path.join(UPLOADS, name)
    f.save(path)
    try:
        img = cv2.imread(path)
        if img is None:
            raise UploadError("not a valid image")
        found = engine().faces(img)            # heavy work happens before any DB write
        with write_txn() as d:                 # short transaction: reload state, tag, insert, commit
            opt = load_opt_outs(d)
            cents = load_centroids(d)
            cur = d.execute("INSERT INTO photos(owner_id,filename,kind,title,created) VALUES(?,?,?,?,?)",
                            (u["id"], name, "image", request.form.get("title", "")[:120], time.time()))
            pid = cur.lastrowid
            n = index_faces(d, pid, found, cents, opt)
    except Exception as ex:  # clean up on failure (write_txn already rolled back)
        if os.path.exists(path):
            os.remove(path)
        if isinstance(ex, UploadError):
            return jsonify(error=str(ex)), 400
        app.logger.exception("upload failed")
        return jsonify(error="Could not process this file."), 400
    return jsonify(ok=True, id=pid, faces=n)


@app.post("/photo/<int:pid>/delete")
@login_required("photographer")
def delete_photo(u, pid):
    row = db().execute("SELECT * FROM photos WHERE id=? AND owner_id=?", (pid, u["id"])).fetchone()
    if not row:
        abort(404)
    with write_txn() as d:       # database first; files only go once the commit succeeded
        d.execute("DELETE FROM photos WHERE id=?", (pid,))
        d.execute("DELETE FROM tags WHERE user_id IS NULL AND tag NOT IN (SELECT tag FROM face_rows)")
    for fn in (row["filename"], row["thumb"]):
        if fn and os.path.exists(os.path.join(UPLOADS, fn)):
            os.remove(os.path.join(UPLOADS, fn))
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
    return send_from_directory(UPLOADS, row["filename"])


# ---------------------------------------------------------------- find me (members)
@app.post("/api/consent")
@login_required("member")
def give_consent(u):
    """Record consent and issue a single-use liveness challenge with a random, server-side pose order."""
    now = time.time()
    order = list(POSES)
    secrets.SystemRandom().shuffle(order)
    cid = secrets.token_hex(16)
    with write_txn() as d:
        d.execute("INSERT OR IGNORE INTO consents(user_id,kind,ts) VALUES(?,?,?)", (u["id"], "biometric", now))
        d.execute("DELETE FROM challenges WHERE user_id=? OR ts<?", (u["id"], now - CHALLENGE_TTL))
        d.execute("INSERT INTO challenges(id,user_id,poses,ts) VALUES(?,?,?,?)", (cid, u["id"], ",".join(order), now))
    return jsonify(ok=True, challenge=cid, poses=order, prompts=[POSE_TEXT[p] for p in order])


def has_consent(uid):
    return bool(db().execute("SELECT 1 FROM consents WHERE user_id=? AND kind='biometric'", (uid,)).fetchone())


def pose_ok(pose, yaw):
    if pose == "straight":
        return abs(yaw) <= YAW_STRAIGHT_MAX
    return yaw >= YAW_TURN_MIN if pose == "left" else yaw <= -YAW_TURN_MIN


def consume_challenge(uid, cid):
    """Return the pose order for a valid challenge, deleting it (single use, even if the attempt later fails)."""
    if not isinstance(cid, str):
        return None
    d = db()
    row = d.execute("SELECT poses, ts FROM challenges WHERE id=? AND user_id=?", (cid, uid)).fetchone()
    d.execute("DELETE FROM challenges WHERE id=?", (cid,))
    d.commit()
    if not row or time.time() - row["ts"] > CHALLENGE_TTL:
        return None
    return row["poses"].split(",")


@app.post("/api/find-me")
@login_required("member")
def find_me(u):
    """Body: {challenge, frames:[dataURL x3]} captured live by the camera, in the server-chosen pose order.
    Baseline liveness: single-use server challenge with a random order of head poses, head yaw of each frame must
    match its prompt, exactly one face per frame, all frames the same person, frames not identical.
    It does NOT stop a replayed or pre-recorded video; use a vendor liveness check before production."""
    if ip_limited("find-me"):
        return jsonify(error="Too many requests, try later"), 429
    if not has_consent(u["id"]):
        return jsonify(error="Consent required"), 403
    data = request.get_json(silent=True)
    data = data if isinstance(data, dict) else {}
    poses = consume_challenge(u["id"], data.get("challenge"))
    if not poses:
        return jsonify(error="Session expired, start again"), 400
    frames = data.get("frames")
    if not isinstance(frames, list) or len(frames) != 3 or not all(isinstance(x, str) for x in frames):
        return jsonify(error="3 live frames required"), 400
    if not take_search_slot(u["id"]):
        return jsonify(error="Too many searches, try later"), 429
    imgs = [decode_data_url(fr) for fr in frames]
    embs = []
    for im, pose in zip(imgs, poses):
        fs = engine().faces(im)
        if len(fs) != 1:
            return jsonify(error="Each frame must show exactly one face"), 400
        if not pose_ok(pose, fs[0][2]):
            return jsonify(error="Liveness check failed (head pose did not match the prompts)"), 400
        embs.append(fs[0][1])
    th, ct = engine().threshold, engine().claim_threshold
    if min(faces.cosine(a, b) for a, b in itertools.combinations(embs, 2)) < th:
        return jsonify(error="Frames are not the same person"), 400
    diffs = [float(np.mean(cv2.absdiff(imgs[0], cv2.resize(im, imgs[0].shape[1::-1])))) for im in imgs[1:]]
    if max(diffs) < 1.0:
        return jsonify(error="Liveness check failed (no movement detected)"), 400
    probe = np.mean(embs, axis=0)
    d = db()
    cents = load_centroids(d)
    nothing = jsonify(ok=True, matched=False, photos=[])
    if not cents.tags:
        return nothing
    owners = {r["tag"]: r["user_id"] for r in d.execute("SELECT tag, user_id FROM tags")}
    scores = cents.scores(probe)
    passing = {}      # tag -> score, for every tag that passes the full claim test
    for i in np.flatnonzero(scores >= ct):
        t = cents.tags[i]
        if claimable(probe, tag_rows_of(d, t), ct):
            passing[t] = float(scores[i])
    mine = {t for t, o in owners.items() if o == u["id"]}
    allowed = {t for t in passing if owners.get(t) in (None, u["id"])}   # others' tags are filtered out first
    top = cents.tags[int(np.argmax(scores))]
    if top in passing and owners.get(top) not in (None, u["id"]) and not (allowed & mine):
        with write_txn() as w:   # best match overall is someone else's: record the dispute, never reveal
            w.execute("INSERT INTO claim_disputes(tag,claimant_id,ts) VALUES(?,?,?)", (top, u["id"], time.time()))
        return nothing
    if not allowed:
        return nothing
    if mine and not (allowed & mine):
        # Already linked but no own tag passes the claim test: accept only if the probe is still the same person as
        # one of the member's stored tags at the looser index threshold (appearance changed), else a different face.
        own = np.array([t in mine for t in cents.tags])
        if cents.best(probe, own)[1] < th:
            return nothing
    won, lost = [], []
    with write_txn() as w:       # re-read owners under the write lock; conditional UPDATE is the final arbiter
        now_owner = {r["tag"]: r["user_id"] for r in w.execute(
            "SELECT tag, user_id FROM tags WHERE tag IN (%s)" % ",".join("?" * len(allowed)), list(allowed))}
        for t in sorted(allowed, key=lambda t: -passing[t]):
            if now_owner.get(t) not in (None, u["id"]):
                lost.append(t)
                continue
            cur = w.execute("UPDATE tags SET user_id=? WHERE tag=? AND (user_id IS NULL OR user_id=?)", (u["id"], t, u["id"]))
            (won if cur.rowcount == 1 else lost).append(t)
        for t in lost:
            w.execute("INSERT INTO claim_disputes(tag,claimant_id,ts) VALUES(?,?,?)", (t, u["id"], time.time()))
    if not won:
        return nothing
    n = d.execute("SELECT COUNT(DISTINCT f.photo_id) c FROM face_rows f JOIN tags t ON t.tag=f.tag WHERE t.user_id=?",
                  (u["id"],)).fetchone()["c"]
    return jsonify(ok=True, matched=True, count=n)


@app.post("/api/delete-my-face-data")
@login_required("member")
def delete_face_data(u):
    """Unlink the member's tags, remember their embeddings as opt-outs so future uploads skip them, drop stored
    rows. Also drops every UNCLAIMED tag that looks like the same person (within the index threshold)."""
    with write_txn() as d:
        cents = load_centroids(d)
        owners = {r["tag"]: r["user_id"] for r in d.execute("SELECT tag, user_id FROM tags")}
        mine = [t for t, o in owners.items() if o == u["id"]]
        gone = set(mine)
        for t in mine:
            if t in cents.index:
                d.execute("INSERT INTO opt_outs(emb,ts) VALUES(?,?)", (blob(cents.centroid(t)), time.time()))
                s = cents.scores(cents.centroid(t))
                gone.update(cents.tags[i] for i in np.flatnonzero(s >= engine().threshold) if owners.get(cents.tags[i]) is None)
        for t in gone:
            d.execute("DELETE FROM face_rows WHERE tag=?", (t,))
            d.execute("DELETE FROM tags WHERE tag=?", (t,))
        d.execute("DELETE FROM consents WHERE user_id=?", (u["id"],))
    return jsonify(ok=True)


if __name__ == "__main__":
    if os.environ.get("FACE_ENGINE") != "mock" and not (os.path.exists(faces.YUNET) and os.path.exists(faces.SFACE)):
        raise SystemExit("Face models are missing. Run first:  python scripts/download_models.py")
    if dev_mode():
        print("DEV mode: plain-HTTP cookies. Open http://127.0.0.1:%s" % os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
