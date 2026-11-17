"""Second-round review fixes: concurrency, claims, delete flows, consents, indexing, performance."""
import io, os, sqlite3, threading, time
import pytest
from test_flow import C, A, person, noisy, durl, jpg, pose, YAW, selfie, gray_img, cv2, np
from test_security import member, challenge, frames_for, upload, _vec, _orth
import faces

PNG = lambda img: cv2.imencode(".png", img)[1].tobytes()


def upload_png(ph, vec_or_img):
    img = vec_or_img if vec_or_img.ndim == 3 else gray_img(vec_or_img)
    r = ph.post("/upload", data={"file": (io.BytesIO(PNG(img)), "x.png"), "consent": "yes"}, content_type="multipart/form-data")
    assert r.status_code == 200, r.get_data(as_text=True)
    return r.get_json()["id"]


def at_angle(v, u, deg):
    a = np.radians(deg); return np.cos(a) * v + np.sin(a) * u


def plane(seed=1):
    v = _vec(seed); return v, _orth(v, seed + 500, np.linalg.norm(v))


def user_id(name):
    with A.app.app_context():
        return A.db().execute("SELECT id FROM users WHERE username=?", (name,)).fetchone()["id"]


def owner_of_tags():
    with A.app.app_context():
        return [(r["tag"], r["user_id"]) for r in A.db().execute("SELECT tag,user_id FROM tags")]


class SlowMock(faces.MockEngine):
    def __init__(self, delay): self.delay = delay
    def faces(self, img):
        time.sleep(self.delay); return super().faces(img)


class Multi(faces.MockEngine):
    """TEST ONLY: reports `n` near-identical faces in one photo."""
    def __init__(self, n): self.n = n
    def faces(self, img):
        box, e, yaw = super().faces(img)[0]
        return [([i * 20, 0, 10, 10], e + i * 1e-3, yaw) for i in range(self.n)]


# ---------------------------------------------------------------- H1
def test_slow_detection_does_not_block_other_writers(monkeypatch):
    monkeypatch.setattr(A, "_engine", SlowMock(2.0))
    ph = C(); ph.signup("h1pro", "photographer")
    res = {}
    t = threading.Thread(target=lambda: res.update(r=upload(ph, person(5))))
    t.start(); time.sleep(0.5)
    t0 = time.time(); r = C().signup("h1late", "member"); dt = time.time() - t0
    assert t.is_alive(), "upload finished too early to prove overlap"
    assert r.status_code == 302 and dt < 1.0, (r.status_code, dt)
    t.join(); assert res["r"].status_code == 200


def test_concurrent_uploads_of_same_person_share_one_tag(monkeypatch):
    monkeypatch.setattr(A, "_engine", SlowMock(0.5))
    p1, p2 = C(), C(); p1.signup("h1a", "photographer"); p2.signup("h1b", "photographer")
    img = person(6); out = []
    ts = [threading.Thread(target=lambda p=p: out.append(upload(p, noisy(img)).status_code)) for p in (p1, p2)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert out == [200, 200] and len(owner_of_tags()) == 1


def test_connection_uses_wal_and_long_busy_timeout():
    with A.app.app_context():
        d = A.db()
        assert d.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert d.execute("PRAGMA busy_timeout").fetchone()[0] >= 30000


# ---------------------------------------------------------------- H2
def test_concurrent_claims_have_exactly_one_winner(monkeypatch):
    ph = C(); ph.signup("h2pro", "photographer"); A_ = person(42); upload(ph, A_)
    m1, m2 = member("h2one"), member("h2two")
    barrier, local, real = threading.Barrier(2, timeout=10), threading.local(), A.claimable
    def synced(*a, **k):        # both requests have read the (unclaimed) owners before either writes
        if not getattr(local, "done", False):
            local.done = True
            try: barrier.wait()
            except threading.BrokenBarrierError: pass
        return real(*a, **k)
    monkeypatch.setattr(A, "claimable", synced)
    out = {}
    ts = [threading.Thread(target=lambda n=n, m=m: out.update({n: selfie(m, A_).get_json()})) for n, m in (("one", m1), ("two", m2))]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert sorted(o["matched"] for o in out.values()) == [False, True], out
    winner, loser = ("h2one", "h2two") if out["one"]["matched"] else ("h2two", "h2one")
    assert owner_of_tags()[0][1] == user_id(winner)
    with A.app.app_context():
        d = A.db().execute("SELECT claimant_id FROM claim_disputes").fetchall()
    assert [r["claimant_id"] for r in d] == [user_id(loser)]


# ---------------------------------------------------------------- M4
def test_delete_face_data_removes_unclaimed_lookalike_tags_only():
    ph = C(); ph.signup("m4pro", "photographer"); v, u = plane(11)
    pid = upload_png(ph, v)
    m, other = member("m4me"), member("m4other")
    assert selfie(m, gray_img(v)).get_json()["matched"]
    with A.app.app_context():
        d = A.db(); oid = user_id("m4other")
        for name, emb, uid in (("anon-like", at_angle(v, u, 21.6), None),      # cos .93: same person, unclaimed
                               ("anon-far", at_angle(v, u, 60), None),          # cos .5: someone else
                               ("anon-others", at_angle(v, u, 21.6), oid)):     # lookalike owned by another member
            d.execute("INSERT INTO tags(tag,user_id,created) VALUES(?,?,1)", (name, uid))
            d.execute("INSERT INTO face_rows(photo_id,tag,emb) VALUES(?,?,?)", (pid, name, A.blob(emb)))
        d.commit()
    assert m.post("/api/delete-my-face-data").get_json()["ok"]
    left = {t for t, _ in owner_of_tags()}
    assert left == {"anon-far", "anon-others"}, left
    with A.app.app_context():
        assert {r["tag"] for r in A.db().execute("SELECT tag FROM face_rows")} == {"anon-far", "anon-others"}
        assert A.db().execute("SELECT COUNT(*) c FROM opt_outs").fetchone()["c"] == 1


# ---------------------------------------------------------------- M7 / L8
def test_sibling_tag_does_not_hide_members_own_match():
    ph = C(); ph.signup("m7pro", "photographer"); v, u = plane(21)
    upload_png(ph, v); upload_png(ph, at_angle(v, u, 28))          # two separate tags (cos .88 < .9)
    sib, me_ = member("m7sib"), member("m7me")
    assert selfie(sib, gray_img(at_angle(v, u, 28))).get_json()["matched"]   # sibling owns the second tag
    assert selfie(me_, gray_img(v)).get_json()["matched"]
    j = selfie(me_, gray_img(at_angle(v, u, 16))).get_json()   # closer to the sibling's tag than to mine, both pass claim
    assert j["matched"] and j["count"] == 1
    mine = [t for t, o in owner_of_tags() if o == user_id("m7me")]
    assert len(mine) == 1


def test_changed_appearance_links_new_tag_and_count_covers_all_tags():
    ph = C(); ph.signup("m7pro2", "photographer"); v, u = plane(22)
    upload_png(ph, v)
    me_ = member("m7change")
    assert selfie(me_, gray_img(v)).get_json()["count"] == 1
    upload_png(ph, at_angle(v, u, 30))                              # new, separate tag
    probe = at_angle(v, u, 20)    # cos .94 to own tag (>= index .9, < claim .95), .985 to the new tag
    assert faces.cosine(v, probe) < 0.95
    j = selfie(me_, gray_img(probe)).get_json()
    assert j["matched"] and j["count"] == 2
    page = me_.c.get("/dashboard").get_data(as_text=True)
    assert page.count("/media/") == j["count"]
    assert len([1 for _, o in owner_of_tags() if o == user_id("m7change")]) == 2


def test_member_with_linked_face_cannot_claim_an_unrelated_face():
    ph = C(); ph.signup("m7pro3", "photographer"); v = _vec(23); w = _vec(24)
    upload_png(ph, v); upload_png(ph, w)
    me_ = member("m7one")
    assert selfie(me_, gray_img(v)).get_json()["matched"]
    assert selfie(me_, gray_img(w)).get_json()["matched"] is False
    assert sorted(o is None for _, o in owner_of_tags()) == [False, True]


# ---------------------------------------------------------------- L3
def test_delete_photo_removes_file_rows_orphan_tags_and_access(monkeypatch):
    ph = C(); ph.signup("l3pro", "photographer"); A1, B1 = person(81), person(82)
    p1 = upload(ph, noisy(A1)).get_json()["id"]; p2 = upload(ph, noisy(A1)).get_json()["id"]; p3 = upload(ph, B1).get_json()["id"]
    m = member("l3m"); assert selfie(m, A1).get_json()["count"] == 2
    with A.app.app_context():
        fn = {r["id"]: r["filename"] for r in A.db().execute("SELECT id,filename FROM photos")}
    assert m.c.get(f"/media/{p1}").status_code == 200
    seen = []; real = os.remove
    def checking_remove(path, *a, **k):   # DB must already be committed when the file goes
        c = sqlite3.connect(A.DB_PATH); seen.append(c.execute("SELECT COUNT(*) FROM photos WHERE id=?", (p1,)).fetchone()[0]); c.close()
        return real(path, *a, **k)
    monkeypatch.setattr(A.os, "remove", checking_remove)
    assert ph.post(f"/photo/{p1}/delete").status_code == 302
    assert seen == [0]
    assert not os.path.exists(os.path.join(A.UPLOADS, fn[p1]))
    with A.app.app_context():
        assert A.db().execute("SELECT COUNT(*) c FROM face_rows WHERE photo_id=?", (p1,)).fetchone()["c"] == 0
    assert m.c.get(f"/media/{p1}").status_code == 404 and ph.c.get(f"/media/{p1}").status_code == 404
    assert m.c.get(f"/media/{p2}").status_code == 200
    # unclaimed tag with no photos left disappears; claimed tag survives
    assert ph.post(f"/photo/{p3}/delete").status_code == 302
    assert ph.post(f"/photo/{p2}/delete").status_code == 302
    tags = owner_of_tags()
    assert len(tags) == 1 and tags[0][1] == user_id("l3m")
    assert not any(os.path.exists(os.path.join(A.UPLOADS, f)) for f in fn.values())


def test_cannot_delete_someone_elses_photo():
    a, b = C(), C(); a.signup("l3a", "photographer"); b.signup("l3b", "photographer")
    pid = upload(a, person(83)).get_json()["id"]
    assert b.post(f"/photo/{pid}/delete").status_code == 404
    assert a.c.get(f"/media/{pid}").status_code == 200


# ---------------------------------------------------------------- L4
def test_consent_is_recorded_once_per_user_and_kind():
    m = member("l4m")
    for _ in range(4): assert m.post("/api/consent").status_code == 200
    with A.app.app_context():
        assert A.db().execute("SELECT COUNT(*) c FROM consents").fetchone()["c"] == 1
        with pytest.raises(sqlite3.IntegrityError):
            A.db().execute("INSERT INTO consents(user_id,kind,ts) VALUES(?,?,1)", (user_id("l4m"), "biometric"))


def test_init_db_dedupes_old_consents(tmp_path, monkeypatch):
    p = str(tmp_path / "old.db"); c = sqlite3.connect(p)
    c.execute("CREATE TABLE consents(user_id INTEGER NOT NULL, kind TEXT NOT NULL, ts REAL NOT NULL)")
    c.executemany("INSERT INTO consents VALUES(1,'biometric',?)", [(1,), (2,), (3,)]); c.commit(); c.close()
    monkeypatch.setattr(A, "DB_PATH", p); A.init_db()
    c = sqlite3.connect(p); assert c.execute("SELECT COUNT(*) FROM consents").fetchone()[0] == 1


# ---------------------------------------------------------------- L7
def test_two_faces_of_one_photo_never_share_a_tag(monkeypatch):
    monkeypatch.setattr(A, "_engine", Multi(3))
    ph = C(); ph.signup("l7pro", "photographer"); img = person(91)
    r = upload(ph, img); assert r.get_json()["faces"] == 3
    with A.app.app_context():
        rows = A.db().execute("SELECT photo_id, tag FROM face_rows").fetchall()
    assert len({r["tag"] for r in rows}) == 3
    upload(ph, noisy(img))     # a later photo: its faces also land on distinct tags
    with A.app.app_context():
        rows = A.db().execute("SELECT photo_id, tag FROM face_rows ORDER BY id DESC LIMIT 3").fetchall()
    assert len({r["tag"] for r in rows}) == 3 and len(owner_of_tags()) == 3


def test_centroids_keep_a_running_mean():
    rng = np.random.default_rng(3); es = [rng.normal(size=16).astype(np.float32) for _ in range(5)]
    c = A.Centroids()
    c.add("t1", es[0]); c.add("t2", es[1]); c.add("t1", es[2]); c.add("t1", es[3]); c.add("t2", es[4])
    assert np.allclose(c.centroid("t1"), np.mean([es[0], es[2], es[3]], axis=0), atol=1e-5)
    assert np.allclose(c.centroid("t2"), np.mean([es[1], es[4]], axis=0), atol=1e-5)
    assert c.best(es[1])[0] in ("t1", "t2")


def test_load_centroids_matches_plain_means():
    ph = C(); ph.signup("l7pro2", "photographer")
    for s in (1, 2, 3): upload(ph, person(s)); upload(ph, noisy(person(s)))
    with A.app.app_context():
        d = A.db(); cents = A.load_centroids(d); ref = {}
        for r in d.execute("SELECT tag, emb FROM face_rows"): ref.setdefault(r["tag"], []).append(A.unblob(r["emb"]))
    assert set(cents.tags) == set(ref) and len(ref) == 3
    for t, v in ref.items(): assert np.allclose(cents.centroid(t), np.mean(v, axis=0), atol=1e-4)


# ---------------------------------------------------------------- M1
def test_best_match_agrees_with_plain_loop_and_respects_mask():
    rng = np.random.default_rng(5); n, dim = 200, 32
    means = rng.normal(size=(n, dim)).astype(np.float32); tags = [f"t{i}" for i in range(n)]
    c = A.Centroids(tags, means * 3, [3] * n)
    for _ in range(50):
        p = rng.normal(size=dim).astype(np.float32)
        best, score = None, -1.0
        for t, m in zip(tags, means):
            s = faces.cosine(p, m)
            if s > score: best, score = t, s
        got, gs = A.best_match(p, c)
        assert got == best and abs(gs - score) < 1e-4
        mask = np.array([i % 2 == 0 for i in range(n)])
        got2, _ = A.best_match(p, c, mask)
        assert int(got2[1:]) % 2 == 0
    assert A.best_match(p, A.Centroids()) == (None, -1.0)


def test_best_match_is_vectorised_rough_timing():
    rng = np.random.default_rng(6); n, dim = 20000, 128
    c = A.Centroids([f"t{i}" for i in range(n)], rng.normal(size=(n, dim)).astype(np.float32), [1] * n)
    ps = rng.normal(size=(20, dim)).astype(np.float32)
    t0 = time.time(); [A.best_match(p, c) for p in ps]
    assert time.time() - t0 < 1.0     # the per-tag Python loop needs several seconds for this


def test_opt_outs_loaded_once_per_upload(monkeypatch):
    monkeypatch.setattr(A, "_engine", Multi(4)); calls = []; real = A.load_opt_outs
    monkeypatch.setattr(A, "load_opt_outs", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    ph = C(); ph.signup("m1pro", "photographer")
    assert upload(ph, person(95)).get_json()["faces"] == 4 and calls == [1]
