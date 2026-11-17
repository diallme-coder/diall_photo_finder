# Diall Photo Finder

Photographers upload photos (images only; video is not supported). Every face is indexed under a random anonymous tag (`anon-xxxxxxxx`) — no names.
A member signs up, consents, takes a live selfie (3 frames), and the app links the matching tag to their account and
shows **only** the photos they appear in.

## Run
```
pip install -r requirements.txt
python scripts/download_models.py    # OpenCV Zoo YuNet + SFace (Apache-2.0), needs internet once
python app.py                         # http://127.0.0.1:5000
python -m pytest tests                # uses FACE_ENGINE=mock (test-only stand-in, not real recognition)
```
Set `SECRET_KEY` and serve behind HTTPS (e.g. gunicorn + reverse proxy) in production; camera access requires HTTPS.

## Privacy design and honest limits
- Search only with a **live camera selfie**: a single-use server challenge assigns a random order of head poses
  (straight / left / right); the server estimates head yaw from each frame (YuNet landmarks) and rejects frames that do not
  match their prompt, so a still photo or noisy copies of one photo fail. This is a **baseline liveness check only**: it does
  **not** stop a replayed or pre-recorded video, or a face-swap/deepfake. Add a vendor liveness/anti-spoof service before public launch.
  Uploaded photos can't be used as the selfie.
- Selfies are never stored. Results are visible only to the matched person. A tag needs a stricter match to be claimed than to be
  indexed, and the probe must match most of the tag's face rows. If the best match is a tag already claimed by someone else, nothing
  is revealed and the attempt is logged in `claim_disputes`.
- Photographers must confirm rights/consent on every upload. Members give explicit biometric consent before the camera opens.
- "Delete my face data" removes the tag and its stored face rows and unlinks the photos, **but deliberately keeps an anonymous
  face template (an opt-out list) so that face is not indexed again in future uploads.** That template is itself biometric data;
  it is not deleted by this action.
- Face embeddings and opt-out templates are stored **unencrypted** in `data/app.db`. Protect the server and its backups; there is no
  retention limit or encryption at rest yet.
- Rate limits: 10 searches/hour/user (atomic), plus per-IP limits on register, login and find-me (in-memory, per process; use a
  reverse proxy with ProxyFix and real client IPs). There is no full brute-force lockout yet.
- Hardening: CSRF protection, auth-checked media, random filenames, strict CSP, HSTS, no-referrer, secure session cookies
  (set `DEV=1` only for plain-HTTP local development), 12 h sessions, image size/type/pixel limits.
- Biometric laws (Illinois BIPA, GDPR, NYC rules, etc.) may require written policies, retention limits and notices — get legal review before launch.
