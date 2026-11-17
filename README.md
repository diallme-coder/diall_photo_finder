# Diall Photo Finder

Photographers upload photos/videos. Every face is indexed under a random anonymous tag (`anon-xxxxxxxx`) — no names.
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

## Privacy design
- Search only with a **live camera selfie** (server challenge + 3 frames + movement check); uploaded photos can't be used.
  This basic liveness check is a baseline — add a vendor liveness/anti-spoof service before public launch.
- Selfies are never stored. Results are visible only to the matched person; a tag already claimed by someone else reveals nothing.
- Photographers must confirm rights/consent on every upload. Members give explicit biometric consent before the camera opens.
- "Delete my face data" removes the tag + embeddings and blocks re-indexing of that face in future uploads.
- Search is rate limited (10/hour/user). CSRF protection, auth-checked media, random filenames.
- Biometric laws (Illinois BIPA, GDPR, NYC rules, etc.) may require written policies, retention limits and notices — get legal review before launch.
