const CSRF = document.querySelector("meta[name=csrf]").content;
const consent = document.getElementById("consent"), start = document.getElementById("start"),
  cam = document.getElementById("cam"), prompt_ = document.getElementById("prompt"), msg = document.getElementById("msg");
consent.onchange = () => start.disabled = !consent.checked;
const post = (url, body) => fetch(url, {method: "POST", headers: {"X-CSRF": CSRF, "Content-Type": "application/json"}, body: JSON.stringify(body || {})});
const sleep = ms => new Promise(r => setTimeout(r, ms));
start.onclick = async () => {
  start.disabled = true; msg.textContent = "";
  try {
    const c = await (await post("/api/consent")).json();
    const stream = await navigator.mediaDevices.getUserMedia({video: {facingMode: "user"}});
    cam.srcObject = stream; cam.hidden = false; await sleep(800);
    const frames = [], cv = document.createElement("canvas");
    for (const p of c.prompts) {
      prompt_.textContent = p; await sleep(1600);
      cv.width = cam.videoWidth; cv.height = cam.videoHeight;
      cv.getContext("2d").drawImage(cam, 0, 0); frames.push(cv.toDataURL("image/jpeg", 0.9));
    }
    stream.getTracks().forEach(t => t.stop()); cam.hidden = true; prompt_.textContent = "Searching…";
    const r = await post("/api/find-me", {challenge: c.challenge, frames});
    const j = await r.json(); prompt_.textContent = "";
    if (!r.ok) msg.textContent = j.error;
    else if (j.matched) location.reload();
    else msg.textContent = "No photos found yet. Check back after more uploads.";
  } catch (e) { msg.textContent = "Camera error: " + e.message; }
  start.disabled = !consent.checked;
};
const wipe = document.getElementById("wipe");
if (wipe) wipe.onclick = async () => { if (confirm("Delete your face data and unlink your photos?")) { await post("/api/delete-my-face-data"); location.reload(); } };
