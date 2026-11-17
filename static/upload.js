const CSRF = document.querySelector("meta[name=csrf]").content;
document.getElementById("up").addEventListener("submit", async e => {
  e.preventDefault();
  const form = e.target, files = form.file.files, st = document.getElementById("status");
  let faces = 0;
  for (let i = 0; i < files.length; i++) {
    st.textContent = `Uploading ${i + 1}/${files.length}…`;
    const fd = new FormData();
    fd.append("file", files[i]); fd.append("title", form.title.value); fd.append("consent", "yes");
    const r = await fetch("/upload", {method: "POST", body: fd, headers: {"X-CSRF": CSRF}});
    const j = await r.json();
    if (!r.ok) { st.textContent = "Error: " + j.error; return; }
    faces += j.faces;
  }
  st.textContent = `Done. ${faces} faces indexed.`; setTimeout(() => location.reload(), 800);
});
