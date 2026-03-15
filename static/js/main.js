// Drag & drop upload
const zone = document.getElementById("uploadZone");
if(zone){
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
  zone.addEventListener("drop", e => {
    e.preventDefault(); zone.classList.remove("drag");
    const file = e.dataTransfer.files[0];
    if(file){ document.getElementById("fileInput").files = e.dataTransfer.files; uploadFile(document.getElementById("fileInput")); }
  });
}

// Auto-dismiss alerts
document.querySelectorAll(".alert-dismissible").forEach(a => setTimeout(() => a && a.remove(), 5000));
