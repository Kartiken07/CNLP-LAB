const $ = (id) => document.getElementById(id);
const text = $("text"), btn = $("classify"), err = $("error"), res = $("result");

text.addEventListener("input", () => { $("count").textContent = text.value.length.toLocaleString() + " characters"; });
$("clear").addEventListener("click", () => { text.value = ""; text.dispatchEvent(new Event("input")); res.hidden = true; err.hidden = true; text.focus(); });

async function classify() {
  err.hidden = true;
  if (!text.value.trim()) { showError("Please paste some article text first."); return; }
  btn.disabled = true; btn.textContent = "Classifying…";
  try {
    const r = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: text.value }) });
    const data = await r.json();
    if (!r.ok) { res.hidden = true; showError(data.error || "Something went wrong."); return; }
    render(data);
  } catch (e) { res.hidden = true; showError("Could not reach the server."); }
  finally { btn.disabled = false; btn.textContent = "Classify"; }
}

function render(d) {
  $("category").textContent = d.category;
  $("confidence").textContent = (d.confidence * 100).toFixed(1) + "%";
  const bars = $("bars"); bars.textContent = "";
  Object.entries(d.probabilities).sort((a, b) => b[1] - a[1]).forEach(([name, p], i) => {
    const row = document.createElement("div"); row.className = "bar" + (i === 0 ? " top" : "");
    const label = document.createElement("span"); label.textContent = name;
    const track = document.createElement("div"); track.className = "track";
    const fill = document.createElement("div"); fill.className = "fill"; fill.style.width = (p * 100).toFixed(1) + "%";
    track.appendChild(fill);
    const pct = document.createElement("span"); pct.textContent = (p * 100).toFixed(1) + "%";
    row.append(label, track, pct); bars.appendChild(row);
  });
  res.hidden = false;
}
function showError(m) { err.textContent = m; err.hidden = false; }

btn.addEventListener("click", classify);
text.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") classify(); });
