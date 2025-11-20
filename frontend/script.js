const API_BASE = "http://127.0.0.1:8000";

const analyzeBtn = document.getElementById("analyzeBtn");
const spectrumFileInput = document.getElementById("spectrumFile");
const analysisResultDiv = document.getElementById("analysisResult");

const chatBtn = document.getElementById("chatBtn");
const chatInput = document.getElementById("chatInput");
const chatResultDiv = document.getElementById("chatResult");
const useRagCheckbox = document.getElementById("useRag");

function preset(text) {
  chatInput.value = text;
}

analyzeBtn.addEventListener("click", async () => {
  const file = spectrumFileInput.files[0];
  if (!file) {
    alert("Please choose a CSV file first.");
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const resp = await fetch(`${API_BASE}/api/analyze-spectrum`, {
      method: "POST",
      body: formData,
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || "Request failed");
    }
    const data = await resp.json();
    renderAnalysisResult(data);
  } catch (err) {
    analysisResultDiv.style.display = "block";
    analysisResultDiv.innerHTML = `<strong>Error:</strong> ${err.message}`;
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Spectrum";
  }
});

function renderAnalysisResult(data) {
  const peaksText =
    data.peaks && data.peaks.length
      ? data.peaks
          .map(
            (p, idx) =>
              `${idx + 1}. E ≈ ${p.peak_energy.toFixed(
                3
              )}, I_norm ≈ ${p.peak_intensity.toFixed(2)}`
          )
          .join("\n")
      : "No clear peaks detected.";

  analysisResultDiv.style.display = "block";
  analysisResultDiv.innerHTML = `
    <h3>Analysis Summary</h3>
    <p><strong>Points:</strong> ${data.num_points}</p>
    <p><strong>Energy range:</strong> ${data.min_energy.toFixed(
      3
    )} – ${data.max_energy.toFixed(3)}</p>
    <p><strong>Max raw intensity:</strong> ${data.max_intensity.toFixed(3)}</p>
    <p><strong>Detected peaks:</strong></p>
    <pre>${peaksText}</pre>

    <p><strong>LLM summary:</strong></p>
    <pre>${data.llm_summary}</pre>

    <div class="cot-block">
        <details>
        <summary style="color:#888; cursor:pointer;">Show reasoning</summary>
        <ul style="color:#888; font-size:12px; margin-top:5px;">
            ${data.llm_cot.map(step => `<li>${step}</li>`).join("")}
        </ul>
        </details>
    </div>    

    <p><strong>Intensity vs. Energy (normalized):</strong></p>
    <canvas id="spectrumPlot" width="600" height="220"></canvas>
  `;
    if (data.curve && data.curve.length) {
        drawSpectrumPlot(data.curve);
    }
}

function drawSpectrumPlot(curve) {
  const canvas = document.getElementById("spectrumPlot");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx || !curve.length) return;

  // Separate paddings
  const paddingX = 60;   // 左右留 40
  const paddingY = 40;   // 上下留 60（你要求的更大 padding）
  const w = canvas.width;
  const h = canvas.height;

  // Extract energy & intensity
  const energies = curve.map((p) => p.energy);
  const intensities = curve.map((p) => p.intensity);

  const minE = Math.min(...energies);
  const maxE = Math.max(...energies);
  const minI = Math.min(...intensities);
  const maxI = Math.max(...intensities);

  const eRange = maxE - minE || 1;
  const iRange = maxI - minI || 1;

  ctx.clearRect(0, 0, w, h);
  ctx.font = "12px sans-serif";
  ctx.fillStyle = "#000";

  // =====================================================================================
  // 1. Draw axes
  // =====================================================================================

  ctx.strokeStyle = "#000";
  ctx.lineWidth = 1.2;

  // X-axis
  ctx.beginPath();
  ctx.moveTo(paddingX, h - paddingY);
  ctx.lineTo(w - paddingX, h - paddingY);
  ctx.stroke();

  // Y-axis
  ctx.beginPath();
  ctx.moveTo(paddingX, h - paddingY);
  ctx.lineTo(paddingX, paddingY);
  ctx.stroke();

  // =====================================================================================
  // 2. X-axis ticks
  // =====================================================================================

  const xTicks = 5;
  for (let i = 0; i <= xTicks; i++) {
    const xVal = minE + (i / xTicks) * eRange;
    const x = paddingX + ((xVal - minE) / eRange) * (w - 2 * paddingX);

    ctx.beginPath();
    ctx.moveTo(x, h - paddingY);
    ctx.lineTo(x, h - paddingY + 5);
    ctx.stroke();

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(xVal.toFixed(2), x, h - paddingY + 7);
  }

  // X label
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText("Energy (arb. units, eV)", w / 2, h - 5);

  // =====================================================================================
  // 3. Y-axis ticks
  // =====================================================================================

  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const yVal = minI + (i / yTicks) * iRange;
    const y = h - paddingY - ((yVal - minI) / iRange) * (h - 2 * paddingY);

    // tick
    ctx.beginPath();
    ctx.moveTo(paddingX - 5, y);
    ctx.lineTo(paddingX, y);
    ctx.stroke();

    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(yVal.toFixed(2), paddingX - 7, y);
  }

  // Y label
  ctx.save();
  ctx.translate(15, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("Normalized Intensity", 0, 0);
  ctx.restore();

  // =====================================================================================
  // 4. Draw curve
  // =====================================================================================

  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 2;
  ctx.beginPath();

  curve.forEach((p, idx) => {
    const x =
      paddingX + ((p.energy - minE) / eRange) * (w - 2 * paddingX);
    const y =
      h - paddingY - ((p.intensity - minI) / iRange) * (h - 2 * paddingY);

    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.stroke();
}



chatBtn.addEventListener("click", async () => {
  const message = chatInput.value.trim();
  if (!message) return;

  chatBtn.disabled = true;
  chatBtn.textContent = "Thinking...";

  try {
    const resp = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        use_rag: useRagCheckbox.checked,
      }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || "Request failed");
    }
    const data = await resp.json();
    renderChatResult(data);
  } catch (err) {
    chatResultDiv.style.display = "block";
    chatResultDiv.innerHTML = `<strong>Error:</strong> ${err.message}`;
  } finally {
    chatBtn.disabled = false;
    chatBtn.textContent = "Send";
  }
});

function renderChatResult(data) {
  const { answer, sources, cot } = data;

  chatResultDiv.style.display = "block";

  let sourcesHTML = "";
  if (sources && sources.length > 0) {
    sourcesHTML = "<h4>RAG Sources</h4>";
    sources.forEach((src, idx) => {
      sourcesHTML += `
        <div style="margin-bottom:10px; padding-left:10px; border-left:3px solid #2563eb;">
          <p style="margin:0;">
            <strong>Source ${idx + 1}</strong>
            &nbsp;(similarity = ${src.similarity.toFixed(3)})
          </p>
          <small>${src.text}</small>
        </div>
      `;
    });
  }

  // CoT block
  let cotHTML = "";
  if (cot && cot.length > 0) {
    cotHTML = `
      <div class="cot-block" style="margin-top:10px;">
        <details>
          <summary style="color:#888; cursor:pointer;">Show reasoning (CoT)</summary>
          <ul style="color:#888; font-size:12px; margin-top:5px;">
            ${cot.map(step => `<li>${step}</li>`).join("")}
          </ul>
        </details>
      </div>
    `;
  }

  chatResultDiv.innerHTML = `
    <h3>Assistant</h3>
    <pre>${answer}</pre>
    ${cotHTML}
    ${sourcesHTML}
  `;
}


