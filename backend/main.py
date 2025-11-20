from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import numpy as np
import json
import re

from openai import OpenAI

from .config import settings
from .models import (
    SpectrumAnalysisResult,
    SpectrumFeature,
    ChatRequest,
    ChatResponse,
    SpectrumCurvePoint,
)
from .spectroscopy import parse_spectrum_csv, normalize_intensity, detect_peaks
from .pinecone_rag import get_rag_store


app = FastAPI(title="X-ray Spectroscopy AI Agent Demo")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_openai_client() -> OpenAI:
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=settings.OPENAI_API_KEY)


@app.post("/api/analyze-spectrum", response_model=SpectrumAnalysisResult)
async def analyze_spectrum(
    file: UploadFile = File(...),
    client: OpenAI = Depends(get_openai_client),
):
    # 1) Parse CSV
    try:
        content = await file.read()
        energy, intensity = parse_spectrum_csv(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    if len(energy) == 0:
        raise HTTPException(status_code=400, detail="Empty spectrum")

    # 2) Normalize
    norm_intensity = normalize_intensity(intensity)

    # 2b) Build curve data for plotting (energy vs normalized intensity)
    curve = [
        SpectrumCurvePoint(
            energy=float(e),
            intensity=float(i),
        )
        for e, i in zip(energy, norm_intensity)
    ]

    # 3) Detect peaks
    peaks: List[SpectrumFeature] = detect_peaks(energy, norm_intensity)

    # 4) LLM summary
    peak_desc_lines = [
        f"- Peak at {p.peak_energy:.3f} eV with normalized intensity {p.peak_intensity:.2f}"
        for p in peaks
    ]
    peak_desc = "\n".join(peak_desc_lines) if peak_desc_lines else "No clear peaks detected."

    energies = [p.energy for p in curve]
    intensities = [p.intensity for p in curve]

    num_points = len(energies)
    min_energy = min(energies)
    max_energy = max(energies)
    max_intensity = max(intensities)

    prompt = f"""
You are an X-ray spectroscopy analysis expert.

Analyze the following spectral statistics and detected peaks.
Provide:
1. A short scientific explanation (4–7 sentences).
2. A short, condensed chain-of-thought (CoT) reasoning (3–6 steps maximum).

IMPORTANT:
- The *summary* may be long (up to 250 tokens).
- Keep the chain-of-thought SHORT and CONDENSED (3–6 steps, no long essays).

Spectrum statistics:
Number of points: {num_points}
Energy range: {min_energy:.3f} – {max_energy:.3f}
Max intensity: {max_intensity:.3f}
Detected peaks: {peak_desc}

Format your answer as JSON:
{{
  "summary": "...",
  "cot": ["step1", "step2", ...]
}}
"""
    try:
        completion = client.responses.create(
            model=settings.OPENAI_MODEL,
            input=prompt,
            max_output_tokens=350,
        )
        raw_text = completion.output[0].content[0].text.strip()

        # 清理可能出现的 ```json 代码块
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        # 从文本中提取第一个 {...} 作为 JSON
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON detected in LLM output for spectrum analysis.")
        json_str = match.group(0)

        data = json.loads(json_str)

        llm_summary = data.get("summary", "")
        llm_cot = data.get("cot", [])
    except Exception as e:
        print("Spectrum JSON parse error:", e)
        llm_summary = "LLM summary unavailable."
        llm_cot = ["Chain-of-thought unavailable due to API error."]



    result = SpectrumAnalysisResult(
        num_points=len(energy),
        min_energy=float(np.min(energy)),
        max_energy=float(np.max(energy)),
        max_intensity=float(np.max(intensity)),
        normalized=True,
        peaks=peaks,
        llm_summary=llm_summary.strip(),
        llm_cot=llm_cot,
        curve=curve,
    )
    return result

@app.post("/api/chat", response_model=ChatResponse)
async def chat_message(
    req: ChatRequest,
    client: OpenAI = Depends(get_openai_client),
):
    rag_store = get_rag_store()

    # --- 1. RAG retrieval ---
    retrieved = rag_store.retrieve(req.message, k=3)
    context_blocks = []
    sources = []
    for text, idx, score in retrieved:
        context_blocks.append(text)
        sources.append({
            "id": idx,
            "similarity": float(score),
            "text": text[:200],
        })

    context_text = "\n\n".join(context_blocks) if context_blocks else "No retrieved context."

    # --- 2. Build prompt with CoT requirement ---
    prompt = f"""
You are an AI assistant specializing in X-ray spectroscopy.

If the answer is based on approximate toy models or simplified rules,
be explicit and conservative. Do not overclaim scientific accuracy.

Use the retrieved context (may be partial or noisy) to answer the user question.

User question:
{req.message}

Retrieved context:
{context_text}

Produce:
1. "answer": a helpful final answer (5–8 sentences max).
2. "cot": a short chain-of-thought (3–6 steps).

IMPORTANT:
- The chain-of-thought MUST be short and condensed.
- Output ONLY valid JSON.
- Do NOT include markdown or code fences.

Return JSON:
{{
  "answer": "...",
  "cot": ["step1", "step2"]
}}
"""

    # --- 3. Call LLM and extract JSON ---
    try:
        completion = client.responses.create(
            model=settings.OPENAI_MODEL,
            input=prompt,
            max_output_tokens=300,   # allow longer answer
        )

        raw_text = completion.output[0].content[0].text.strip()

        # Clean potential code blocks
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        # Extract JSON substring
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON detected in LLM output.")
        json_str = match.group(0)

        data = json.loads(json_str)

        answer = data.get("answer", "")
        cot = data.get("cot", [])

    except Exception as e:
        print("Chat JSON parse error:", e)
        answer = "LLM response unavailable due to an API error."
        cot = ["Chain-of-thought unavailable due to error."]

    # --- 4. Return final structured response ---
    return {
        "answer": answer,
        "cot": cot,
        "sources": sources
    }
