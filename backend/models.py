from pydantic import BaseModel
from typing import List, Optional


class SpectrumPoint(BaseModel):
    energy: float
    intensity: float


class SpectrumAnalysisRequest(BaseModel):
    points: List[SpectrumPoint]


class SpectrumFeature(BaseModel):
    peak_energy: float
    peak_intensity: float

class SpectrumCurvePoint(BaseModel):
    energy: float
    intensity: float

class SpectrumAnalysisResult(BaseModel):
    num_points: int
    min_energy: float
    max_energy: float
    max_intensity: float
    normalized: bool
    peaks: List[SpectrumFeature]
    llm_summary: str
    llm_cot: List[str]
    curve: List[SpectrumCurvePoint]


class ChatRequest(BaseModel):
    message: str
    use_rag: bool = False

class RAGSource(BaseModel):
    id: str
    similarity: float
    text: str

class ChatResponse(BaseModel):
    answer: str
    cot: List[str]
    sources: Optional[List[RAGSource]] = None
