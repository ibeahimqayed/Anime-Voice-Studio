"""Private Anime Voice Studio speech-generation API.

Run on a computer/server with enough resources for Chatterbox Multilingual v3.
No recordings, user text, or generated WAVs are retained by this service.
"""
from __future__ import annotations

import io
import math
import os
import re
import secrets
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

MAX_UPLOAD = 30 * 1024 * 1024
MAX_CHARS = 3000
SUFFIXES = {".wav", ".mp3", ".m4a", ".mp4", ".mov", ".m4v"}
LANGUAGES = {"ar", "ja", "en"}
EMOTIONS = {"Challenge", "Majestic", "Angry", "Sarcastic", "Calm", "Sad"}
MODEL_VERSION = os.getenv("CHATTERBOX_T3_MODEL", "v3")
ALLOWED_ORIGINS = [x.strip() for x in os.getenv(
    "AVS_CORS_ORIGINS", "http://localhost:8765,http://127.0.0.1:8765"
).split(",") if x.strip()]

app = FastAPI(title="Anime Voice Studio Private TTS", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=False,
)

_model = None
_model_load_lock = threading.Lock()
_generate_lock = threading.Lock()


def _device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _model_instance():
    global _model
    if _model is None:
        with _model_load_lock:
            if _model is None:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                _model = ChatterboxMultilingualTTS.from_pretrained(
                    device=_device(), t3_model=MODEL_VERSION
                )
    return _model


def _require_token(authorization: str | None) -> None:
    expected = os.getenv("AVS_API_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="الخادم غير مُعد: ضع AVS_API_TOKEN أولاً")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="مفتاح الخادم مطلوب")
    if not secrets.compare_digest(authorization[7:], expected):
        raise HTTPException(status_code=401, detail="مفتاح الخادم غير صحيح")


def _segments(text: str, limit: int = 250) -> list[str]:
    """Chunk multilingual dialogue without truncating Japanese or Arabic text."""
    sentences = re.findall(r"[^.!?؟。！？؛;\n]+[.!?؟。！？؛;]?", text)
    result: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        while sentence:
            if len(sentence) <= limit:
                result.append(sentence)
                break
            split_at = sentence.rfind(" ", 1, limit + 1)
            if split_at < limit // 2:
                split_at = limit
            result.append(sentence[:split_at].strip())
            sentence = sentence[split_at:].strip()
    return [s for s in result if s]


def _prepare_reference(raw: Path, out: Path, start: float, end: float) -> None:
    args = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", str(start), "-i", str(raw), "-t", str(end - start),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "24000",
        "-c:a", "pcm_s16le", str(out),
    ]
    try:
        completed = subprocess.run(args, capture_output=True, timeout=90, check=False)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="أداة FFmpeg غير مثبتة على الخادم")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=422, detail="انتهت مهلة قراءة الفيديو؛ جرّب ملفًا أصغر")
    if completed.returncode or not out.is_file() or out.stat().st_size < 2048:
        raise HTTPException(status_code=422, detail="لم أجد مسار صوت صالحًا في الملف أو المقطع المحدد")
    try:
        with wave.open(str(out), "rb") as audio:
            if audio.getnframes() / audio.getframerate() < 2.5:
                raise HTTPException(status_code=422, detail="الجزء المستخرج قصير جدًا؛ اختر كلامًا أطول")
    except wave.Error:
        raise HTTPException(status_code=422, detail="تعذرت قراءة المقطع الصوتي المستخرج")


def _synthesize(text: str, language: str, emotion: str, intensity: float, speed: float,
                volume: float, reference: Path | None, cross_language: bool) -> bytes:
    import numpy as np
    import torch

    model = _model_instance()
    # Chatterbox's exaggeration controls expressiveness, not guaranteed emotions.
    bias = {"Challenge": 0.08, "Majestic": 0.02, "Angry": 0.17,
            "Sarcastic": 0.07, "Calm": -0.13, "Sad": -0.07}[emotion]
    exaggeration = max(0.25, min(1.2, 0.38 + intensity / 100 * 0.55 + bias))
    cfg_weight = 0.0 if cross_language else (0.3 if emotion in ("Challenge", "Angry") else 0.5)
    clips = []
    for segment in _segments(text):
        kwargs = {"language_id": language, "exaggeration": exaggeration, "cfg_weight": cfg_weight}
        if reference is not None:
            kwargs["audio_prompt_path"] = str(reference)
        with torch.inference_mode():
            output = model.generate(segment, **kwargs)
        track = output.detach().float().cpu().reshape(-1).numpy()
        if not np.isfinite(track).all() or not len(track):
            raise RuntimeError("The voice model returned invalid audio")
        clips.append(track)
        clips.append(np.zeros(int(model.sr * 0.17), dtype=np.float32))
    sound = np.concatenate(clips[:-1])
    # Speed is applied via FFmpeg to preserve pitch as far as possible.
    with tempfile.TemporaryDirectory(prefix="avs_audio_") as directory:
        raw_out = Path(directory) / "raw.wav"
        edited_out = Path(directory) / "final.wav"
        _write_wav(raw_out, sound, model.sr)
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-i", str(raw_out), "-filter:a", f"atempo={speed:.2f},volume={volume / 100:.3f}",
                   "-c:a", "pcm_s16le", str(edited_out)]
        completed = subprocess.run(command, capture_output=True, timeout=90, check=False)
        if completed.returncode or not edited_out.is_file():
            raise RuntimeError("Audio post-processing failed")
        return edited_out.read_bytes()


def _write_wav(path: Path, sound, sample_rate: int) -> None:
    import numpy as np
    data = (np.clip(sound, -1, 1) * 32767).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(int(sample_rate))
        audio.writeframes(data.tobytes())


@app.get("/health")
def health():
    return {"status": "ok", "configured": bool(os.getenv("AVS_API_TOKEN")),
            "model": "Chatterbox Multilingual", "checkpoint": MODEL_VERSION,
            "model_loaded": _model is not None}


@app.post("/generate")
async def generate(
    text: str = Form(...), language: str = Form(...), emotion: str = Form("Challenge"),
    intensity: float = Form(70), speed: float = Form(1.0), volume: float = Form(70),
    clip_start: float = Form(0), clip_end: float = Form(0),
    reference_language: str = Form(""), has_permission: bool = Form(False),
    reference: UploadFile | None = File(None), authorization: str | None = Header(None),
):
    _require_token(authorization)
    text = text.strip()
    if not text or len(text) > MAX_CHARS:
        raise HTTPException(422, detail="النص المطلوب يجب أن يكون من 1 إلى 3000 حرف")
    if language not in LANGUAGES or emotion not in EMOTIONS:
        raise HTTPException(422, detail="اللغة أو الأسلوب غير مدعوم")
    if not all(math.isfinite(x) for x in (intensity, speed, volume, clip_start, clip_end)):
        raise HTTPException(422, detail="قيم التحكم غير صالحة")
    if not (0 <= intensity <= 100 and 0.6 <= speed <= 1.4 and 0 <= volume <= 100):
        raise HTTPException(422, detail="قيمة المشاعر أو السرعة أو الصوت خارج الحدود")
    if reference_language and reference_language not in LANGUAGES:
        raise HTTPException(422, detail="لغة المرجع غير مدعومة")
    if reference is not None:
        suffix = Path(reference.filename or "").suffix.lower()
        if suffix not in SUFFIXES:
            raise HTTPException(422, detail="صيغة المرجع غير مدعومة")
        if not has_permission:
            raise HTTPException(422, detail="تأكيد حق استخدام الصوت مطلوب")
        if not (0 <= clip_start < clip_end <= 21600 and 3 <= clip_end - clip_start <= 60):
            raise HTTPException(422, detail="حدد مقطعًا بين 3 و60 ثانية")
    elif clip_start != 0 or clip_end != 0:
        raise HTTPException(422, detail="لم يصل ملف مرجعي للمقطع المحدد")

    with tempfile.TemporaryDirectory(prefix="avs_request_") as directory:
        prompt = None
        if reference is not None:
            suffix = Path(reference.filename).suffix.lower()
            # Enforce limit while streaming, without storing arbitrarily large uploads.
            original = Path(directory) / f"uploaded{suffix}"
            size = 0
            with original.open("wb") as target:
                while chunk := await reference.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, detail="حجم المرجع أكبر من 30MB")
                    target.write(chunk)
            if size < 1024:
                raise HTTPException(422, detail="الملف المرجعي فارغ أو غير صالح")
            prompt = Path(directory) / "prompt.wav"
            await run_in_threadpool(_prepare_reference, original, prompt, clip_start, clip_end)
        # Serialize GPU access; each response is an entire WAV, never saved permanently.
        def produce():
            with _generate_lock:
                return _synthesize(text, language, emotion, intensity, speed, volume,
                                   prompt, bool(prompt and reference_language and reference_language != language))
        try:
            audio = await run_in_threadpool(produce)
        except HTTPException:
            raise
        except Exception:
            # Don't leak internal library paths or request details in public API errors.
            raise HTTPException(500, detail="تعذر التوليد على الخادم. راجع سجل تشغيل المحرك")
    return Response(audio, media_type="audio/wav", headers={"Cache-Control": "no-store"})
