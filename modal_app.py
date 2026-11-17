"""Anime Voice Studio — on-demand Chatterbox Multilingual v3 GPU API.

Deploy with `modal deploy modal_app.py` from a clone of the repo.
Required Modal Secret named `anime-voice-studio-engine`: AVS_API_TOKEN.
Do not put API tokens in GitHub files or client-side app code.
"""

import modal

app = modal.App("anime-voice-studio-engine")

# Build may download several GB of model and PyTorch dependencies. Keep versions
# aligned by letting the official chatterbox-tts package select its requirements.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "chatterbox-tts",
        "fastapi>=0.115,<1",
        "uvicorn[standard]>=0.34,<1",
        "python-multipart>=0.0.20,<1",
    )
    .add_local_file("backend.py", "/root/avs_backend.py")
    .env({"AVS_CORS_ORIGINS": "https://ibeahimqayed.github.io"})
)


@app.function(
    image=image,
    gpu="L4",
    secrets=[modal.Secret.from_name("anime-voice-studio-engine")],
    timeout=600,
    scaledown_window=30,
    max_containers=1,
)
@modal.asgi_app()
def voice_api():
    import sys
    sys.path.insert(0, "/root")
    from avs_backend import app as fastapi_app

    return fastapi_app
