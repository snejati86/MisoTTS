<div align="center">

<img src="images/repo_banner.png" alt="Miso TTS 8B" width="100%">

# Miso TTS 8B

### State-of-the-Art Text-to-Speech Model

<p>
  <a href="https://misolabs.ai"><img alt="Website" src="https://img.shields.io/badge/Website-misolabs.ai-black?style=for-the-badge"></a>
  <a href="https://huggingface.co/MisoLabs/MisoTTS"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-MisoTTS-yellow?style=for-the-badge"></a>
  <a href="https://github.com/MisoLabsAI"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-MisoLabsAI-181717?style=for-the-badge&logo=github&labelColor=555555"></a>
  <a href="https://x.com/MisoLabsAI"><img alt="X" src="https://img.shields.io/badge/-MisoLabsAI-181717?style=for-the-badge&logo=x&labelColor=555555"></a>
</p>

<p>
  <a href="#quickstart">Quickstart</a> |
  <a href="#model-introduction">Model Introduction</a> |
  <a href="#model-summary">Model Summary</a> |
  <a href="#usage">Usage</a> |
  <a href="#safety">Safety</a>
</p>

</div>

---

## Quickstart

To quickly try the model, you can use the demo hosted on our [landing page](https://misolabs.ai)
at misolabs.ai. To try it locally, follow the instructions below.

If you do not have `uv` installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone the repository and create the environment:

```bash
git clone https://github.com/MisoLabsAI/MisoTTS.git
cd MisoTTS
uv sync --python 3.10
source .venv/bin/activate
```

Then run the example conversation. By default, `run_misotts.py` loads the public
model from [MisoLabs/MisoTTS](https://huggingface.co/MisoLabs/MisoTTS) and
downloads it into the Hugging Face cache if it is not already present on your
machine. The model also uses the Llama 3.2 tokenizer from
[`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B),
which is gated by Meta on Hugging Face. Before running locally, request/accept
access to that repository and log in:

```bash
uv run huggingface-cli login
```

Then run:

```bash
uv run python run_misotts.py
```

The script writes `full_conversation.wav` in the repository root.
If you already have an authorized local tokenizer path or repo, set
`MISO_TTS_TOKENIZER_MODEL` to that path or repo id.

### Studio app: frontend and backend

The repository also includes MisoTTS Studio, a local FastAPI backend plus a
static frontend served from the same app. It keeps one warm model instance in
memory, exposes model status and live logs, generates playable WAV renders, and
supports voice cloning by passing prompt audio plus its transcript.

One-command local start:

```bash
uv run python scripts/start_studio.py
```

Open `http://127.0.0.1:7860` to use the Studio UI. On Apple Silicon, the native
launcher defaults to PyTorch Metal/MPS with `float16` when MPS is available.

If you want to run Uvicorn directly:

```bash
uv run uvicorn studio_server:app --host 127.0.0.1 --port 7860
```

The API is also available:

- `GET /api/status` - model load state and render count.
- `GET /api/logs` - recent Studio log entries.
- `GET /api/logs/stream` - live log stream using Server-Sent Events.
- `POST /api/warm` - starts or restarts model warm-loading.
- `POST /api/render` - multipart form with `text`, generation settings, and
  optional `prompt_audio` plus `prompt_transcript` for cloning.
- `GET /api/renders/{render_id}/audio` - playable WAV.
- `GET /api/renders/{render_id}/download` - downloadable WAV.

Set `MISO_TTS_AUTOLOAD=0` if you want the server to start without warming the
model until `/api/warm` or `/api/render` is called.

#### Docker

The Docker image runs the same frontend and backend together at
`http://127.0.0.1:7860` on macOS, Windows, and Linux:

```bash
docker compose up --build
```

Or without Compose:

```bash
docker build -t misotts-studio .
docker run --rm -p 7860:7860 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v misotts-hf-cache:/app/.cache/huggingface \
  -v misotts-outputs:/app/outputs \
  misotts-studio
```

The default Docker image is portable CPU Linux. Docker Desktop on macOS does not
expose Apple Metal/MPS into Linux containers, so use the native `uv run python
scripts/start_studio.py` path when you want Metal on Apple Silicon. On Linux,
you can set `MISO_TTS_DEVICE=cuda` and use an appropriate CUDA-enabled PyTorch
base image if you adapt the Dockerfile for NVIDIA GPUs.

The model needs access to the gated Llama tokenizer. For Docker, either pass
`HF_TOKEN` or mount a Hugging Face cache that already contains an authorized
tokenizer download.

#### Apple Silicon / Metal

On Apple Silicon, the native Studio defaults to PyTorch's Metal Performance
Shaders backend with `device="mps"` and `dtype="float16"`. In the UI, leave
`Metal GPU` and `Float16` selected, then click **Warm Model** or **Generate**.
From the API:

```bash
curl -X POST http://127.0.0.1:7860/api/warm \
  -H "Content-Type: application/json" \
  -d '{"device":"mps","dtype":"float16","force":true}'
```

This is an experimental PyTorch MPS path, not an MLX port. A native MLX version
would require reimplementing the model, Mimi audio tokenizer integration, and
watermarking path in MLX and converting weights.

With `pip` instead of `uv`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
python run_misotts.py
```

---

## Model Introduction

Miso TTS 8B is a text-to-dialogue RVQ Transformer inspired by the Sesame CSM architecture. It
generates Mimi audio codes from text and optional audio context, using a large
Llama 3.2-style backbone and a smaller autoregressive audio decoder. To find out more
about the architecture, read [our blog post](https://misolabs.ai/blog/miso-tts-8b).

The model is designed for high-quality conversational speech generation.
This repository contains the inference
code, model definition, and setup instructions for running Miso TTS locally.

---

## Model Summary

| Item                | Value           |
| ------------------- | --------------- |
| Model               | Miso TTS 8B     |
| Organization        | Miso Labs       |
| Task                | Text-to-speech  |
| Architecture        | RVQ Transformer |
| Backbone            | `llama-8B`      |
| Audio decoder       | `llama-300M`    |
| Text vocabulary     | `128,256`       |
| Audio vocabulary    | `2,051`         |
| Audio codebooks     | `32`            |
| Audio tokenizer     | Mimi            |
| Max sequence length | `2,048`         |

### Architecture

Miso TTS 8B uses two transformer components:

- A large backbone transformer that consumes text/audio-frame embeddings.
- A smaller decoder transformer that autoregressively predicts higher-order
  audio codebooks within each frame.

The backbone accepts interleaved text and audio tokens, allowing it to condition its generations on
the conversation history.

---

## Usage

### Python

```python
import torch
import torchaudio

from generator import load_miso_8b

device = "cuda" if torch.cuda.is_available() else "cpu"

generator = load_miso_8b(
    device=device,
    model_path_or_repo_id="MisoLabs/MisoTTS",
)

audio = generator.generate(
    text="Hello from Miso.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("miso.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

### Prompted generation

Miso TTS can condition on prior audio for voice cloning.
This is optional; the quickstart example above runs without
prompt audio.

```python
import torchaudio

from generator import Segment, load_miso_8b

generator = load_miso_8b(device="cuda")

prompt_audio, sample_rate = torchaudio.load("prompt.wav")
prompt_audio = torchaudio.functional.resample(
    prompt_audio.squeeze(0),
    orig_freq=sample_rate,
    new_freq=generator.sample_rate,
)

context = [
    Segment(
        speaker=0,
        text="This is the transcript for the prompt audio.",
        audio=prompt_audio,
    )
]

audio = generator.generate(
    text="This is the next sentence to synthesize.",
    speaker=0,
    context=context,
    max_audio_length_ms=10_000,
)
```

---

## Weights

The model weights are hosted publicly on Hugging Face:

```bash
uv run python run_misotts.py
```

The default model repository is
[MisoLabs/MisoTTS](https://huggingface.co/MisoLabs/MisoTTS). The first run
downloads the model automatically through Hugging Face Hub; later runs reuse the
cached copy.

The first run also downloads the SilentCipher watermarking model from
`sony/silentcipher`. If that separate download times out, rerun the command; the
Hugging Face cache resumes from files that already completed.

---

## Deployment Notes

Miso TTS 8B is a large model. For best results, use a CUDA GPU with sufficient
VRAM for the checkpoint precision you are loading. The default inference path
uses `torch.bfloat16`.

---

## Safety

Miso TTS is a speech generation model. Do not use it to impersonate people,
create deceptive audio, commit fraud, or generate harmful content.

Generated audio is watermarked by default. If you deploy this model in another
application, use your own private watermark key and keep it secret.

---

## Links

- Website: [misolabs.ai](https://misolabs.ai)
- Hugging Face: [MisoLabs/MisoTTS](https://huggingface.co/MisoLabs/MisoTTS)
- GitHub: [MisoLabsAI](https://github.com/MisoLabsAI)
- X: [@MisoLabsAI](https://x.com/MisoLabsAI)
