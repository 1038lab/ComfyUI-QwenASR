# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ComfyUI-QwenASR is a ComfyUI custom node pack for Qwen3-ASR speech-to-text. It is installed under `ComfyUI/custom_nodes/` and loaded by ComfyUI at startup.

## Installation & Setup

```bash
# Install in a ComfyUI environment
cd ComfyUI/custom_nodes
git clone https://github.com/1038lab/ComfyUI-QwenASR.git
cd ComfyUI-QwenASR
pip install -r requirements.txt
```

There are no build steps, tests, or linting commands. The node is loaded directly by ComfyUI's extension mechanism.

## Architecture

### ComfyUI integration layer (`__init__.py` + `AILab_QwenASR.py`)

- `__init__.py` auto-discovers all `*.py` files in the repo root (excluding itself) and registers any `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` they export. `WEB_DIRECTORY = "./web"` registers the JS frontend.
- `AILab_QwenASR.py` is the sole node definition file. It defines two ComfyUI nodes:
  - `AILab_Qwen3ASR` ("ASR (QwenASR)") — simple STT, returns plain text
  - `AILab_Qwen3ASRSubtitle` ("Subtitle (QwenASR)") — subtitle generation with timestamps, returns text + subtitles + language + output path
- Both nodes use `comfy.model_management` for device selection and `folder_paths` for ComfyUI path resolution.
- Model instances are cached in `_ASR_MODEL_CACHE` (keyed by model path, dtype, device, attention, aligner, batch size, max tokens) to avoid redundant loads between runs.

### Inference library (`qwen_asr/`)

This is a self-contained Python package wrapping the Qwen3-ASR model:

- `qwen_asr/inference/qwen3_asr.py` — `Qwen3ASRModel` class with two backends:
  - `from_pretrained()` → **transformers** backend (used by the ComfyUI node)
  - `Qwen3ASRModel.LLM()` → **vLLM** backend (streaming ASR only; streaming requires vLLM)
- `qwen_asr/inference/qwen3_forced_aligner.py` — `Qwen3ForcedAligner`, used for timestamp-accurate subtitles
- `qwen_asr/inference/utils.py` — audio normalization, chunking, language handling, `parse_asr_output`
- `qwen_asr/core/transformers_backend/` — custom model config, model, and processor for transformers
- `qwen_asr/core/vllm_backend/` — vLLM model registration shim

### Model management

Models are stored at `ComfyUI/models/Qwen3-ASR/<model-name>/`. Resolution order in `_resolve_model_path()`:
1. Scan `folder_paths` registered paths and `extra_model_paths.yaml`
2. Check `~/.cache/huggingface/hub` or `~/.cache/modelscope/hub` (copies to local storage)
3. Download via HuggingFace or ModelScope (controlled by `config.json` `defaults.source`)

### Configuration (`config.json`)

Loaded with file-mtime caching. Controls available models, aligners, download source, and inference defaults. Can be edited without restarting ComfyUI (cache invalidates on mtime change).

### Audio flow

ComfyUI `AUDIO` dict → `_normalize_audio()` → mono float32 numpy array at native sample rate → `Qwen3ASRModel.transcribe()` which internally resamples to 16kHz and chunks long audio. Subtitle timestamps come from `Qwen3ForcedAligner.align()` applied per chunk, then offset-corrected and merged.

### Subtitle grouping

`_group_time_stamps()` in `AILab_QwenASR.py` merges word-level aligner tokens into subtitle lines using configurable `split_mode` (combination of punctuation / pause / length thresholds).
