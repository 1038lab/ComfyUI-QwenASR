# ComfyUI-QwenASR
# ComfyUI custom nodes for Qwen3-ASR speech-to-text models.
# Models License Notice:
# - Qwen3-ASR: Apache-2.0 License (https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
# This integration script follows GPL-3.0 License.

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import folder_paths
import comfy.model_management as model_management

_CURRENT_DIR = Path(__file__).parent
_QWEN_ASR_DIR = _CURRENT_DIR / "qwen_asr"
if _QWEN_ASR_DIR.exists() and str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

try:
    from qwen_asr import Qwen3ASRModel
except Exception as _e:
    Qwen3ASRModel = None
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None

# ComfyUI model folder registration
QWEN3_ASR_ROOT = os.path.join(folder_paths.models_dir, "Qwen3-ASR")
os.makedirs(QWEN3_ASR_ROOT, exist_ok=True)
folder_paths.add_model_folder_path("Qwen3-ASR", QWEN3_ASR_ROOT)

SUPPORTED_LANGUAGES = [
    "auto",
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish",
    "Portuguese", "Indonesian", "Italian", "Korean", "Russian", "Thai",
    "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish",
    "Danish", "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek",
    "Hungarian", "Macedonian", "Romanian",
]

_ASR_MODEL_CACHE = {}
_CONFIG_CACHE = {"mtime": None, "data": None}


def _default_config():
    return {
        "models": {
            "Qwen/Qwen3-ASR-1.7B": "Qwen3-ASR-1.7B",
            "Qwen/Qwen3-ASR-0.6B": "Qwen3-ASR-0.6B",
        },
        "aligners": {
            "None": None,
            "Qwen/Qwen3-ForcedAligner-0.6B": "Qwen3-ForcedAligner-0.6B",
        },
        "sources": ["HuggingFace", "ModelScope"],
        "defaults": {
            "repo_id": "Qwen/Qwen3-ASR-0.6B",
            "source": "HuggingFace",
            "precision": "bf16",
            "attention": "auto",
            "language": "auto",
            "forced_aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
        },
    }


def _load_config():
    path = _CURRENT_DIR / "config.json"
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = None

    cache = _CONFIG_CACHE
    if cache["data"] is not None and cache["mtime"] == mtime:
        return cache["data"]

    data = _default_config()
    if mtime is not None:
        try:
            import json

            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data.update(loaded)
        except Exception as e:
            print(f"[Qwen3ASR] Failed to read config.json: {e}")

    cache["mtime"] = mtime
    cache["data"] = data
    return data


def _get_model_ids():
    models = _load_config().get("models") or {}
    if isinstance(models, dict) and models:
        return models
    return _default_config()["models"]


def _get_aligner_ids():
    aligners = _load_config().get("aligners") or {}
    if isinstance(aligners, dict) and aligners:
        return aligners
    return _default_config()["aligners"]


def _get_sources():
    sources = _load_config().get("sources") or []
    if isinstance(sources, list) and sources:
        return sources
    return _default_config()["sources"]


def _get_defaults():
    defaults = _load_config().get("defaults") or {}
    if isinstance(defaults, dict) and defaults:
        return defaults
    return _default_config()["defaults"]
_EXTRA_MODEL_PATHS = None


def _normalize_paths(paths):
    normalized = []
    for p in paths:
        if not isinstance(p, str):
            continue
        p = p.strip()
        if not p:
            continue
        normalized.append(os.path.normpath(os.path.expanduser(p)))
    return normalized


def _extract_yaml_paths(data):
    if not isinstance(data, dict):
        return []

    def split_paths(value):
        if isinstance(value, str):
            lines = [line.strip() for line in value.splitlines()]
            return [line for line in lines if line]
        if isinstance(value, list):
            return [v for v in value if isinstance(v, str) and v.strip()]
        return []

    def is_abs(p):
        if not p:
            return False
        if os.path.isabs(p):
            return True
        if len(p) > 1 and p[1] == ":":
            return True
        return False

    def pull(d):
        found = []
        for key in ("paths", "roots", "folders", "models", "search_paths"):
            value = d.get(key)
            found.extend(split_paths(value))
        return found

    paths = []
    paths.extend(pull(data))
    for section in data.values():
        if not isinstance(section, dict):
            continue
        base_path = section.get("base_path")
        if isinstance(base_path, str) and base_path.strip():
            paths.append(base_path)
        for key, value in section.items():
            if key == "base_path" or key == "is_default":
                continue
            for item in split_paths(value):
                if is_abs(item) or not isinstance(base_path, str):
                    paths.append(item)
                else:
                    paths.append(os.path.join(base_path, item))
    return paths


def _load_extra_model_paths():
    global _EXTRA_MODEL_PATHS
    if _EXTRA_MODEL_PATHS is not None:
        return _EXTRA_MODEL_PATHS

    candidates = []
    try:
        base_path = Path(getattr(folder_paths, "base_path", ""))
        if base_path:
            candidates.append(base_path / "extra_model_paths.yaml")
    except Exception:
        pass

    collected = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            import yaml
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print(f"[Qwen3ASR] Failed to read {path}: {e}")
            continue
        collected.extend(_extract_yaml_paths(data))

    normalized = []
    for p in _normalize_paths(collected):
        if os.path.isdir(p) and p not in normalized:
            normalized.append(p)

    _EXTRA_MODEL_PATHS = tuple(normalized)
    return _EXTRA_MODEL_PATHS


def _find_local_model(repo_id: str) -> Optional[str]:
    model_name = _get_model_ids().get(repo_id) or _get_aligner_ids().get(repo_id) or repo_id.split("/")[-1]
    candidates = []

    default_root = _model_storage_path(repo_id)
    if os.path.isdir(default_root):
        candidates.append(default_root)

    try:
        asr_roots = folder_paths.get_folder_paths("Qwen3-ASR") or []
        for root in asr_roots:
            candidates.append(os.path.join(root, model_name))
    except Exception:
        pass

    for root in _load_extra_model_paths():
        candidates.append(os.path.join(root, model_name))
        candidates.append(os.path.join(root, "Qwen3-ASR", model_name))
        candidates.append(os.path.join(root, "ASR", "Qwen3-ASR", model_name))

    for path in candidates:
        if os.path.isdir(path) and os.listdir(path):
            return path
    return None


def _model_storage_path(repo_id: str) -> str:
    name = _get_model_ids().get(repo_id) or _get_aligner_ids().get(repo_id) or repo_id.replace("/", "_")
    return os.path.join(QWEN3_ASR_ROOT, name)


def _try_copy_cached(repo_id: str, target_dir: str) -> bool:
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        return True

    hf_cache = os.path.join(Path.home(), ".cache", "huggingface", "hub")
    hf_dir = os.path.join(hf_cache, f"models--{repo_id.replace('/', '--')}")
    snapshots = os.path.join(hf_dir, "snapshots")
    if os.path.isdir(snapshots):
        entries = sorted(os.listdir(snapshots))
        if entries:
            source = os.path.join(snapshots, entries[-1])
            try:
                shutil.copytree(source, target_dir, dirs_exist_ok=True)
                return True
            except Exception:
                return False

    ms_cache = os.path.join(Path.home(), ".cache", "modelscope", "hub")
    ms_dir = os.path.join(ms_cache, repo_id.replace("/", os.sep))
    if os.path.isdir(ms_dir):
        try:
            shutil.copytree(ms_dir, target_dir, dirs_exist_ok=True)
            return True
        except Exception:
            return False

    return False


def _download_to_local(repo_id: str, source: str, target_dir: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    if source == "ModelScope":
        try:
            from modelscope import snapshot_download
        except Exception as e:
            raise RuntimeError("modelscope is required for ModelScope downloads") from e
        snapshot_download(repo_id, local_dir=target_dir)
    else:
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise RuntimeError("huggingface_hub is required for HuggingFace downloads") from e
        snapshot_download(repo_id, local_dir=target_dir)

    return target_dir


def _normalize_audio(audio) -> Optional[Tuple[np.ndarray, int]]:
    if audio is None:
        return None

    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if waveform is None or sample_rate is None:
        return None

    wave = waveform[0]
    if wave.ndim == 2 and wave.shape[0] > 1:
        wave = torch.mean(wave, dim=0)
    else:
        wave = wave.squeeze(0)

    return (wave.detach().cpu().numpy().astype(np.float32), int(sample_rate))


def _resolve_model_path(repo_id: str, source: str) -> str:
    local_path = _find_local_model(repo_id)
    if local_path:
        return local_path

    target_dir = _model_storage_path(repo_id)
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        return target_dir

    if _try_copy_cached(repo_id, target_dir):
        return target_dir

    return _download_to_local(repo_id, source, target_dir)


def _build_dtype(precision: str, device: torch.device) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        if device.type == "mps":
            return torch.float16
        return torch.bfloat16
    return torch.float32


def _cache_key(
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
    attention: str,
    forced_aligner_path: str,
    max_inference_batch_size: int,
    max_new_tokens: int,
) -> tuple:
    return (
        model_path,
        str(dtype),
        str(device),
        attention,
        forced_aligner_path or "",
        int(max_inference_batch_size),
        int(max_new_tokens),
    )


def _load_cached_model(
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
    attention: str,
    forced_aligner_path: str,
    max_inference_batch_size: int = 32,
    max_new_tokens: int = 256,
):
    key = _cache_key(model_path, dtype, device, attention, forced_aligner_path, max_inference_batch_size, max_new_tokens)
    cached = _ASR_MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    model_kwargs = {
        "dtype": dtype,
        "device_map": str(device),
        "max_inference_batch_size": int(max_inference_batch_size),
        "max_new_tokens": int(max_new_tokens),
    }
    if attention != "auto":
        model_kwargs["attn_implementation"] = attention
    if forced_aligner_path:
        model_kwargs["forced_aligner"] = forced_aligner_path
        model_kwargs["forced_aligner_kwargs"] = {
            "dtype": dtype,
            "device_map": str(device),
        }
        if attention != "auto":
            model_kwargs["forced_aligner_kwargs"]["attn_implementation"] = attention

    model = Qwen3ASRModel.from_pretrained(model_path, **model_kwargs)
    _ASR_MODEL_CACHE[key] = model
    return model


def _format_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _build_srt(time_stamps) -> str:
    if not time_stamps:
        return ""
    lines = []
    for idx, item in enumerate(time_stamps, start=1):
        lines.append(str(idx))
        lines.append(f"{_format_srt_time(item.start_time)} --> {_format_srt_time(item.end_time)}")
        lines.append(item.text or "")
        lines.append("")
    return "\n".join(lines).strip()


def _join_tokens(a: str, b: str) -> str:
    if not a:
        return b
    if not b:
        return a
    # If either side contains CJK, join without space.
    for ch in (a[-1], b[0]):
        if "\u4e00" <= ch <= "\u9fff":
            return f"{a}{b}"
    return f"{a} {b}"


def _group_time_stamps(time_stamps, max_gap_sec: float, max_chars: int, split_mode: str):
    if not time_stamps:
        return []
    groups = []
    cur = None
    punct = ("。", "！", "？", ".", "!", "?")
    for item in time_stamps:
        text = (item.text or "").strip()
        if not text:
            continue
        if cur is None:
            cur = {
                "start": item.start_time,
                "end": item.end_time,
                "text": text,
            }
            continue

        gap = float(item.start_time) - float(cur["end"])
        too_far = gap > max_gap_sec
        too_long = max_chars > 0 and (len(cur["text"]) + len(text)) > max_chars
        end_sentence = any(cur["text"].endswith(p) for p in punct)

        split_by_punct = split_mode in ("split_by_punctuation", "split_by_punctuation_or_length", "split_by_punctuation_or_pause", "split_by_punctuation_or_pause_or_length")
        split_by_length = split_mode in ("split_by_length", "split_by_punctuation_or_length", "split_by_punctuation_or_pause_or_length")
        split_by_pause = split_mode in ("split_by_pause", "split_by_punctuation_or_pause", "split_by_punctuation_or_pause_or_length")

        should_split = False
        if split_by_punct and end_sentence:
            should_split = True
        if split_by_length and too_long:
            should_split = True
        if split_by_pause and too_far:
            should_split = True

        if should_split:
            groups.append(cur)
            cur = {
                "start": item.start_time,
                "end": item.end_time,
                "text": text,
            }
        else:
            cur["text"] = _join_tokens(cur["text"], text).strip()
            cur["end"] = item.end_time

    if cur is not None:
        groups.append(cur)
    return groups


def _build_srt_from_groups(groups) -> str:
    if not groups:
        return ""
    lines = []
    for idx, g in enumerate(groups, start=1):
        lines.append(str(idx))
        lines.append(f"{_format_srt_time(g['start'])} --> {_format_srt_time(g['end'])}")
        lines.append(g["text"])
        lines.append("")
    return "\n".join(lines).strip()


def _default_output_dir() -> str:
    base = folder_paths.get_output_directory()
    return os.path.join(base, "ComfyUI-QwenASR")


def _is_dir_path(path: str) -> bool:
    if not path:
        return False
    if path.endswith(("/", "\\")):
        return True
    return os.path.isdir(path)


def _make_default_filename(ext: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"qwenasr_subtitle_{stamp}{ext}"


class AILab_Qwen3ASR:
    @classmethod
    def INPUT_TYPES(cls):
        defaults = _get_defaults()
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio input to transcribe."}),
            },
            "optional": {
                "model": (list(_get_model_ids().keys()), {"default": defaults.get("repo_id", "Qwen/Qwen3-ASR-0.6B"), "tooltip": "Choose the ASR model size."}),
                "precision": (["bf16", "fp16", "fp32"], {"default": defaults.get("precision", "bf16"), "tooltip": "Inference precision."}),
                "language": (SUPPORTED_LANGUAGES, {"default": defaults.get("language", "auto"), "tooltip": "Force language or auto-detect."}),
                "hints": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional hints/keywords (names, terms) to improve recognition."}),
                "unload_models": ("BOOLEAN", {"default": True, "tooltip": "Unload cached model after inference."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("TEXT",)
    FUNCTION = "transcribe"
    CATEGORY = "🧪AILab/🎙️QwenASR"

    def transcribe(
        self,
        audio,
        model="Qwen/Qwen3-ASR-0.6B",
        precision="bf16",
        language="auto",
        hints="",
        unload_models=True,
    ):
        if Qwen3ASRModel is None:
            raise RuntimeError(f"qwen-asr not available: {_IMPORT_ERROR}")

        device = model_management.get_torch_device()
        dtype = _build_dtype(precision, device)

        source = _get_defaults().get("source", "HuggingFace")
        model_path = _resolve_model_path(model, source)

        audio_data = _normalize_audio(audio)
        if audio_data is None:
            return ("",)

        lang = None if language == "auto" else language
        ctx = hints.strip() if isinstance(hints, str) else ""

        model = _load_cached_model(model_path, dtype, device, "auto", "")
        results = model.transcribe(
            audio=audio_data,
            language=lang,
            context=ctx if ctx else None,
            return_time_stamps=False,
        )

        result = results[0]
        text = result.text or ""

        if unload_models:
            _ASR_MODEL_CACHE.clear()
            try:
                model_management.soft_empty_cache()
            except Exception:
                pass

        return (text,)


class AILab_Qwen3ASRSubtitle:
    @classmethod
    def INPUT_TYPES(cls):
        defaults = _get_defaults()
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio input to transcribe."}),
            },
            "optional": {
                "model": (list(_get_model_ids().keys()), {"default": defaults.get("repo_id", "Qwen/Qwen3-ASR-0.6B"), "tooltip": "Choose the ASR model size."}),
                "precision": (["bf16", "fp16", "fp32"], {"default": defaults.get("precision", "bf16"), "tooltip": "Inference precision."}),
                "attention": (["auto", "flash_attention_2", "sdpa", "eager"], {"default": defaults.get("attention", "auto"), "tooltip": "Attention backend override."}),
                "forced_aligner": (list(_get_aligner_ids().keys()), {"default": defaults.get("forced_aligner", "Qwen/Qwen3-ForcedAligner-0.6B"), "tooltip": "Forced aligner for timestamped subtitles."}),
                "language": (SUPPORTED_LANGUAGES, {"default": defaults.get("language", "auto"), "tooltip": "Force language or auto-detect."}),
                "hints": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional hints/keywords (names, terms) to improve recognition."}),
                "output_format": (["none", "txt", "srt"], {"default": "none", "tooltip": "File save format only (does not change subtitle output)."}),
                "output_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional output file path (relative goes to ComfyUI output)."}),
                "split_mode": (["split_by_punctuation_or_pause_or_length", "split_by_punctuation_or_pause", "split_by_punctuation_or_length", "split_by_punctuation", "split_by_pause", "split_by_length"], {"default": "split_by_punctuation_or_pause_or_length", "tooltip": "Sentence splitting strategy."}),
                "max_gap_sec": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 8.0, "step": 0.1, "tooltip": "Max silence gap to keep the same sentence."}),
                "max_chars": ("INT", {"default": 40, "min": 0, "max": 200, "tooltip": "Optional max characters per line (0 = no limit)."}),
                "max_inference_batch_size": ("INT", {"default": 32, "min": 1, "max": 256, "tooltip": "Batch size for inference/alignment to avoid OOM."}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 2048, "tooltip": "Max new tokens per chunk."}),
                "unload_models": ("BOOLEAN", {"default": True, "tooltip": "Unload cached model after inference."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("TEXT", "SUBTITLES", "LANUGAGE", "OUTPUT_PATH")
    FUNCTION = "transcribe"
    CATEGORY = "🧪AILab/🎙️QwenASR"

    def transcribe(
        self,
        audio,
        model="Qwen/Qwen3-ASR-0.6B",
        precision="bf16",
        attention="auto",
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        language="auto",
        hints="",
        output_format="none",
        output_path="",
        split_mode="split_by_punctuation_or_pause_or_length",
        max_gap_sec=0.6,
        max_chars=60,
        max_inference_batch_size=32,
        max_new_tokens=256,
        unload_models=True,
    ):
        if Qwen3ASRModel is None:
            raise RuntimeError(f"qwen-asr not available: {_IMPORT_ERROR}")

        device = model_management.get_torch_device()
        dtype = _build_dtype(precision, device)

        source = _get_defaults().get("source", "HuggingFace")
        model_path = _resolve_model_path(model, source)

        forced_aligner_path = ""
        if forced_aligner and forced_aligner != "None":
            forced_aligner_path = _resolve_model_path(forced_aligner, source)

        audio_data = _normalize_audio(audio)
        if audio_data is None:
            return ("", "", "")

        lang = None if language == "auto" else language
        ctx = hints.strip() if isinstance(hints, str) else ""

        model = _load_cached_model(
            model_path,
            dtype,
            device,
            attention,
            forced_aligner_path,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )
        results = model.transcribe(
            audio=audio_data,
            language=lang,
            context=ctx if ctx else None,
            return_time_stamps=True,
        )

        result = results[0]
        text = result.text or ""
        detected_lang = result.language or ""
        subtitles = ""
        file_path = ""
        time_stamps = getattr(result, "time_stamps", None)
        groups = _group_time_stamps(time_stamps, max_gap_sec=max_gap_sec, max_chars=max_chars, split_mode=split_mode)
        # Always build subtitle output
        lines = []
        for g in groups:
            lines.append(f"{g['start']:.2f}-{g['end']:.2f}: {g['text']}")
        subtitles = "\n".join(lines) if lines else ""

        # Optional file save
        if output_format != "none":
            out_path = (output_path or "").strip()
            if not os.path.isabs(out_path):
                if out_path == "":
                    out_path = _default_output_dir()
                out_path = os.path.join(folder_paths.get_output_directory(), out_path)

            if _is_dir_path(out_path):
                ext = ".srt" if output_format == "srt" else ".txt"
                out_path = os.path.join(out_path, _make_default_filename(ext))
            else:
                root, ext = os.path.splitext(out_path)
                if not ext:
                    ext = ".srt" if output_format == "srt" else ".txt"
                    out_path = root + ext
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if output_format == "srt":
                file_content = _build_srt_from_groups(groups)
            else:
                file_content = subtitles
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(file_content)
            file_path = out_path

        if unload_models:
            _ASR_MODEL_CACHE.clear()
            try:
                model_management.soft_empty_cache()
            except Exception:
                pass

        return (text, subtitles, detected_lang, file_path)


NODE_CLASS_MAPPINGS = {
    "AILab_Qwen3ASR": AILab_Qwen3ASR,
    "AILab_Qwen3ASRSubtitle": AILab_Qwen3ASRSubtitle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_Qwen3ASR": "ASR (QwenASR)",
    "AILab_Qwen3ASRSubtitle": "Subtitle (QwenASR)",
}
