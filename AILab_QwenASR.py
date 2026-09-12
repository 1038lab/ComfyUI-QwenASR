# ComfyUI-QwenASR
# ComfyUI custom nodes for Qwen3-ASR speech-to-text models.

import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union

import numpy as np
import torch
import folder_paths
import comfy.model_management as model_management
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    AutoModelForTokenClassification,
)

_CURRENT_DIR = Path(__file__).parent

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
_ALIGNER_CACHE = {}
_CONFIG_CACHE = {"mtime": None, "data": None}
_EXTRA_MODEL_PATHS = None


class WordTimestamp:
    def __init__(self, text: str = "", start_time: float = 0.0, end_time: float = 0.0):
        self.text = str(text)
        self.start_time = float(start_time)
        self.end_time = float(end_time)

    def __repr__(self):
        return f"WordTimestamp({self.text!r}, {self.start_time:.3f}, {self.end_time:.3f})"


class ForcedAlignResult:
    def __init__(self, items: List[WordTimestamp]):
        self.items = items

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _default_config():
    return {
        "models": {
            "Qwen/Qwen3-ASR-1.7B-hf": "Qwen3-ASR-1.7B-hf",
            "Qwen/Qwen3-ASR-0.6B-hf": "Qwen3-ASR-0.6B-hf",
        },
        "aligners": {
            "None": None,
            "Qwen/Qwen3-ForcedAligner-0.6B-hf": "Qwen3-ForcedAligner-0.6B-hf",
        },
        "sources": ["HuggingFace", "ModelScope"],
        "defaults": {
            "repo_id": "Qwen/Qwen3-ASR-0.6B-hf",
            "source": "HuggingFace",
            "precision": "bf16",
            "attention": "auto",
            "language": "auto",
            "forced_aligner": "Qwen/Qwen3-ForcedAligner-0.6B-hf",
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
            if key in ("base_path", "is_default"):
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


def _model_storage_path(repo_id: str) -> str:
    name = _get_model_ids().get(repo_id) or _get_aligner_ids().get(repo_id) or repo_id.split("/")[-1]
    return os.path.join(QWEN3_ASR_ROOT, name)


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


def _normalize_audio(audio, target_sr: int = 16000) -> Optional[Tuple[np.ndarray, int]]:
    if audio is None:
        return None

    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if waveform is None or sample_rate is None:
        return None

    wave = waveform[0]
    if wave.ndim == 2 and wave.shape[0] > 1:
        wave = torch.mean(wave, dim=0)
    elif wave.ndim == 2:
        wave = wave.squeeze(0)

    if int(sample_rate) != target_sr:
        import torchaudio.functional as F
        wave = F.resample(wave, int(sample_rate), target_sr)
        sample_rate = target_sr

    return (wave.detach().cpu().numpy().astype(np.float32), int(sample_rate))


def _build_dtype(precision: str, device: torch.device) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        if device.type == "mps":
            return torch.float16
        return torch.bfloat16
    return torch.float32


def _is_hf_native(repo_id_or_path: str) -> bool:
    s = str(repo_id_or_path).lower()
    return "-hf" in s


# -------------------------------------------------------------------------
# Native Transformers Engine Implementation
# -------------------------------------------------------------------------

class _HFASREngine:
    def __init__(self, model_path: str, dtype: torch.dtype, device: torch.device, attention: str):
        self.processor = AutoProcessor.from_pretrained(model_path)
        kwargs = {"dtype": dtype}
        if attention != "auto":
            kwargs["attn_implementation"] = attention
        if device.type in ("mps", "cpu"):
            self.model = AutoModelForMultimodalLM.from_pretrained(model_path, **kwargs).to(device).eval()
        else:
            kwargs["device_map"] = str(device)
            self.model = AutoModelForMultimodalLM.from_pretrained(model_path, **kwargs).eval()
        self.device = device
        self.dtype = dtype

    def transcribe(
        self,
        audio_data: Tuple[np.ndarray, int],
        language: Optional[str] = None,
        context: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> Tuple[str, str]:
        wave, sr = audio_data
        duration = len(wave) / float(sr)

        if duration <= 120.0:
            return self._transcribe_chunk(wave, language, context, max_new_tokens)

        chunk_samples = 60 * sr
        texts = []
        detected_lang = ""
        for start_idx in range(0, len(wave), chunk_samples):
            chunk_wave = wave[start_idx : start_idx + chunk_samples]
            if len(chunk_wave) < int(0.5 * sr):
                continue
            sub_text, sub_lang = self._transcribe_chunk(chunk_wave, language, context, max_new_tokens)
            if sub_text:
                texts.append(sub_text)
            if not detected_lang and sub_lang:
                detected_lang = sub_lang

        full_text = " ".join(texts)
        return full_text, detected_lang

    def _transcribe_chunk(self, wave: np.ndarray, language: Optional[str], context: Optional[str], max_new_tokens: int):
        kwargs = {"audio": wave}
        if language and language != "auto":
            kwargs["language"] = language
        if context:
            kwargs["prompt"] = context

        inputs = self.processor.apply_transcription_request(**kwargs)
        inputs = inputs.to(self.device, self.dtype)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        parsed = self.processor.decode(generated_ids, return_format="parsed")[0]
        if isinstance(parsed, dict):
            text = parsed.get("transcription", "") or ""
            lang = parsed.get("language", "") or ""
        else:
            text = str(parsed)
            lang = ""
        return text.strip(), lang


class _HFAlignerEngine:
    def __init__(self, model_path: str, dtype: torch.dtype, device: torch.device, attention: str):
        self.processor = AutoProcessor.from_pretrained(model_path)
        kwargs = {"dtype": dtype}
        if attention != "auto":
            kwargs["attn_implementation"] = attention
        if device.type in ("mps", "cpu"):
            self.model = AutoModelForTokenClassification.from_pretrained(model_path, **kwargs).to(device).eval()
        else:
            kwargs["device_map"] = str(device)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path, **kwargs).eval()
        self.device = device
        self.dtype = dtype

    def align(
        self,
        audio: Union[Tuple[np.ndarray, int], Any],
        text: Union[str, List[str]],
        language: Union[str, List[str]] = "English",
    ) -> List[ForcedAlignResult]:
        if isinstance(audio, tuple) and len(audio) == 2:
            wave, sr = audio
        else:
            norm = _normalize_audio(audio)
            wave, sr = norm if norm is not None else (np.zeros(16000, dtype=np.float32), 16000)

        transcript = text[0] if isinstance(text, list) else text
        lang = language[0] if isinstance(language, list) else language

        aligner_inputs, word_lists = self.processor.prepare_forced_aligner_inputs(
            audio=wave,
            transcript=transcript,
            language=lang,
        )
        aligner_inputs = aligner_inputs.to(self.device, self.dtype)

        with torch.inference_mode():
            outputs = self.model(**aligner_inputs)

        timestamp_token_id = getattr(self.model.config, "timestamp_token_id", None)
        timestamps = self.processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=aligner_inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=timestamp_token_id,
        )[0]

        items = [
            WordTimestamp(
                text=item.get("text", "") if isinstance(item, dict) else getattr(item, "text", ""),
                start_time=float(item.get("start_time", 0.0) if isinstance(item, dict) else getattr(item, "start_time", 0.0)),
                end_time=float(item.get("end_time", 0.0) if isinstance(item, dict) else getattr(item, "end_time", 0.0)),
            )
            for item in timestamps
        ]
        return [ForcedAlignResult(items=items)]


class UnifiedASRModel:
    def __init__(self, asr_engine: Any, aligner_engine: Any = None):
        self.asr = asr_engine
        self.aligner = aligner_engine

    def transcribe(
        self,
        audio_data: Tuple[np.ndarray, int],
        language: Optional[str] = None,
        context: Optional[str] = None,
        return_time_stamps: bool = False,
        max_new_tokens: int = 256,
    ) -> Tuple[str, str, Optional[List[WordTimestamp]]]:
        text, detected_lang = self.asr.transcribe(
            audio_data=audio_data,
            language=language,
            context=context,
            max_new_tokens=max_new_tokens,
        )

        time_stamps = None
        if return_time_stamps and self.aligner is not None and text.strip():
            lang_align = detected_lang or language or "English"
            align_results = self.aligner.align(audio=audio_data, text=text, language=lang_align)
            if align_results and len(align_results) > 0:
                time_stamps = list(align_results[0])

        return text, detected_lang, time_stamps


def _load_cached_aligner(
    aligner_path: str,
    dtype: torch.dtype,
    device: torch.device,
    attention: str = "auto",
):
    key = (aligner_path, str(dtype), str(device), attention)
    cached = _ALIGNER_CACHE.get(key)
    if cached is not None:
        return cached

    if not _is_hf_native(aligner_path):
        raise ValueError(
            f"Model '{aligner_path}' is not supported. "
            "Please use the official native model: 'Qwen/Qwen3-ForcedAligner-0.6B-hf'."
        )

    aligner = _HFAlignerEngine(aligner_path, dtype, device, attention)
    _ALIGNER_CACHE[key] = aligner
    return aligner


def _load_cached_asr(
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
    attention: str = "auto",
    forced_aligner_path: str = "",
    max_inference_batch_size: int = 32,
    max_new_tokens: int = 256,
) -> UnifiedASRModel:
    key = (
        model_path,
        str(dtype),
        str(device),
        attention,
        forced_aligner_path or "",
        int(max_inference_batch_size),
        int(max_new_tokens),
    )
    cached = _ASR_MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    if not _is_hf_native(model_path):
        raise ValueError(
            f"Model '{model_path}' is not supported. "
            "Please use official native models: 'Qwen/Qwen3-ASR-0.6B-hf' or 'Qwen/Qwen3-ASR-1.7B-hf'."
        )

    asr_engine = _HFASREngine(model_path, dtype, device, attention)
    aligner_engine = None
    if forced_aligner_path:
        aligner_engine = _load_cached_aligner(forced_aligner_path, dtype, device, attention)

    unified = UnifiedASRModel(asr_engine=asr_engine, aligner_engine=aligner_engine)
    _ASR_MODEL_CACHE[key] = unified
    return unified


# -------------------------------------------------------------------------
# Subtitle & Text Utilities
# -------------------------------------------------------------------------

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
    for ch in (a[-1], b[0]):
        if "\u4e00" <= ch <= "\u9fff":
            return f"{a}{b}"
    return f"{a} {b}"


def _restore_punctuation(time_stamps, full_text: str):
    if not time_stamps or not full_text:
        return time_stamps

    punct_set = set(".,!?:;—…~。，！？、；：”’'\"()（）[]【】《》")
    text_len = len(full_text)
    text_pos = 0

    restored = []
    for ts in time_stamps:
        raw_word = str(ts.text or "").strip()
        if not raw_word:
            restored.append(ts)
            continue

        pattern = re.escape(raw_word)
        m = re.search(pattern, full_text[text_pos:], re.IGNORECASE)
        if not m:
            restored.append(ts)
            continue

        word_start = text_pos + m.start()
        word_end = text_pos + m.end()
        matched_cased_word = full_text[word_start:word_end]

        trail_pos = word_end
        trail_punct = []
        while trail_pos < text_len:
            ch = full_text[trail_pos]
            if ch in punct_set:
                trail_punct.append(ch)
                trail_pos += 1
            elif ch.isspace():
                trail_pos += 1
            else:
                break

        new_text = matched_cased_word + "".join(trail_punct)
        restored.append(type(ts)(text=new_text, start_time=ts.start_time, end_time=ts.end_time))
        text_pos = trail_pos

    return restored


_DANGLING_TAILS = {
    "for", "to", "of", "in", "on", "at", "with", "by", "from", "about", "into", "through",
    "the", "a", "an",
    "and", "or", "but", "nor", "so", "yet",
    "that", "which", "who", "whom", "whose",
    "as", "if", "when", "than", "because", "while", "where"
}


def _group_time_stamps(time_stamps, max_gap_sec: float, max_chars: int, split_mode: str):
    if not time_stamps:
        return []
    groups = []
    cur = None
    sentence_punct = ("。", "！", "？", ".", "!", "?")
    clause_punct = (",", ";", ":", "，", "；", "：", "、")
    all_punct = sentence_punct + clause_punct

    n_items = len(time_stamps)
    for idx, item in enumerate(time_stamps):
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
        end_sentence = any(cur["text"].endswith(p) for p in sentence_punct)
        end_clause = any(cur["text"].endswith(p) for p in clause_punct)
        this_word_ends_sentence = any(text.endswith(p) for p in sentence_punct)
        this_word_ends_clause = any(text.endswith(p) for p in all_punct)

        # Lookahead: check if the sentence ends in the next 1-3 words (Orphan Protection)
        sentence_ends_soon = this_word_ends_sentence
        if not sentence_ends_soon:
            for lookahead in range(1, 4):
                if idx + lookahead < n_items:
                    next_t = (time_stamps[idx + lookahead].text or "").strip()
                    if any(next_t.endswith(p) for p in sentence_punct):
                        sentence_ends_soon = True
                        break

        # Lookahead: check if a clause punct (comma, etc.) ends on this word or next 1-2 words
        clause_ends_soon = this_word_ends_clause
        if not clause_ends_soon:
            for lookahead in range(1, 3):
                if idx + lookahead < n_items:
                    next_t = (time_stamps[idx + lookahead].text or "").strip()
                    if any(next_t.endswith(p) for p in all_punct):
                        clause_ends_soon = True
                        break

        # Dangling check: does current accumulated text end with a dangling preposition/article/conjunction?
        last_word = cur["text"].split()[-1].lower().rstrip(",.?!;:，。！？；：") if cur["text"].split() else ""
        is_dangling = last_word in _DANGLING_TAILS

        too_long = False
        if max_chars > 0:
            projected_len = len(cur["text"]) + 1 + len(text)
            # Soft tolerance: if a clause/sentence ends soon or last word is dangling, allow up to 25% overshoot
            effective_max = int(max_chars * 1.25) if (sentence_ends_soon or clause_ends_soon or is_dangling) else max_chars
            if projected_len > effective_max:
                if not sentence_ends_soon:
                    too_long = True

        split_by_punct = "punctuation" in split_mode
        split_by_length = "length" in split_mode
        split_by_pause = "pause" in split_mode

        min_clause_len = 10 if any('\u4e00' <= c <= '\u9fff' for c in cur["text"]) else 20

        should_split = False
        if split_by_punct:
            if end_sentence:
                should_split = True
            elif end_clause and (len(cur["text"]) >= min_clause_len or too_long):
                should_split = True
        if split_by_pause and too_far and not should_split:
            should_split = True
        if split_by_length and too_long and not should_split:
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


# -------------------------------------------------------------------------
# Inverse Text Normalization (ITN) & Text Formatting
# -------------------------------------------------------------------------

_CN_NUM = {
    '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
}
_CN_UNIT = {
    '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000
}
_ITN_CACHE = {"mtime": None, "data": None}


def _default_itn_rules():
    return {
        "protected_words": [],
        "custom_replacements": {},
        "enable_number_conversion": True,
        "enable_acronym_cleanup": True,
    }


def _load_itn_rules():
    path = _CURRENT_DIR / "itn_rules.json"
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = None

    cache = _ITN_CACHE
    if cache["data"] is not None and cache["mtime"] == mtime:
        return cache["data"]

    data = _default_itn_rules()
    if mtime is not None:
        try:
            import json
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = {
                    "protected_words": list(loaded.get("protected_words", [])),
                    "custom_replacements": dict(loaded.get("custom_replacements", {})),
                    "enable_number_conversion": bool(loaded.get("enable_number_conversion", True)),
                    "enable_acronym_cleanup": bool(loaded.get("enable_acronym_cleanup", True)),
                }
        except Exception as e:
            print(f"[Qwen3ASR] Failed to read itn_rules.json: {e}")

    cache["mtime"] = mtime
    cache["data"] = data
    return data


def _parse_cn_int(s: str) -> int:
    if not s:
        return 0
    if all(c in _CN_NUM for c in s) and not any(u in _CN_UNIT for u in s):
        return int(''.join(str(_CN_NUM[c]) for c in s))
    total = 0
    r = 0
    for c in s:
        if c in _CN_NUM:
            r = _CN_NUM[c]
        elif c in _CN_UNIT:
            unit = _CN_UNIT[c]
            if unit >= 10000:
                total = (total + (r if r else 1)) * unit
                r = 0
            else:
                total += (r if r else 1) * unit
                r = 0
    total += r
    return total


def _convert_cn_number(match):
    s = match.group(0)
    if s.startswith('百分之'):
        val = s[3:]
        if '点' in val:
            p = val.split('点')
            int_p = _parse_cn_int(p[0]) if p[0] else 0
            dec_p = ''.join(str(_CN_NUM.get(c, c)) for c in p[1])
            return f'{int_p}.{dec_p}%'
        return f'{_parse_cn_int(val)}%'

    if '点' in s:
        parts = s.split('点')
        int_part = _parse_cn_int(parts[0]) if parts[0] else 0
        dec_part = ''.join(str(_CN_NUM.get(c, c)) for c in parts[1])
        return f'{int_part}.{dec_part}'

    return str(_parse_cn_int(s))


def _normalize_text(text: str) -> str:
    if not text:
        return ""

    cfg = _load_itn_rules()

    # 1. Custom user replacements (handles any language or special jargon)
    custom_replacements = cfg.get("custom_replacements") or {}
    for k, v in custom_replacements.items():
        if k.startswith("_"):
            continue
        if re.search(r'[a-zA-Z]', k):
            text = re.sub(re.escape(k), v, text, flags=re.IGNORECASE)
        elif k in text:
            text = text.replace(k, v)

    # 2. Mask protected words from conversion
    protected = cfg.get("protected_words") or []
    placeholders = {}
    for idx, word in enumerate(protected):
        if word in text:
            key = f"__PROT_{idx}__"
            placeholders[key] = word
            text = text.replace(word, key)

    # 3. Chinese number conversion
    if cfg.get("enable_number_conversion", True):
        text = re.sub(r'百分之[零一二两三四五六七八九十百]+(?:点[零一二两三四五六七八九]+)?', _convert_cn_number, text)
        text = re.sub(r'[零一二两三四五六七八九十百千万亿]+点[零一二两三四五六七八九]+', _convert_cn_number, text)
        text = re.sub(r'[零一二两三四五六七八九十百千万亿]{2,}', _convert_cn_number, text)

    # 4. Spaced Latin acronym cleanup (e.g. 'A S R' -> 'ASR', 'U S B' -> 'USB')
    if cfg.get("enable_acronym_cleanup", True):
        text = re.sub(r'(?:(?<=[^a-zA-Z])|^)([a-zA-Z])(?:\s+([a-zA-Z]))+(?=[^a-zA-Z]|$)', 
                      lambda m: m.group(0).replace(' ', ''), text)

    # 5. Restore protected words
    for key, word in placeholders.items():
        text = text.replace(key, word)

    return text


# -------------------------------------------------------------------------
# ComfyUI Nodes
# -------------------------------------------------------------------------

class AILab_Qwen3ASR:
    @classmethod
    def INPUT_TYPES(cls):
        defaults = _get_defaults()
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio input to transcribe."}),
            },
            "optional": {
                "model": (list(_get_model_ids().keys()), {"default": defaults.get("repo_id", "Qwen/Qwen3-ASR-0.6B-hf"), "tooltip": "Choose the ASR model size."}),
                "precision": (["bf16", "fp16", "fp32"], {"default": defaults.get("precision", "bf16"), "tooltip": "Inference precision."}),
                "language": (SUPPORTED_LANGUAGES, {"default": defaults.get("language", "auto"), "tooltip": "Force language or auto-detect."}),
                "hints": ("STRING", {"default": "", "multiline": True, "placeholder": "Optional: Context hotwords, names, or terminology (e.g. Qwen3-ASR)...", "tooltip": "Optional hints/keywords (names, terms) to improve recognition."}),
                "normalize_text": ("BOOLEAN", {"default": True, "tooltip": "Normalize spoken numbers ('一百二十八' -> '128') and acronym spacing ('A S R' -> 'ASR')."}),
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
        model="Qwen/Qwen3-ASR-0.6B-hf",
        precision="bf16",
        language="auto",
        hints="",
        normalize_text=True,
        unload_models=True,
    ):
        device = model_management.get_torch_device()
        dtype = _build_dtype(precision, device)

        source = _get_defaults().get("source", "HuggingFace")
        attention = _get_defaults().get("attention", "auto")

        print(f"[QwenASR] [ASR] Resolving model '{model}' from {source}...")
        model_path = _resolve_model_path(model, source)

        audio_data = _normalize_audio(audio)
        if audio_data is None:
            print("[QwenASR] [ASR] Error: Invalid or missing audio input.")
            return ("",)

        lang = None if language == "auto" else language
        ctx = hints.strip() if isinstance(hints, str) else ""

        print(f"[QwenASR] [ASR] Transcribing audio ({len(audio_data[0]) / float(audio_data[1]):.1f}s, language: {language})...")
        model_runner = _load_cached_asr(model_path, dtype, device, attention, "")
        text, detected_lang, _ = model_runner.transcribe(
            audio_data=audio_data,
            language=lang,
            context=ctx if ctx else None,
            return_time_stamps=False,
        )

        if normalize_text:
            text = _normalize_text(text)

        print(f"[QwenASR] [ASR] Transcription complete: {len(text)} characters (detected: {detected_lang or 'N/A'}).")

        if unload_models:
            _ASR_MODEL_CACHE.clear()
            _ALIGNER_CACHE.clear()
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
                "model": (list(_get_model_ids().keys()), {"default": defaults.get("repo_id", "Qwen/Qwen3-ASR-0.6B-hf"), "tooltip": "Choose the ASR model size."}),
                "precision": (["bf16", "fp16", "fp32"], {"default": defaults.get("precision", "bf16"), "tooltip": "Inference precision."}),
                "attention": (["auto", "flash_attention_2", "sdpa", "eager"], {"default": defaults.get("attention", "auto"), "tooltip": "Attention backend override."}),
                "forced_aligner": (list(_get_aligner_ids().keys()), {"default": defaults.get("forced_aligner", "Qwen/Qwen3-ForcedAligner-0.6B-hf"), "tooltip": "Forced aligner for timestamped subtitles."}),
                "language": (SUPPORTED_LANGUAGES, {"default": defaults.get("language", "auto"), "tooltip": "Force language or auto-detect."}),
                "hints": ("STRING", {"default": "", "multiline": True, "placeholder": "Optional: Context hotwords, names, or terminology (e.g. Qwen3-ASR)...", "tooltip": "Optional hints/keywords (names, terms) to improve recognition."}),
                "output_format": (["none", "txt", "srt"], {"default": "none", "tooltip": "File save format only (does not change subtitle output)."}),
                "output_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional output file path (relative goes to ComfyUI output)."}),
                "split_mode": (["split_by_punctuation_or_pause", "split_by_punctuation_or_pause_or_length", "split_by_punctuation_or_length", "split_by_punctuation", "split_by_pause", "split_by_length"], {"default": "split_by_punctuation_or_pause", "tooltip": "Sentence splitting strategy. 'split_by_punctuation_or_pause' is recommended for natural, readable subtitle clauses without arbitrary character slicing."}),
                "max_gap_sec": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 8.0, "step": 0.1, "tooltip": "Max silence gap to keep the same sentence."}),
                "max_chars": ("INT", {"default": 40, "min": 0, "max": 200, "tooltip": "Optional max characters per line (0 = no limit, 80 recommended if length splitting is enabled)."}),
                "max_inference_batch_size": ("INT", {"default": 32, "min": 1, "max": 256, "tooltip": "Batch size for inference/alignment to avoid OOM."}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 2048, "tooltip": "Max new tokens per chunk."}),
                "normalize_text": ("BOOLEAN", {"default": True, "tooltip": "Normalize spoken numbers ('一百二十八' -> '128') and acronym spacing ('A S R' -> 'ASR')."}),
                "unload_models": ("BOOLEAN", {"default": True, "tooltip": "Unload cached model after inference."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("TEXT", "SUBTITLES", "LANGUAGE", "OUTPUT_PATH")
    FUNCTION = "transcribe"
    CATEGORY = "🧪AILab/🎙️QwenASR"

    def transcribe(
        self,
        audio,
        model="Qwen/Qwen3-ASR-0.6B-hf",
        precision="bf16",
        attention="auto",
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B-hf",
        language="auto",
        hints="",
        output_format="none",
        output_path="",
        split_mode="split_by_punctuation_or_pause",
        max_gap_sec=0.6,
        max_chars=80,
        max_inference_batch_size=32,
        max_new_tokens=256,
        normalize_text=True,
        unload_models=True,
    ):
        device = model_management.get_torch_device()
        dtype = _build_dtype(precision, device)

        source = _get_defaults().get("source", "HuggingFace")

        print(f"[QwenASR] [Subtitle] Resolving models (ASR: {model}, Aligner: {forced_aligner}) from {source}...")
        model_path = _resolve_model_path(model, source)

        forced_aligner_path = ""
        if forced_aligner and forced_aligner != "None":
            forced_aligner_path = _resolve_model_path(forced_aligner, source)

        audio_data = _normalize_audio(audio)
        if audio_data is None:
            print("[QwenASR] [Subtitle] Error: Invalid or missing audio input.")
            return ("", "", "", "")

        lang = None if language == "auto" else language
        ctx = hints.strip() if isinstance(hints, str) else ""

        duration = len(audio_data[0]) / float(audio_data[1])
        print(f"[QwenASR] [Subtitle] Transcribing audio ({duration:.1f}s, language: {language})...")
        model_runner = _load_cached_asr(
            model_path,
            dtype,
            device,
            attention,
            forced_aligner_path,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )
        text, detected_lang, time_stamps = model_runner.transcribe(
            audio_data=audio_data,
            language=lang,
            context=ctx if ctx else None,
            return_time_stamps=bool(forced_aligner_path),
            max_new_tokens=max_new_tokens,
        )

        if time_stamps and text:
            time_stamps = _restore_punctuation(time_stamps, text)

        if normalize_text:
            text = _normalize_text(text)

        subtitles = ""
        file_path = ""
        groups = _group_time_stamps(time_stamps, max_gap_sec=max_gap_sec, max_chars=max_chars, split_mode=split_mode)

        if normalize_text:
            for g in groups:
                g["text"] = _normalize_text(g["text"])

        lines = [f"{g['start']:.2f}-{g['end']:.2f}: {g['text']}" for g in groups]
        subtitles = "\n".join(lines) if lines else text

        if forced_aligner_path and time_stamps:
            print(f"[QwenASR] [Subtitle] Forced alignment generated {len(time_stamps)} word timestamps -> {len(groups)} subtitle lines.")
        elif forced_aligner_path:
            print("[QwenASR] [Subtitle] Notice: Aligner active but no word timestamps were generated (silence or unaligned speech).")
        else:
            print("[QwenASR] [Subtitle] Notice: Forced aligner set to 'None'. Outputting plain text.")

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
                file_content = _build_srt_from_groups(groups) if groups else text
            else:
                file_content = subtitles
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(file_content)
            file_path = out_path

        if unload_models:
            _ASR_MODEL_CACHE.clear()
            _ALIGNER_CACHE.clear()
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
