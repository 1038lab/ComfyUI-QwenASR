# Iterative chunking forced alignment for long audio.
#
# Feeds the aligner large audio windows with deliberately FEWER words
# than the audio contains, guaranteeing a tail of empty audio.
# This keeps the aligner in "too few words" mode — the safe direction
# where every word gets an accurate timestamp.
#
# The word count per chunk is derived from the script's actual speaking rate
# (total_words / total_duration), minus a configurable tail buffer.
# Each iteration's last word timestamp anchors the next chunk's start.

import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import comfy.model_management as model_management

_CURRENT_DIR = Path(__file__).parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from AILab_QwenASR import (
    SUPPORTED_LANGUAGES,
    _get_defaults,
    _get_aligner_ids,
    _build_dtype,
    _resolve_model_path,
    _normalize_audio,
    _normalize_text,
    _load_cached_aligner,
    _load_cached_asr,
    _ALIGNER_CACHE,
    _ASR_MODEL_CACHE,
    WordTimestamp,
    _restore_punctuation,
)


def _detect_text_language(text: str) -> str:
    if not text:
        return "English"
    import re
    if re.search(r"[\u4e00-\u9fff]", text):
        return "Chinese"
    if re.search(r"[\u3040-\u30ff]", text):
        return "Japanese"
    if re.search(r"[\uac00-\ud7af]", text):
        return "Korean"
    if re.search(r"[\u0400-\u04ff]", text):
        return "Russian"
    return "English"


def _split_words(aligner, text: str, language: str) -> List[str]:
    processor = getattr(aligner, "processor", None)
    if processor is not None and hasattr(processor, "split_words_for_alignment"):
        try:
            return processor.split_words_for_alignment(text, language)
        except Exception:
            pass
    import re
    if language in ("Chinese", "Cantonese", "Japanese"):
        tokens = re.findall(r"[\u4e00-\u9fff\u3040-\u30ff]|[a-zA-Z0-9']+|[^\s\w]", text)
        return [t.strip() for t in tokens if t.strip()]
    return text.split()


def _join_words(words: List[str], language: str) -> str:
    if language in ("Chinese", "Cantonese", "Japanese"):
        return "".join(words)
    return " ".join(words)


class AILab_Qwen3ForcedAlign:
    """
    Iterative chunking forced alignment for long audio with a known transcript.

    Always feeds fewer words than the audio contains, using the script's own
    speaking rate to estimate how many words fit in (chunk_duration - tail_buffer).
    The aligner accurately timestamps every word, and the last word's position
    anchors the next iteration.
    """
    @classmethod
    def INPUT_TYPES(cls):
        defaults = _get_defaults()
        aligner_choices = [k for k in _get_aligner_ids().keys() if k != "None"]
        if not aligner_choices:
            aligner_choices = ["Qwen/Qwen3-ForcedAligner-0.6B-hf"]
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio input to align."}),
            },
            "optional": {
                "text": ("STRING", {"default": "", "multiline": True, "placeholder": "Optional: Enter transcript to align, or leave blank to auto-transcribe speech...", "tooltip": "Known transcript text to force-align against the audio. If left blank, speech is auto-transcribed first."}),
                "language": (SUPPORTED_LANGUAGES, {"default": defaults.get("language", "auto"), "tooltip": "Language of the transcript. 'auto' detects from audio/text."}),
                "forced_aligner": (aligner_choices, {"default": defaults.get("forced_aligner", "Qwen/Qwen3-ForcedAligner-0.6B-hf"), "tooltip": "Forced aligner model."}),
                "precision": (["bf16", "fp16", "fp32"], {"default": defaults.get("precision", "bf16"), "tooltip": "Inference precision."}),
                "attention": (["auto", "flash_attention_2", "sdpa", "eager"], {"default": defaults.get("attention", "auto"), "tooltip": "Attention backend override."}),
                "chunk_audio_sec": ("INT", {"default": 240, "min": 60, "max": 300, "step": 10, "tooltip": "Audio window size per iteration (seconds). Must be under the model's 300s limit."}),
                "min_tail_sec": ("INT", {"default": 60, "min": 10, "max": 120, "step": 5, "tooltip": "Minimum seconds of empty audio after the last word. Larger = safer but more iterations."}),
                "backoff_words": ("INT", {"default": 15, "min": 3, "max": 50, "step": 1, "tooltip": "Words to back off from the end of each chunk to avoid edge effects."}),
                "normalize_text": ("BOOLEAN", {"default": True, "tooltip": "Normalize numbers ('一百二十八' -> '128') and acronym spacing ('A S R' -> 'ASR') in output timestamps."}),
                "unload_models": ("BOOLEAN", {"default": True, "tooltip": "Unload cached models after inference."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("WORD_TIMESTAMPS",)
    FUNCTION = "align"
    CATEGORY = "🧪AILab/🎙️QwenASR"

    def align(
        self,
        audio,
        text="",
        language="auto",
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B-hf",
        precision="bf16",
        attention="auto",
        chunk_audio_sec=240,
        min_tail_sec=60,
        backoff_words=15,
        normalize_text=True,
        unload_models=True,
    ):
        source = _get_defaults().get("source", "HuggingFace")
        audio_data = _normalize_audio(audio)
        if audio_data is None:
            print("[QwenASR] [ForcedAlign] Error: Invalid or missing audio input.")
            return ("",)

        device = model_management.get_torch_device()
        dtype = _build_dtype(precision, device)

        wave, sr = audio_data
        total_duration = len(wave) / float(sr)
        total_samples = len(wave)

        # 1. Resolve & load forced aligner model first (triggers download on first run)
        print(f"[QwenASR] [ForcedAlign] Resolving aligner model '{forced_aligner}' from {source}...")
        aligner_path = _resolve_model_path(forced_aligner, source)
        aligner = _load_cached_aligner(aligner_path, dtype, device, attention)

        # 2. Handle transcript: auto-transcribe if empty
        text = (text or "").strip()
        if not text:
            print("[QwenASR] [ForcedAlign] No text transcript provided. Auto-transcribing audio first with Qwen3-ASR...")
            default_asr = _get_defaults().get("repo_id", "Qwen/Qwen3-ASR-0.6B-hf")
            asr_path = _resolve_model_path(default_asr, source)
            asr_model = _load_cached_asr(asr_path, dtype, device, attention)
            trans_lang = None if language == "auto" else language
            auto_text, detected_lang, _ = asr_model.transcribe(
                audio_data=audio_data,
                language=trans_lang,
            )
            text = (auto_text or "").strip()
            if not text:
                print("[QwenASR] [ForcedAlign] Transcription yielded empty speech. Returning empty timestamps.")
                return ("",)
            if language == "auto":
                language = detected_lang or "English"
            print(f"[QwenASR] [ForcedAlign] Auto-transcription completed ({len(text)} chars, language: {language})")

        if language == "auto":
            language = _detect_text_language(text)

        print(f"[QwenASR] [ForcedAlign] Starting alignment (duration: {total_duration:.1f}s, language: {language})...")

        # 3. Tokenize words with language awareness
        words = _split_words(aligner, text, language)
        if not words:
            words = text.split()
        if not words:
            words = [text]

        # 4. Short audio — single pass
        if total_duration <= chunk_audio_sec:
            results = aligner.align(audio=audio_data, text=text, language=language)
            all_items = list(results[0]) if results else []
        else:
            # Speaking rate for this script
            words_per_sec = len(words) / total_duration
            target_words = int((chunk_audio_sec - min_tail_sec) * words_per_sec)
            target_words = max(target_words, 20)

            all_items = []
            word_cursor = 0
            audio_cursor_sec = 0.0

            iteration = 0
            while word_cursor < len(words):
                iteration += 1
                remaining_words = len(words) - word_cursor
                remaining_audio = total_duration - audio_cursor_sec

                is_last = remaining_words <= target_words or remaining_audio <= chunk_audio_sec

                if is_last:
                    chunk_start_sample = int(round(audio_cursor_sec * sr))
                    chunk_wav = wave[chunk_start_sample:]
                    chunk_text = _join_words(words[word_cursor:], language)

                    chunk_results = aligner.align(
                        audio=(chunk_wav, sr),
                        text=chunk_text,
                        language=language,
                    )

                    if chunk_results:
                        for item in chunk_results[0]:
                            all_items.append(type(item)(
                                text=item.text,
                                start_time=round(item.start_time + audio_cursor_sec, 3),
                                end_time=round(item.end_time + audio_cursor_sec, 3),
                            ))
                    break

                # Normal chunk: big audio window, fewer words
                chunk_start_sample = int(round(audio_cursor_sec * sr))
                chunk_end_sample = min(chunk_start_sample + int(round(chunk_audio_sec * sr)), total_samples)
                chunk_wav = wave[chunk_start_sample:chunk_end_sample]
                chunk_duration = len(chunk_wav) / float(sr)

                word_end = min(word_cursor + target_words, len(words))
                chunk_words = words[word_cursor:word_end]
                chunk_text = _join_words(chunk_words, language)

                chunk_results = aligner.align(
                    audio=(chunk_wav, sr),
                    text=chunk_text,
                    language=language,
                )

                items = list(chunk_results[0]) if chunk_results else []

                if items:
                    keep_count = min(max(len(items) - backoff_words, 1), len(items))

                    for item in items[:keep_count]:
                        all_items.append(type(item)(
                            text=item.text,
                            start_time=round(item.start_time + audio_cursor_sec, 3),
                            end_time=round(item.end_time + audio_cursor_sec, 3),
                        ))

                    last_kept = items[keep_count - 1]
                    anchor_time = round(last_kept.end_time + audio_cursor_sec, 3)
                    word_cursor = word_cursor + keep_count
                    audio_cursor_sec = max(anchor_time, audio_cursor_sec + 1.0)
                else:
                    word_cursor += max(target_words // 2, 1)
                    audio_cursor_sec += chunk_duration

        if all_items and transcript_text:
            all_items = _restore_punctuation(all_items, transcript_text)

        # Format output
        lines = []
        for item in all_items:
            word = (item.text or "").strip()
            if word:
                if normalize_text:
                    word = _normalize_text(word)
                lines.append(f"{item.start_time:.2f}-{item.end_time:.2f}: {word}")
        word_timestamps = "\n".join(lines)

        print(f"[QwenASR] [ForcedAlign] Alignment complete: {len(all_items)} word timestamps generated.")

        if unload_models:
            _ALIGNER_CACHE.clear()
            _ASR_MODEL_CACHE.clear()
            try:
                model_management.soft_empty_cache()
            except Exception:
                pass

        return (word_timestamps,)


NODE_CLASS_MAPPINGS = {
    "AILab_Qwen3ForcedAlign": AILab_Qwen3ForcedAlign,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_Qwen3ForcedAlign": "Forced Align (QwenASR)",
}
