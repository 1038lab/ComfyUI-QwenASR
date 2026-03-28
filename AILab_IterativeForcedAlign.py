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
    _load_cached_aligner,
    _ALIGNER_CACHE,
    Qwen3ForcedAligner,
    _ALIGNER_IMPORT_ERROR,
)


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
            aligner_choices = ["Qwen/Qwen3-ForcedAligner-0.6B"]
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio input to align."}),
                "text": ("STRING", {"default": "", "multiline": True, "tooltip": "Known transcript text to force-align against the audio."}),
                "language": ([lang for lang in SUPPORTED_LANGUAGES if lang != "auto"], {"default": "English", "tooltip": "Language of the transcript."}),
            },
            "optional": {
                "forced_aligner": (aligner_choices, {"default": defaults.get("forced_aligner", "Qwen/Qwen3-ForcedAligner-0.6B"), "tooltip": "Forced aligner model."}),
                "precision": (["bf16", "fp16", "fp32"], {"default": defaults.get("precision", "bf16"), "tooltip": "Inference precision."}),
                "attention": (["auto", "flash_attention_2", "sdpa", "eager"], {"default": defaults.get("attention", "auto"), "tooltip": "Attention backend override."}),
                "chunk_audio_sec": ("INT", {"default": 240, "min": 60, "max": 300, "step": 10, "tooltip": "Audio window size per iteration (seconds). Must be under the model's 300s limit."}),
                "min_tail_sec": ("INT", {"default": 60, "min": 10, "max": 120, "step": 5, "tooltip": "Minimum seconds of empty audio after the last word. Larger = safer but more iterations."}),
                "backoff_words": ("INT", {"default": 15, "min": 3, "max": 50, "step": 1, "tooltip": "Words to back off from the end of each chunk to avoid edge effects."}),
                "unload_models": ("BOOLEAN", {"default": True, "tooltip": "Unload cached aligner model after inference."}),
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
        language="English",
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        precision="bf16",
        attention="auto",
        chunk_audio_sec=240,
        min_tail_sec=60,
        backoff_words=15,
        unload_models=True,
    ):
        if Qwen3ForcedAligner is None:
            raise RuntimeError(f"Qwen3ForcedAligner not available: {_ALIGNER_IMPORT_ERROR}")

        text = (text or "").strip()
        if not text:
            return ("",)

        device = model_management.get_torch_device()
        dtype = _build_dtype(precision, device)
        source = _get_defaults().get("source", "HuggingFace")

        audio_data = _normalize_audio(audio)
        if audio_data is None:
            return ("",)

        wave, sr = audio_data
        total_duration = len(wave) / sr
        total_samples = len(wave)
        words = text.split()

        aligner_path = _resolve_model_path(forced_aligner, source)
        aligner = _load_cached_aligner(aligner_path, dtype, device, attention)

        # Short audio — single pass
        if total_duration <= chunk_audio_sec:
            results = aligner.align(audio=audio_data, text=text, language=language)
            all_items = list(results[0])
        else:
            # Speaking rate for this script
            words_per_sec = len(words) / total_duration
            # How many words fit in the "safe zone" (chunk minus tail buffer)
            target_words = int((chunk_audio_sec - min_tail_sec) * words_per_sec)
            target_words = max(target_words, 20)

            print(f"[IterativeForcedAlign] {len(words)} words, {total_duration:.1f}s audio")
            print(f"  Speaking rate: {words_per_sec:.2f} words/sec")
            print(f"  Target words per chunk: {target_words} ({chunk_audio_sec}s audio - {min_tail_sec}s tail)")

            all_items = []
            word_cursor = 0
            audio_cursor_sec = 0.0

            iteration = 0
            while word_cursor < len(words):
                iteration += 1
                remaining_words = len(words) - word_cursor
                remaining_audio = total_duration - audio_cursor_sec

                # Last chunk: remaining words fit within the audio, just align them all
                is_last = remaining_words <= target_words or remaining_audio <= chunk_audio_sec

                if is_last:
                    chunk_start_sample = int(round(audio_cursor_sec * sr))
                    chunk_wav = wave[chunk_start_sample:]
                    chunk_text = " ".join(words[word_cursor:])

                    print(f"  Iter {iteration} (final): audio {audio_cursor_sec:.1f}s–{total_duration:.1f}s, {remaining_words} words")

                    chunk_results = aligner.align(
                        audio=(chunk_wav, sr),
                        text=chunk_text,
                        language=language,
                    )

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
                chunk_text = " ".join(chunk_words)

                print(f"  Iter {iteration}: audio {audio_cursor_sec:.1f}s–{audio_cursor_sec + chunk_duration:.1f}s ({chunk_duration:.0f}s), words [{word_cursor}:{word_end}] ({len(chunk_words)} words)")

                chunk_results = aligner.align(
                    audio=(chunk_wav, sr),
                    text=chunk_text,
                    language=language,
                )

                items = list(chunk_results[0])

                # Keep all words except the last backoff_words
                keep_count = max(len(items) - backoff_words, 1)

                for item in items[:keep_count]:
                    all_items.append(type(item)(
                        text=item.text,
                        start_time=round(item.start_time + audio_cursor_sec, 3),
                        end_time=round(item.end_time + audio_cursor_sec, 3),
                    ))

                # Anchor: last kept word's end timestamp.
                # Next chunk starts audio here — no overlap, no re-alignment
                # of already-committed words.
                last_kept = items[keep_count - 1]
                anchor_time = round(last_kept.end_time + audio_cursor_sec, 3)

                # Advance cursors
                word_cursor = word_cursor + keep_count
                audio_cursor_sec = max(anchor_time, audio_cursor_sec + 1.0)

                print(f"    Kept {keep_count}/{len(items)} words, next audio from {audio_cursor_sec:.1f}s")

            print(f"[IterativeForcedAlign] Done: {len(all_items)} timestamps in {iteration} iterations")

        # Format output
        lines = []
        for item in all_items:
            word = (item.text or "").strip()
            if word:
                lines.append(f"{item.start_time:.2f}-{item.end_time:.2f}: {word}")
        word_timestamps = "\n".join(lines)

        if unload_models:
            _ALIGNER_CACHE.clear()
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
