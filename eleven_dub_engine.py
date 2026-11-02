import os
import re
import io
import json
import time
import base64
import hashlib
import zipfile
import subprocess
from typing import Optional, List, Dict, Any, Callable, Tuple
from datetime import datetime
import requests  # type: ignore
import numpy as np  # type: ignore
import streamlit as st  # type: ignore

from audio_processor import AudioProcessor  # type: ignore
from vision_analyzer import VisionAnalyzer  # type: ignore
from video_editor import AutoVideoEditor  # type: ignore

CACHE_DIR = os.path.join(os.getcwd(), "studio_cache")
THUMB_CACHE_DIR = os.path.join(CACHE_DIR, "thumbnails")
ELEVEN_CACHE_DIR = os.path.join(CACHE_DIR, "elevenlabs_audio")
CLIP_PREVIEW_DIR = os.path.join(CACHE_DIR, "clip_previews")
VI_SUB_CACHE_FILE = os.path.join(CACHE_DIR, "vi_subtitles_cache.json")

for _d in (CACHE_DIR, THUMB_CACHE_DIR, ELEVEN_CACHE_DIR, CLIP_PREVIEW_DIR):
    os.makedirs(_d, exist_ok=True)

DEFAULT_ELEVEN_MODELS = {
    "eleven_v4": "🔥 Eleven v4 (Mới nhất - Nhanh & Cảm xúc nhất, hỗ trợ Audio Tags)",
    "eleven_v4_turbo": "⚡ Eleven v4 Turbo (Tốc độ cao, giữ nguyên cảm xúc v4)",
    "eleven_multilingual_v2": "💎 Eleven Multilingual v2 (Ổn định, chuẩn Phim tài liệu dài)",
    "eleven_v3": "🎭 Eleven v3 (Biểu cảm cao & Hỗ trợ thẻ cảm xúc)",
    "eleven_turbo_v2_5": "🚀 Eleven Turbo v2.5 (Nhanh, mượt & Tiết kiệm 50% Credit)",
    "eleven_flash_v2_5": "💨 Eleven Flash v2.5 (Siêu tốc, tiết kiệm Credit)"
}

ELEVEN_V4_EMOTIVE_TAGS = [
    ("[calm, documentary tone] ", "🍃 [calm, documentary tone]"),
    ("[sighs] ", "😮‍💨 [sighs] (Thở dài)"),
    ("[takes a deep breath] ", "🫁 [takes a deep breath] (Hít sâu)"),
    ("[whispers] ", "🤫 [whispers] (Thì thầm)"),
    ("[excited] ", "🔥 [excited] (Hào hứng)"),
    ("[somber, serious] ", "🎬 [somber, serious] (Trầm lắng)"),
    ("[short pause] ", "⏸️ [short pause] (Ngắt nhịp)")
]

DEFAULT_ELEVEN_VOICES = [
    {"voice_id": "pNInz6obpgDQGcFmaJgB", "name": "Adam", "category": "premade", "gender": "male", "accent": "american", "use_case": "narrative", "description": "deep, resonant documentary", "preview_url": ""},
    {"voice_id": "nPczCjzI2devNBz1zQrb", "name": "Brian", "category": "premade", "gender": "male", "accent": "american", "use_case": "narrative", "description": "deep, resonant and comforting", "preview_url": ""},
    {"voice_id": "JBFqnCBsd6RMkjVDRZzb", "name": "George", "category": "premade", "gender": "male", "accent": "british", "use_case": "narrative", "description": "warm, raspy storyteller", "preview_url": ""},
    {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "category": "premade", "gender": "female", "accent": "american", "use_case": "calm", "description": "calm, matter-of-fact", "preview_url": ""},
    {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Sarah", "category": "premade", "gender": "female", "accent": "american", "use_case": "conversational", "description": "soft, warm, natural", "preview_url": ""},
    {"voice_id": "Xb7hH8MSUJpSbSDYk0k2", "name": "Alice", "category": "premade", "gender": "female", "accent": "british", "use_case": "narrative", "description": "clear, engaging educator", "preview_url": ""}
]

PAUSE_TAG_DURATIONS = {
    "takes a deep breath": 0.75, "deep breath": 0.70, "breathes deeply": 0.75,
    "sighs": 0.60, "sigh": 0.55, "exhales": 0.55,
    "short pause": 0.45, "pause": 0.60, "long pause": 0.90, "clears throat": 0.60
}

MANDATORY_DURATION_WORD_MAP = {
    5: 13, 8: 20, 10: 25, 12: 30, 15: 38, 20: 50, 25: 63, 30: 75, 45: 113, 60: 150, 90: 225, 120: 300
}


def natural_sort_key(file_obj_or_path):
    name = file_obj_or_path.name if hasattr(file_obj_or_path, "name") else os.path.basename(str(file_obj_or_path))
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def format_timestamp_srt(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000))
    total_sec = int(seconds)
    if ms >= 1000:
        total_sec += 1
        ms -= 1000
    return f"{total_sec // 3600:02d}:{(total_sec % 3600) // 60:02d}:{total_sec % 60:02d},{ms:03d}"


def format_timestamp_mmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_sec = int(round(seconds))
    if total_sec >= 3600:
        return f"{total_sec // 3600:02d}:{(total_sec % 3600) // 60:02d}:{total_sec % 60:02d}"
    return f"{total_sec // 60:02d}:{total_sec % 60:02d}"


def parse_timestamp_srt(ts_str: str) -> float:
    s = str(ts_str).strip().replace(",", ".")
    if ":" not in s:
        dash_parts = s.split("-")
        if len(dash_parts) >= 3 and all(p.isdigit() for p in dash_parts[:3]):
            try:
                h, m, sec = float(dash_parts[0]), float(dash_parts[1]), float(dash_parts[2])
                ms = float(dash_parts[3]) / 1000.0 if len(dash_parts) >= 4 and dash_parts[3].isdigit() else 0.0
                return h * 3600.0 + m * 60.0 + sec + ms
            except Exception:
                pass
        try:
            return max(0.0, float(s))
        except Exception:
            return 0.0
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
    except Exception:
        pass
    return 0.0


def clean_eleven_api_key(raw_key: str) -> str:
    return re.sub(r'[^\x20-\x7E]', '', str(raw_key or '')).strip()


def _has_vn_chars(s: str) -> bool:
    return len(re.findall(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', str(s).lower())) >= 2


def _clean_srt_english_line(raw_text: str) -> str:
    s = str(raw_text or "").strip()
    s = re.sub(r'\[\s*giọng\s+điệu\s+bình\s+tĩnh[^\]]*\]', '[calm, documentary tone]', s, flags=re.IGNORECASE)
    s = re.sub(r'^(?:EN|VI|VO\s*\d+)\s*:\s*', '', s, flags=re.IGNORECASE).strip()
    s = re.sub(r'\(\s*(?:Số từ|Target|\d+\s*từ|Actual).*?\)\s*$', '', s, flags=re.IGNORECASE).strip()
    return s


def strip_non_spoken_tags_for_credit_saving(text: str, strip_all_tags: bool = False) -> str:
    s = _clean_srt_english_line(text)
    if strip_all_tags:
        s = re.sub(r'\[[^\]]+\]\s*', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def connect_elevenlabs_api(api_key: str) -> Dict[str, Any]:
    """Đồng bộ Giọng đọc, Hạn mức Credit và Danh sách Model (luôn giữ Eleven v4 ở đầu)."""
    clean_k = clean_eleven_api_key(api_key)
    if not clean_k:
        return {"ok": False, "error": "Chưa nhập ElevenLabs API Key"}
    headers = {"xi-api-key": clean_k}
    res_out: Dict[str, Any] = {
        "ok": False,
        "voices": DEFAULT_ELEVEN_VOICES.copy(),
        "models": DEFAULT_ELEVEN_MODELS.copy(),
        "tier": "PRO",
        "char_used": 0,
        "char_limit": 0,
        "error": ""
    }
    try:
        v_res = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers, timeout=15)
        if v_res.status_code == 200:
            voices_raw = v_res.json().get("voices", [])
            if voices_raw:
                res_out["voices"] = [
                    {
                        "voice_id": v.get("voice_id", ""),
                        "name": v.get("name", "Unknown"),
                        "category": v.get("category", "premade"),
                        "gender": str((v.get("labels") or {}).get("gender", "")).lower(),
                        "accent": str((v.get("labels") or {}).get("accent", "")).lower(),
                        "use_case": str((v.get("labels") or {}).get("use_case") or (v.get("labels") or {}).get("use case") or "").lower(),
                        "description": str((v.get("labels") or {}).get("description") or v.get("description") or "").lower(),
                        "preview_url": v.get("preview_url", "")
                    }
                    for v in voices_raw
                ]
                res_out["ok"] = True
        else:
            res_out["error"] = f"HTTP {v_res.status_code}"
    except Exception as e:
        res_out["error"] = str(e)

    try:
        s_res = requests.get("https://api.elevenlabs.io/v1/user/subscription", headers=headers, timeout=10)
        if s_res.status_code == 200:
            sd = s_res.json()
            res_out["tier"] = str(sd.get("tier", "PRO")).upper()
            res_out["char_used"] = int(sd.get("character_count", 0))
            res_out["char_limit"] = int(sd.get("character_limit", 0))
            res_out["ok"] = True
    except Exception:
        pass

    try:
        m_res = requests.get("https://api.elevenlabs.io/v1/models", headers=headers, timeout=10)
        if m_res.status_code == 200:
            m_list = [m for m in m_res.json() if m.get("can_do_text_to_speech")]
            if m_list:
                synced_models = {
                    "eleven_v4": "🔥 Eleven v4 (Mới nhất - Nhanh & Cảm xúc nhất, hỗ trợ Audio Tags)",
                    "eleven_v4_turbo": "⚡ Eleven v4 Turbo (Tốc độ cao, giữ nguyên cảm xúc v4)"
                }
                for m in m_list:
                    mid = m.get("model_id", "")
                    mname = m.get("name", mid)
                    if mid and mid not in synced_models:
                        synced_models[mid] = f"{mname} ({mid})"
                res_out["models"] = synced_models
    except Exception:
        pass

    return res_out


def clone_elevenlabs_voice_api(
    api_key: str, voice_name: str, gender: str, style: str,
    description: str, audio_file_paths: List[str], remove_noise: bool = True
) -> Dict[str, Any]:
    clean_k = clean_eleven_api_key(api_key)
    if not clean_k:
        return {"ok": False, "error": "Thiếu ElevenLabs API Key"}
    files_payload, handles = [], []
    try:
        for p in audio_file_paths:
            if os.path.exists(p):
                fh = open(p, "rb")
                handles.append(fh)
                files_payload.append(("files", (os.path.basename(p), fh, "audio/mpeg")))
        if not files_payload:
            return {"ok": False, "error": "Không có file âm thanh mẫu"}
        data_p = {
            "name": voice_name,
            "description": description or f"{gender} {style} documentary voice",
            "remove_background_noise": "true" if remove_noise else "false",
            "labels": json.dumps({"gender": gender, "use_case": style, "description": style})
        }
        r = requests.post("https://api.elevenlabs.io/v1/voices/add", headers={"xi-api-key": clean_k}, data=data_p, files=files_payload, timeout=90)
        if r.status_code == 200:
            return {"ok": True, "voice_id": r.json().get("voice_id", "")}
        return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:150]}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        for fh in handles:
            fh.close()


def get_cached_elevenlabs_mp3(text: str, voice_id: str, model_id: str, speed: float, stability: int, similarity: int, style: int) -> str:
    spoken_clean = _clean_srt_english_line(text)
    if not spoken_clean:
        return ""
    for candidate_txt in (spoken_clean, re.sub(r'\[[^\]]+\]\s*', '', spoken_clean).strip()):
        if not candidate_txt:
            continue
        h_exact = f"{candidate_txt}_{voice_id}_{model_id}_{speed:.2f}_{stability}_{similarity}_{style}"
        p_exact_wav = os.path.join(ELEVEN_CACHE_DIR, f"eleven_{hashlib.md5(h_exact.encode('utf-8')).hexdigest()[:16]}.wav")
        if os.path.exists(p_exact_wav) and os.path.getsize(p_exact_wav) > 512:
            return p_exact_wav
        h_text_only = f"{candidate_txt}_{voice_id}"
        p_text = os.path.join(ELEVEN_CACHE_DIR, f"eleven_txt_{hashlib.md5(h_text_only.encode('utf-8')).hexdigest()[:16]}.wav")
        if os.path.exists(p_text) and os.path.getsize(p_text) > 512:
            return p_text
    return ""


def generate_elevenlabs_tts_segment(
    api_key: str, text: str, voice_id: str, model_id: str = "eleven_v4",
    speed: float = 1.0, stability: int = 45, similarity: int = 80,
    style: int = 15, force_regenerate: bool = False, strip_tags_for_credit: bool = False
) -> Dict[str, Any]:
    clean_k = clean_eleven_api_key(api_key)
    if not clean_k:
        return {"ok": False, "error": "Chưa nhập ElevenLabs API Key"}

    api_text = strip_non_spoken_tags_for_credit_saving(text, strip_all_tags=strip_tags_for_credit)
    if not api_text:
        return {"ok": False, "error": "Câu thoại trống"}

    if not force_regenerate:
        existing_p = get_cached_elevenlabs_mp3(api_text, voice_id, model_id, speed, stability, similarity, style)
        if existing_p:
            dur = AutoVideoEditor._get_media_duration_fast(existing_p)
            if dur > 0:
                return {"ok": True, "audio_path": existing_p, "real_duration": round(dur, 3), "cached": True, "chars_billed": 0}

    h_key = f"{api_text}_{voice_id}_{model_id}_{speed:.2f}_{stability}_{similarity}_{style}"
    f_hash = hashlib.md5(h_key.encode("utf-8")).hexdigest()[:16]
    raw_mp3 = os.path.join(ELEVEN_CACHE_DIR, f"raw_{f_hash}.mp3")
    clean_wav = os.path.join(ELEVEN_CACHE_DIR, f"eleven_{f_hash}.wav")
    txt_cache_wav = os.path.join(ELEVEN_CACHE_DIR, f"eleven_txt_{hashlib.md5(f'{api_text}_{voice_id}'.encode('utf-8')).hexdigest()[:16]}.wav")

    effective_model = st.session_state.get("verified_eleven_model_override") or model_id
    payload = {
        "text": api_text,
        "model_id": effective_model,
        "voice_settings": {
            "stability": float(stability) / 100.0,
            "similarity_boost": float(similarity) / 100.0,
            "style": float(style) / 100.0,
            "use_speaker_boost": True,
            "speed": float(np.clip(speed, 0.70, 1.20))
        }
    }
    headers = {"xi-api-key": clean_k, "Content-Type": "application/json", "Accept": "audio/mpeg"}
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=60)
        if res.status_code != 200 and ("v4" in effective_model or "v3" in effective_model):
            st.session_state["verified_eleven_model_override"] = "eleven_multilingual_v2"
            payload["model_id"] = "eleven_multilingual_v2"
            payload["text"] = re.sub(r'\[[^\]]+\]\s*', '', api_text).strip() or api_text
            res = requests.post(url, headers=headers, json=payload, timeout=60)
        if res.status_code != 200:
            return {"ok": False, "error": f"HTTP {res.status_code}: {res.text[:160]}"}
        with open(raw_mp3, "wb") as f:
            f.write(res.content)

        subprocess.run(
            [
                "ffmpeg", "-y", "-i", raw_mp3,
                "-af", "areverse,silenceremove=start_periods=1:start_duration=0.02:start_threshold=-52dB,areverse",
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                clean_wav
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
        )
        if not os.path.exists(clean_wav) or os.path.getsize(clean_wav) < 512:
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw_mp3, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", clean_wav],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
            )
        final_p = clean_wav if (os.path.exists(clean_wav) and os.path.getsize(clean_wav) > 512) else raw_mp3
        try:
            if os.path.exists(clean_wav):
                with open(clean_wav, "rb") as rf, open(txt_cache_wav, "wb") as wf:
                    wf.write(rf.read())
        except Exception:
            pass
        dur = AutoVideoEditor._get_media_duration_fast(final_p)
        return {"ok": True, "audio_path": final_p, "real_duration": round(max(0.2, dur), 3), "cached": False, "chars_billed": len(api_text)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def lock_voice_audio_to_exact_slot(input_audio_path: str, slot_duration_sec: float, stt: int) -> Tuple[str, float]:
    """Co/giãn mượt mà độ dài file Voice bằng FFmpeg atempo để khớp 100% khung giờ kịch bản."""
    if not input_audio_path or not os.path.exists(input_audio_path):
        return input_audio_path, 0.0
    raw_d = AutoVideoEditor._get_media_duration_fast(input_audio_path)
    if raw_d <= 0.1:
        return input_audio_path, round(slot_duration_sec, 2)
    if abs(raw_d - slot_duration_sec) <= 0.06:
        return input_audio_path, round(slot_duration_sec, 2)

    ratio = float(np.clip(raw_d / max(0.5, slot_duration_sec), 0.65, 1.65))
    out_wav = os.path.join(ELEVEN_CACHE_DIR, f"locked_slot_{stt:04d}_{int(slot_duration_sec*10)}_{os.path.basename(input_audio_path)}.wav")
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_audio_path,
            "-filter:a", f"atempo={ratio:.4f}",
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
            out_wav
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
        if os.path.exists(out_wav) and os.path.getsize(out_wav) > 512:
            return out_wav, round(slot_duration_sec, 2)
    except Exception:
        pass
    return input_audio_path, round(raw_d, 2)


def compute_emotion_tags_duration(text: str) -> Tuple[int, float, List[str]]:
    raw_tags = re.findall(r'\[([^\]]+)\]', str(text))
    if not raw_tags:
        return 0, 0.0, []
    total_pause_sec = 0.0
    clean_tag_list = []
    for tg in raw_tags:
        t_low = tg.strip().lower()
        clean_tag_list.append(f"[{tg.strip()}]")
        if "tone" in t_low or "giọng" in t_low:
            continue
        for kw, d_sec in PAUSE_TAG_DURATIONS.items():
            if kw in t_low:
                total_pause_sec += d_sec
                break
    return len(raw_tags), round(total_pause_sec, 2), clean_tag_list


def count_clean_english_words(text: str) -> List[str]:
    no_tags = re.sub(r'\[[^\]]+\]', ' ', str(text))
    cleaned = re.sub(r'[^\w\s\']', ' ', no_tags)
    return [w for w in cleaned.split() if w.strip("'")]


def compute_script_timing_metrics(
    text: str,
    slot_duration_sec: float,
    eleven_speed: float = 1.0,
    target_wps: float = 2.50,
    min_words: int = 1,
    max_words: int = 2000,
    slot_fill_pct: int = 100
) -> Dict[str, Any]:
    tag_count, pause_sec, _ = compute_emotion_tags_duration(text)
    spoken_text = re.sub(r'\[[^\]]+\]', '', str(text)).strip()
    words = count_clean_english_words(spoken_text)
    word_count = len(words)

    effective_wps = max(0.5, float(target_wps) * max(0.70, min(1.20, float(eleven_speed))))
    rounded_slot = int(round(slot_duration_sec))

    if pause_sec == 0.0 and abs(effective_wps - 2.5) <= 0.05 and slot_fill_pct == 100 and rounded_slot in MANDATORY_DURATION_WORD_MAP:
        ideal_max_words = MANDATORY_DURATION_WORD_MAP[rounded_slot]
    else:
        usable_speech_slot = max(0.4, (float(slot_duration_sec) - pause_sec) * (float(slot_fill_pct) / 100.0))
        ideal_max_words = int(np.clip(round(usable_speech_slot * effective_wps), min_words, max_words))

    speech_only_sec = word_count / effective_wps
    est_sec = round(speech_only_sec + pause_sec, 2)
    ms_per_word = int(round((speech_only_sec / max(1, word_count)) * 1000)) if word_count > 0 else int(round(1000.0 / effective_wps))
    diff_sec = round(est_sec - slot_duration_sec, 2)

    is_over = word_count > (ideal_max_words + 2)
    is_under = word_count < (ideal_max_words - 2)

    tag_info_str = f" | 🎭 {tag_count} thẻ ({pause_sec}s)" if tag_count > 0 else ""
    if is_over:
        status = f"⚠️ Dài hơn khung (+{max(0.1, diff_sec):.1f}s | {word_count}/{ideal_max_words} từ{tag_info_str})"
    elif is_under:
        status = f"⚠️ Ngắn hơn khung ({diff_sec:.1f}s | {word_count}/{ideal_max_words} từ{tag_info_str})"
    else:
        status = f"✅ Chuẩn khớp 100% ({slot_duration_sec:.1f}s | {word_count}/{ideal_max_words} từ{tag_info_str})"

    return {
        "word_count": word_count,
        "tag_count": tag_count,
        "emotion_sec": pause_sec,
        "est_sec": est_sec,
        "ms_per_word": ms_per_word,
        "ideal_max_words": ideal_max_words,
        "status": status
    }


def translate_to_vietnamese_subtitle(gemini_key: str, text_en: str, existing_vi: str = "") -> str:
    clean_src = _clean_srt_english_line(re.sub(r'\[[^\]]+\]', '', str(text_en)))
    if not clean_src:
        return "Chưa có nội dung thuyết minh."
    if existing_vi and _has_vn_chars(existing_vi):
        return existing_vi.strip()
    if _has_vn_chars(clean_src):
        return clean_src

    cache_db = {}
    if os.path.exists(VI_SUB_CACHE_FILE):
        try:
            with open(VI_SUB_CACHE_FILE, "r", encoding="utf-8") as f:
                cache_db = json.load(f)
        except Exception:
            cache_db = {}

    if clean_src in cache_db and _has_vn_chars(cache_db[clean_src]):
        return cache_db[clean_src]

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": "en", "tl": "vi", "dt": "t", "q": clean_src},
            headers=headers, timeout=6
        )
        if r.status_code == 200:
            data = r.json()
            if data and isinstance(data[0], list):
                vi_out = "".join(part[0] for part in data[0] if part and part[0]).strip()
                if _has_vn_chars(vi_out):
                    cache_db[clean_src] = vi_out
                    try:
                        with open(VI_SUB_CACHE_FILE, "w", encoding="utf-8") as f:
                            json.dump(cache_db, f, indent=2, ensure_ascii=False)
                    except Exception:
                        pass
                    return vi_out
    except Exception:
        pass
    return clean_src


def create_synced_preview_video_clip(
    video_paths: List[str],
    start_sec: float,
    end_sec: float,
    voice_audio_path: str,
    row_id: int
) -> str:
    if not video_paths:
        return ""
    _, seg_metas = VisionAnalyzer.get_video_timeline_info(video_paths)
    if not seg_metas:
        return ""
    chosen_seg = next((sm for sm in seg_metas if sm["start_global"] <= start_sec <= sm["end_global"]), seg_metas[0])
    local_start = max(0.0, start_sec - chosen_seg["start_global"])
    slot_dur = max(1.0, end_sec - start_sec)
    has_voice = bool(voice_audio_path and os.path.exists(str(voice_audio_path)))
    v_tag = hashlib.md5(f"pv_{chosen_seg['video_path']}_{local_start:.2f}_{slot_dur:.2f}_{voice_audio_path if has_voice else 'no'}".encode()).hexdigest()[:10]
    out_clip = os.path.join(CLIP_PREVIEW_DIR, f"pv_{row_id:04d}_{v_tag}.mp4")
    if os.path.exists(out_clip) and os.path.getsize(out_clip) > 2048:
        return out_clip

    try:
        if has_voice:
            filter_str = (
                "[0:a]aresample=44100,aformat=sample_rates=44100:channel_layouts=stereo,asetpts=PTS-STARTPTS,volume=0.70[a0];"
                "[1:a]aresample=44100,aformat=sample_rates=44100:channel_layouts=stereo,asetpts=PTS-STARTPTS,volume=1.0[a1];"
                "[a0][a1]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
            )
            cmd = [
                "ffmpeg", "-y", "-ss", f"{local_start:.3f}", "-t", f"{slot_dur:.3f}",
                "-i", chosen_seg["video_path"], "-i", str(voice_audio_path),
                "-filter_complex", filter_str, "-map", "0:v:0", "-map", "[aout]",
                "-vf", "scale=640:360:flags=fast_bilinear",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-c:a", "aac", "-b:a", "160k", "-shortest", out_clip
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
        if not os.path.exists(out_clip) or os.path.getsize(out_clip) <= 2048:
            cmd_vo = [
                "ffmpeg", "-y", "-ss", f"{local_start:.3f}", "-t", f"{slot_dur:.3f}",
                "-i", chosen_seg["video_path"],
                *(["-i", str(voice_audio_path), "-map", "0:v:0", "-map", "1:a:0"] if has_voice else []),
                "-vf", "scale=640:360:flags=fast_bilinear",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-c:a", "aac", "-b:a", "160k", "-shortest", out_clip
            ]
            subprocess.run(cmd_vo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
        if os.path.exists(out_clip) and os.path.getsize(out_clip) > 2048:
            return out_clip
    except Exception:
        pass
    return ""


def import_zip_voice_pack_to_rows(zip_file_obj, existing_rows: List[Dict[str, Any]], video_paths: List[str]) -> List[Dict[str, Any]]:
    extracted_files = []
    with zipfile.ZipFile(zip_file_obj, "r") as zf:
        for member in sorted(zf.namelist(), key=natural_sort_key):
            if member.lower().endswith((".mp3", ".wav", ".m4a")):
                safe_n = os.path.basename(member)
                dst_p = os.path.join(ELEVEN_CACHE_DIR, f"zip_{int(time.time())}_{safe_n}")
                with open(dst_p, "wb") as out_f:
                    out_f.write(zf.read(member))
                extracted_files.append((safe_n, dst_p))

    if not extracted_files:
        return existing_rows

    updated_rows = existing_rows.copy() if existing_rows else []
    va = VisionAnalyzer(temp_dir=THUMB_CACHE_DIR)
    curr_t = 0.0
    for idx, (fname, fpath) in enumerate(extracted_files):
        r_dur = round(max(0.5, AutoVideoEditor._get_media_duration_fast(fpath) or 4.0), 3)
        if idx < len(updated_rows):
            slot_d = float(updated_rows[idx].get("Khung Giờ (s)", r_dur))
            locked_p, locked_d = lock_voice_audio_to_exact_slot(fpath, slot_d, idx + 1)
            updated_rows[idx]["audio_path"] = locked_p
            updated_rows[idx]["Voice Thực Tế (s)"] = locked_d
            updated_rows[idx]["Trạng Thái Khớp Nhịp"] = f"✅ Đã nạp từ ZIP ({locked_d}s / {slot_d}s)"
        else:
            m_tc = re.search(r'(\d{2}-\d{2}-\d{2}(?:-\d{3})?)', fname)
            s_sec = parse_timestamp_srt(m_tc.group(1)) if m_tc else curr_t
            e_sec = round(s_sec + r_dur, 3)
            curr_t = e_sec + 1.0
            th_p, st_p = va.extract_slot_filmstrip(video_paths, s_sec, e_sec, idx + 1) if video_paths else ("", "")
            updated_rows.append({
                "STT": idx + 1, "Mốc Bắt Đầu": format_timestamp_srt(s_sec), "Mốc Kết Thúc": format_timestamp_srt(e_sec),
                "Khung Giờ (s)": round(r_dur, 2), "Số Từ": 0, "Tối Đa Từ Chuẩn": 0,
                "Thời Gian Cảm Xúc (s)": 0.0, "Voice Ước Tính (s)": r_dur, "Voice Thực Tế (s)": r_dur,
                "Trạng Thái Khớp Nhịp": f"✅ Đã nạp từ ZIP ({fname})",
                "Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)": f"[Audio từ ZIP] {fname}",
                "Phụ Đề Tiếng Việt (Dễ Hiểu)": f"File âm thanh nạp từ gói ZIP: {fname}",
                "audio_path": fpath, "frame_path": th_p, "strip_path": st_p
            })
    return updated_rows


def render_elevenlabs_dubbing_station_ui(
    saved_cfg: dict,
    gemini_api_key: str,
    elevenlabs_api_key: str,
    uploaded_videos: Any,
    uploaded_audios: Any,
    resolve_or_cache_fn: Callable,
    flatten_files_fn: Callable
) -> Dict[str, Any]:
    def _get_all_loaded_video_paths() -> List[str]:
        act_vids = [p for p in st.session_state.get("direct_video_paths", []) if os.path.exists(p)]
        if uploaded_videos:
            for vf in flatten_files_fn(uploaded_videos):
                pv, _ = resolve_or_cache_fn(vf)
                if pv and os.path.exists(pv) and pv not in act_vids:
                    act_vids.append(pv)
        return act_vids

    el_tab_main, el_tab_clone = st.tabs([
        "🔥 ElevenLabs Studio V2.1 Pro (Eleven v4, Bộ Lọc Giọng, Thẻ Cảm Xúc & Khóa Khớp Mốc Kịch Bản)",
        "🧬 Trạm Clone Giọng Nhanh (Instant Voice Cloning)"
    ])

    with el_tab_main:
        col_left, col_right = st.columns([0.40, 0.60])
        all_el_voices = st.session_state.get("eleven_voices_list", DEFAULT_ELEVEN_VOICES)
        models_map = st.session_state.get("eleven_models_map", DEFAULT_ELEVEN_MODELS)

        with col_left:
            st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
            st.markdown("##### ⚙️ 1. Chọn Model Eleven v4 & Lọc Giọng Đọc")
            m_keys = list(models_map.keys())
            saved_em = saved_cfg.get("eleven_model_id", "eleven_v4")
            eleven_model_id = st.selectbox(
                "Mô hình AI (Model):",
                options=m_keys,
                format_func=lambda k: models_map.get(k, k),
                index=m_keys.index(saved_em) if saved_em in m_keys else 0
            )
            st.caption("✅ Hỗ trợ đầy đủ Eleven v4 Emotive Tags & tự động chuyển sang Multilingual v2 nếu tài khoản chưa mở v4.")

            fc1, fc2 = st.columns(2)
            with fc1:
                flt_gender = st.selectbox("Giới tính:", ["all", "male", "female"], format_func=lambda x: "Tất cả (Nam & Nữ)" if x == "all" else ("👨 Giọng Nam (Male)" if x == "male" else "👩 Giọng Nữ (Female)"))
            with fc2:
                flt_cat = st.selectbox("Nguồn giọng:", ["all", "cloned", "premade"], format_func=lambda x: "Tất cả nguồn" if x == "all" else ("🧬 Giọng đã Clone" if x == "cloned" else "🌟 Giọng Mặc định"))

            fc3, fc4 = st.columns(2)
            with fc3:
                flt_emo = st.selectbox(
                    "Cảm xúc / Phong cách:",
                    ["all", "narrative", "calm", "deep", "raspy", "conversational"],
                    format_func=lambda x: "Mọi cảm xúc" if x == "all" else x.capitalize()
                )
            with fc4:
                flt_search = st.text_input("🔍 Tìm tên / Vùng miền:", value="", placeholder="VD: Adam, Brian...")

            filtered_el_voices = [
                v for v in all_el_voices
                if (flt_gender == "all" or v.get("gender") == flt_gender)
                and (flt_cat == "all" or (flt_cat == "cloned" and v.get("category") == "cloned") or (flt_cat == "premade" and v.get("category") != "cloned"))
                and (flt_emo == "all" or flt_emo in f"{v.get('use_case','')} {v.get('description','')}".lower())
                and (not flt_search.strip() or flt_search.strip().lower() in f"{v.get('name','')} {v.get('accent','')} {v.get('description','')}".lower())
            ] or all_el_voices

            v_ids = [v["voice_id"] for v in filtered_el_voices]
            v_lbls = {
                v["voice_id"]: f"{'🧬 [Clone]' if v.get('category') == 'cloned' else ('👩' if v.get('gender') == 'female' else '👨')} {v['name']} ({v.get('accent') or v.get('use_case') or 'Pro'})"
                for v in filtered_el_voices
            }
            saved_vid = saved_cfg.get("eleven_voice_id", v_ids[0])
            eleven_voice_id = st.selectbox(
                f"Giọng đọc đang chọn ({len(filtered_el_voices)} giọng):",
                options=v_ids,
                format_func=lambda k: v_lbls.get(k, k),
                index=v_ids.index(saved_vid) if saved_vid in v_ids else 0
            )

            chosen_v = next((v for v in filtered_el_voices if v["voice_id"] == eleven_voice_id), None)
            if chosen_v and chosen_v.get("preview_url"):
                st.audio(chosen_v["preview_url"])

            st.markdown("##### 🎛️ 2. Tốc Độ AI & Thanh Trượt Cảm Xúc")
            eleven_speed = st.slider("⚡ Tốc độ nhả chữ AI (Native Speed 0.70x -> 1.20x)", 0.70, 1.20, float(saved_cfg.get("eleven_speed", 1.00)), 0.02)
            s_c1, s_c2, s_c3 = st.columns(3)
            with s_c1:
                eleven_stability = st.slider("Stability (%)", 0, 100, int(saved_cfg.get("eleven_stability", 45)), 5)
            with s_c2:
                eleven_similarity = st.slider("Similarity (%)", 0, 100, int(saved_cfg.get("eleven_similarity", 80)), 5)
            with s_c3:
                eleven_style = st.slider("Style Boost (%)", 0, 100, int(saved_cfg.get("eleven_style", 15)), 5)

            target_wps = st.slider("🗣️ Hệ số tốc độ từ chuẩn (Từ/giây - WPS | 0.5 -> 10.0)", 0.50, 10.00, float(saved_cfg.get("target_wps", 2.50)), 0.05)
            voice_lead_ms = st.slider("⏱️ Độ trễ / sớm khi vào giọng (-5000ms -> +5000ms)", -5000, 5000, int(saved_cfg.get("voice_lead_ms", 0)), 25)
            align_map = {
                "start": "⏮️ Vào giọng ngay đầu khung giờ (Chuẩn SRT)",
                "center": "⏺️ Căn câu Voice vào chính giữa khung giờ",
                "end": "⏭️ Dồn câu Voice về sát cuối khung giờ"
            }
            al_keys = list(align_map.keys())
            saved_al = saved_cfg.get("slot_align_mode", "start")
            slot_align_mode = st.selectbox("Vị trí đặt Voice trong khung giờ:", options=al_keys, format_func=lambda k: align_map[k], index=al_keys.index(saved_al) if saved_al in al_keys else 0)

            keep_v4_emotive_tags_in_api = st.checkbox(
                "🎭 Giữ nguyên Thẻ Cảm Xúc ([sighs], [whispers]...) khi gửi lên Eleven v4",
                value=bool(saved_cfg.get("keep_v4_emotive_tags_in_api", True))
            )
            auto_fit_voice_slot = st.checkbox(
                "🔒 Khóa cứng độ dài từng câu Voice khớp 100% khung giờ kịch bản (FFmpeg Atempo)",
                value=bool(saved_cfg.get("auto_fit_voice_slot", True))
            )
            if st.button("🔄 Áp dụng Giọng & Tốc độ này cho toàn bộ danh sách", use_container_width=True):
                for r_item in st.session_state.get("eleven_script_rows", []):
                    r_item["row_voice_id"] = eleven_voice_id
                    r_item["row_speed"] = eleven_speed
                st.toast("✅ Đã đồng bộ Giọng & Tốc độ cho tất cả các câu!")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
            st.markdown("##### 🎭 Chèn Nhanh Thẻ Đạo Diễn Cảm Xúc (Dành riêng cho Eleven v4 / v3 — nhấp để chèn):")
            tag_cols = st.columns(4)
            for idx_tg, (tag_code, tag_label) in enumerate(ELEVEN_V4_EMOTIVE_TAGS):
                with tag_cols[idx_tg % 4]:
                    if st.button(tag_label, key=f"ins_tag_{idx_tg}", use_container_width=True):
                        cur_buf = st.session_state.get("eleven_raw_text_buffer", "")
                        st.session_state["eleven_raw_text_buffer"] = (cur_buf + " " + tag_code).lstrip()
                        st.rerun()

            raw_script_input = st.text_area(
                "📝 Dán Kịch Bản (.SRT hoặc Văn Bản) Vào Đây — Với Eleven v4 bạn có thể chèn [sighs], [takes a deep breath], [calm, documentary tone]:",
                value=st.session_state.get("eleven_raw_text_buffer", ""),
                height=165,
                placeholder="1\n00:00:05,000 --> 00:00:20,000\n[calm, documentary tone] Standing beside the timber wall, the carpenter FASTENS vertical wooden studs..."
            )
            st.session_state["eleven_raw_text_buffer"] = raw_script_input
            st.caption(f"📊 Tổng ký tự dự kiến: **{len(raw_script_input):,}** ký tự")

            b_c1, b_c2, b_c3 = st.columns(3)
            with b_c1:
                parse_srt_btn = st.button("🔍 1. Bóc Tách Kịch Bản / .SRT Thành Bảng", type="primary", use_container_width=True)
            with b_c2:
                uploaded_zip_pack = st.file_uploader("📦 Nạp gói .ZIP Voice có sẵn:", type=["zip"], label_visibility="collapsed")
            with b_c3:
                if uploaded_zip_pack is not None:
                    if st.button("📥 Giải Nén & Khớp .ZIP", use_container_width=True):
                        p_vids = _get_all_loaded_video_paths()
                        st.session_state["eleven_script_rows"] = import_zip_voice_pack_to_rows(uploaded_zip_pack, st.session_state.get("eleven_script_rows", []), p_vids)
                        st.rerun()

            if parse_srt_btn:
                act_vids = _get_all_loaded_video_paths()
                txt_parse = raw_script_input.strip()
                if not txt_parse and uploaded_audios:
                    for af in flatten_files_fn(uploaded_audios):
                        if af.name.lower().endswith((".srt", ".txt", ".md")):
                            ap_p, _ = resolve_or_cache_fn(af)
                            with open(ap_p, "r", encoding="utf-8", errors="ignore") as f:
                                txt_parse = f.read().strip()
                            st.session_state["eleven_raw_text_buffer"] = txt_parse
                            break

                if not txt_parse:
                    st.warning("⚠️ Vui lòng dán kịch bản vào ô trên hoặc tải file .SRT ở Mục 1!")
                else:
                    tmp_srt = os.path.join(CACHE_DIR, "temp_parse_script.srt")
                    with open(tmp_srt, "w", encoding="utf-8") as f:
                        f.write(txt_parse)
                    parsed_blocks = AudioProcessor().parse_script_file(tmp_srt).get("segments", [])
                    if not parsed_blocks:
                        lines = [ln.strip() for ln in re.split(r'\n+', txt_parse) if ln.strip()]
                        parsed_blocks = []
                        cur_s = 5.0
                        for ln in lines:
                            w_cnt = max(5, len(count_clean_english_words(ln)))
                            dur_s = max(4.0, round(w_cnt / max(0.8, target_wps)))
                            parsed_blocks.append({"start": cur_s, "end": cur_s + dur_s, "duration": dur_s, "text": ln, "vi_text": ""})
                            cur_s += dur_s + 10.0

                    va = VisionAnalyzer(temp_dir=THUMB_CACHE_DIR)
                    rows_built = []
                    for idx_b, blk in enumerate(parsed_blocks):
                        s_t = float(blk["start"])
                        e_t = float(blk["end"])
                        slot_d = round(max(1.0, e_t - s_t), 2)
                        txt_b = _clean_srt_english_line(blk["text"])
                        th_p, st_p = va.extract_slot_filmstrip(act_vids, s_t, e_t, idx_b + 1) if act_vids else ("", "")
                        vi_sub = translate_to_vietnamese_subtitle(gemini_api_key, txt_b, blk.get("vi_text", ""))
                        m_inf = compute_script_timing_metrics(txt_b, slot_d, eleven_speed, target_wps)
                        cached_p = get_cached_elevenlabs_mp3(txt_b, eleven_voice_id, eleven_model_id, eleven_speed, eleven_stability, eleven_similarity, eleven_style)
                        real_d_cached = 0.0
                        if cached_p and auto_fit_voice_slot:
                            cached_p, real_d_cached = lock_voice_audio_to_exact_slot(cached_p, slot_d, idx_b + 1)

                        rows_built.append({
                            "STT": idx_b + 1,
                            "Mốc Bắt Đầu": format_timestamp_srt(s_t),
                            "Mốc Kết Thúc": format_timestamp_srt(e_t),
                            "Khung Giờ (s)": slot_d,
                            "Số Từ": m_inf["word_count"],
                            "Tối Đa Từ Chuẩn": m_inf["ideal_max_words"],
                            "Thời Gian Cảm Xúc (s)": m_inf["emotion_sec"],
                            "Voice Ước Tính (s)": m_inf["est_sec"],
                            "Voice Thực Tế (s)": real_d_cached,
                            "Trạng Thái Khớp Nhịp": m_inf["status"],
                            "Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)": txt_b,
                            "Phụ Đề Tiếng Việt (Dễ Hiểu)": vi_sub,
                            "row_voice_id": eleven_voice_id,
                            "row_speed": eleven_speed,
                            "audio_path": cached_p,
                            "frame_path": th_p,
                            "strip_path": st_p or th_p
                        })
                    st.session_state["eleven_script_rows"] = rows_built
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

        # ======================================================================
        # BẢNG CHỈNH SỬA TRỰC TIẾP & HÀNG ĐỢI TẠO VOICE ELEVEN V4 HÀNG LOẠT
        # ======================================================================
        if st.session_state.get("eleven_script_rows"):
            st.markdown("##### 🎧 Danh Sách Đoạn Thoại & Khung Giờ Kịch Bản (Nhấp Đúp Để Sửa Chữ Hoặc Mốc Giờ)")
            disp_cols = [
                "STT", "Mốc Bắt Đầu", "Mốc Kết Thúc", "Khung Giờ (s)",
                "Số Từ", "Tối Đa Từ Chuẩn", "Voice Thực Tế (s)",
                "Trạng Thái Khớp Nhịp", "Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)", "Phụ Đề Tiếng Việt (Dễ Hiểu)"
            ]
            clean_tb = [{k: r.get(k, "") for k in disp_cols} for r in st.session_state["eleven_script_rows"]]
            edited_sc_tb = st.data_editor(clean_tb, use_container_width=True, height=280, num_rows="dynamic", key="el_v4_table_editor")

            new_rows_sync = []
            for ir, ed_r in enumerate(edited_sc_tb):
                base_r = st.session_state["eleven_script_rows"][ir] if ir < len(st.session_state["eleven_script_rows"]) else {}
                s_new = parse_timestamp_srt(str(ed_r.get("Mốc Bắt Đầu", "00:00:00,000")))
                e_new = parse_timestamp_srt(str(ed_r.get("Mốc Kết Thúc", "00:00:05,000")))
                sl_new = round(max(0.5, e_new - s_new), 2)
                tx_new = str(ed_r.get("Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)", ""))
                vi_new = str(ed_r.get("Phụ Đề Tiếng Việt (Dễ Hiểu)", base_r.get("Phụ Đề Tiếng Việt (Dễ Hiểu)", "")))
                m_u = compute_script_timing_metrics(tx_new, sl_new, eleven_speed, target_wps)
                merged_r = dict(base_r)
                merged_r.update({
                    "STT": ir + 1,
                    "Mốc Bắt Đầu": format_timestamp_srt(s_new),
                    "Mốc Kết Thúc": format_timestamp_srt(e_new),
                    "Khung Giờ (s)": sl_new,
                    "Số Từ": m_u["word_count"],
                    "Tối Đa Từ Chuẩn": m_u["ideal_max_words"],
                    "Thời Gian Cảm Xúc (s)": m_u["emotion_sec"],
                    "Voice Ước Tính (s)": m_u["est_sec"],
                    "Trạng Thái Khớp Nhịp": m_u["status"],
                    "Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)": tx_new,
                    "Phụ Đề Tiếng Việt (Dễ Hiểu)": vi_new
                })
                new_rows_sync.append(merged_r)
            st.session_state["eleven_script_rows"] = new_rows_sync

            q_c1, q_c2, q_c3, q_c4 = st.columns([0.34, 0.24, 0.24, 0.18])
            with q_c1:
                gen_all_btn = st.button("▶️ Tạo Tất Cả Voice Eleven v4 (Chống Trùng Cache)", type="primary", use_container_width=True)
            with q_c2:
                ready_mp3s = [r for r in st.session_state["eleven_script_rows"] if r.get("audio_path") and os.path.exists(str(r["audio_path"]))]
                if ready_mp3s:
                    z_buf = io.BytesIO()
                    with zipfile.ZipFile(z_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for r in ready_mp3s:
                            tc_c = str(r["Mốc Bắt Đầu"]).replace(":", "-").replace(",", "-")
                            ext = os.path.splitext(str(r["audio_path"]))[1] or ".wav"
                            zf.write(r["audio_path"], arcname=f"{int(r['STT']):03d}_{tc_c}{ext}")
                    st.download_button(
                        f"📦 Tải Toàn Bộ {len(ready_mp3s)} Voice (.ZIP)",
                        data=z_buf.getvalue(),
                        file_name=f"ElevenLabs_v4_Audio_Pack_{datetime.now().strftime('%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
            with q_c3:
                srt_lines = [f"{int(r['STT'])}\n{r['Mốc Bắt Đầu']} --> {r['Mốc Kết Thúc']}\n{r['Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)']}\n" for r in st.session_state["eleven_script_rows"]]
                st.download_button("📥 Xuất File Phụ Đề (.SRT)", data="\n".join(srt_lines).encode("utf-8"), file_name="Subtitles_v4.srt", mime="text/plain", use_container_width=True)
            with q_c4:
                if st.button("🗑️ Xóa Bảng", use_container_width=True):
                    st.session_state["eleven_script_rows"] = []
                    st.rerun()

            batch_hud = st.empty()
            if gen_all_btn:
                if not clean_eleven_api_key(elevenlabs_api_key):
                    st.error("⚠️ Vui lòng nhập ElevenLabs API Key ở thanh bên trái trước khi tạo Voice!")
                else:
                    rows_g = st.session_state["eleven_script_rows"]
                    for ig, rg in enumerate(rows_g):
                        slot_d = float(rg["Khung Giờ (s)"])
                        cur_tx = str(rg["Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)"])
                        r_vid = rg.get("row_voice_id") or eleven_voice_id
                        r_spd = float(rg.get("row_speed") or eleven_speed)
                        batch_hud.info(f"⏳ Đang tạo Voice Eleven v4 câu #{rg['STT']}/{len(rows_g)}...")
                        res_tts = generate_elevenlabs_tts_segment(
                            elevenlabs_api_key, cur_tx, r_vid, eleven_model_id,
                            r_spd, eleven_stability, eleven_similarity, eleven_style,
                            force_regenerate=False, strip_tags_for_credit=(not keep_v4_emotive_tags_in_api)
                        )
                        if res_tts.get("ok"):
                            a_final = res_tts["audio_path"]
                            real_d = res_tts["real_duration"]
                            if auto_fit_voice_slot:
                                a_final, real_d = lock_voice_audio_to_exact_slot(a_final, slot_d, int(rg["STT"]))
                            rg["audio_path"] = a_final
                            rg["Voice Thực Tế (s)"] = real_d
                            rg["Trạng Thái Khớp Nhịp"] = f"✅ Đã khóa khớp ({real_d}s / {slot_d}s)"
                        else:
                            rg["Trạng Thái Khớp Nhịp"] = f"❌ Lỗi: {res_tts.get('error', '')}"
                    batch_hud.empty()
                    st.session_state["eleven_script_rows"] = rows_g
                    st.rerun()

            # Chi tiết từng dòng (Nghe thử Audio, Xem ảnh 6 khung hình, Tạo lẻ từng câu)
            with st.expander("🎬 XEM ẢNH 6 KHUNG HÌNH, NGHE THỬ VOICE & CHỈNH RIÊNG TỪNG CÂU", expanded=True):
                act_vids_pv = _get_all_loaded_video_paths()
                for ip, rp in enumerate(st.session_state["eleven_script_rows"]):
                    s_sec_p = parse_timestamp_srt(str(rp["Mốc Bắt Đầu"]))
                    e_sec_p = parse_timestamp_srt(str(rp["Mốc Kết Thúc"]))
                    c1, c2, c3 = st.columns([0.32, 0.46, 0.22])
                    with c1:
                        pv_clip = st.session_state.get(f"pv_clip_path_{ip}", "")
                        if pv_clip and os.path.exists(pv_clip):
                            st.video(pv_clip)
                        elif rp.get("strip_path") and os.path.exists(str(rp["strip_path"])):
                            st.image(rp["strip_path"], use_container_width=True)
                        st.caption(f"🎯 **`{rp['Mốc Bắt Đầu']} --> {rp['Mốc Kết Thúc']}`** (`{rp['Khung Giờ (s)']}s`)")
                    with c2:
                        st.markdown(f"**Câu #{rp['STT']:02d}** — {rp['Trạng Thái Khớp Nhịp']}")
                        st.markdown(
                            f'<div translate="no" class="notranslate" style="background:#0f172a;border:1px solid #6366f1;padding:8px 12px;border-radius:8px;font-size:13.5px;color:#fff;margin-bottom:6px;">'
                            f'🇬🇧 {rp["Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)"]}</div>',
                            unsafe_allow_html=True
                        )
                        st.caption(f"🇻🇳 {rp.get('Phụ Đề Tiếng Việt (Dễ Hiểu)', '')}")
                        if rp.get("audio_path") and os.path.exists(str(rp["audio_path"])):
                            st.audio(rp["audio_path"])
                    with c3:
                        if st.button(f"🎙️ Tạo Lại Audio #{rp['STT']}", key=f"regen_v4_{ip}", use_container_width=True):
                            r1 = generate_elevenlabs_tts_segment(
                                elevenlabs_api_key, rp["Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)"],
                                rp.get("row_voice_id") or eleven_voice_id, eleven_model_id,
                                float(rp.get("row_speed") or eleven_speed), eleven_stability,
                                eleven_similarity, eleven_style, force_regenerate=True,
                                strip_tags_for_credit=(not keep_v4_emotive_tags_in_api)
                            )
                            if r1.get("ok"):
                                a_l, d_l = (
                                    lock_voice_audio_to_exact_slot(r1["audio_path"], float(rp["Khung Giờ (s)"]), int(rp["STT"]))
                                    if auto_fit_voice_slot else (r1["audio_path"], r1["real_duration"])
                                )
                                rp["audio_path"], rp["Voice Thực Tế (s)"] = a_l, d_l
                                rp["Trạng Thái Khớp Nhịp"] = f"✅ Đã tạo mới ({d_l}s / {rp['Khung Giờ (s)']}s)"
                                st.rerun()
                        if act_vids_pv and st.button(f"🎬 Xem Thử Đoạn #{rp['STT']}", key=f"pv_v4_{ip}", use_container_width=True):
                            st.session_state[f"pv_clip_path_{ip}"] = create_synced_preview_video_clip(
                                act_vids_pv, s_sec_p, e_sec_p, rp.get("audio_path", ""), int(rp["STT"])
                            )
                            st.rerun()
                        if rp.get("audio_path") and os.path.exists(str(rp["audio_path"])):
                            with open(rp["audio_path"], "rb") as fa:
                                st.download_button(
                                    f"⬇️ Tải Audio #{rp['STT']}",
                                    data=fa,
                                    file_name=f"{int(rp['STT']):03d}_{str(rp['Mốc Bắt Đầu']).replace(':','-')}.wav",
                                    mime="audio/wav",
                                    key=f"dl_v4_{ip}",
                                    use_container_width=True
                                )
                    st.markdown("<hr style='margin:6px 0;border-color:rgba(99,102,241,0.2);'>", unsafe_allow_html=True)

    with el_tab_clone:
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.markdown("##### 🧬 Trạm Clone Giọng Nhanh (Instant Voice Cloning)")
        clc1, clc2, clc3 = st.columns(3)
        with clc1:
            clone_name = st.text_input("Tên giọng mới (Bắt buộc) *:", placeholder="VD: Quantum Narrator - Nam Trầm Documentary")
        with clc2:
            clone_gender = st.selectbox("Nhãn Giới tính:", ["male", "female"])
        with clc3:
            clone_style = st.selectbox("Nhãn Phong cách / Cảm xúc:", ["narrative", "calm", "deep", "raspy", "conversational"])
        clone_desc = st.text_input("Mô tả đặc điểm giọng (Tùy chọn):", placeholder="VD: Deep American male voice for survival documentary...")
        clone_rm_noise = st.checkbox("Tự động lọc sạch tạp âm / nhạc nền nhỏ (Remove Background Noise)", value=True)
        clone_files_up = st.file_uploader("Chọn File âm thanh mẫu (.mp3, .wav, .m4a - Tốt nhất dài 1-3 phút) *:", type=["mp3", "wav", "m4a"], accept_multiple_files=True)
        if st.button("🚀 Bắt Đầu Clone Giọng Mới Lên ElevenLabs", type="primary", use_container_width=True):
            if not clone_name.strip() or not clone_files_up:
                st.warning("⚠️ Vui lòng nhập tên giọng và tải lên ít nhất 1 file âm thanh mẫu!")
            else:
                s_paths = [resolve_or_cache_fn(cf)[0] for cf in clone_files_up]
                r_cl = clone_elevenlabs_voice_api(elevenlabs_api_key, clone_name.strip(), clone_gender, clone_style, clone_desc.strip(), s_paths, clone_rm_noise)
                if r_cl.get("ok"):
                    st.success(f"✅ Clone thành công giọng '{clone_name}'!")
                    c_new = connect_elevenlabs_api(elevenlabs_api_key)
                    if c_new.get("ok"):
                        st.session_state["eleven_voices_list"] = c_new["voices"]
                else:
                    st.error(f"❌ Lỗi Clone: {r_cl.get('error')}")
        st.markdown('</div>', unsafe_allow_html=True)

    return {
        "eleven_model_id": eleven_model_id,
        "eleven_voice_id": eleven_voice_id,
        "eleven_speed": eleven_speed,
        "eleven_stability": eleven_stability,
        "eleven_similarity": eleven_similarity,
        "eleven_style": eleven_style,
        "target_wps": target_wps,
        "voice_lead_ms": voice_lead_ms,
        "slot_align_mode": slot_align_mode,
        "keep_v4_emotive_tags_in_api": keep_v4_emotive_tags_in_api,
        "auto_fit_voice_slot": auto_fit_voice_slot
    }