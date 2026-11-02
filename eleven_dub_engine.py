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

MANDATORY_DURATION_WORD_MAP = {8: 20, 10: 25, 12: 30, 15: 38, 20: 50}

VALID_ACTION_VERBS = {
    "ADJUSTS", "ALIGNS", "ANCHORS", "APPLIES", "ASSEMBLES", "ATTACHES", "BINDS", "BOLTS",
    "BRACES", "BUILDS", "CARRIES", "CARVES", "CAULKS", "CHECKS", "CHISELS", "CHOPS",
    "CLADS", "CLAMPS", "CLEANS", "CLEARS", "CLIMBS", "COMPACTS", "CONNECTS", "COVERS",
    "CUTS", "DIGS", "DRILLS", "DRIVES", "ELEVATES", "ERECTS", "EXCAVATES", "FASTENS",
    "FILLS", "FINISHES", "FITS", "FIXES", "FRAMES", "GRADES", "GRINDS", "GUIDES",
    "HAMMERS", "HANGS", "HAULS", "HEATS", "HEWS", "HOLDS", "HOISTS", "IGNITES",
    "INSERTS", "INSTALLS", "INSULATES", "JOINS", "KNEADS", "LAYERS", "LAYS", "LEANS",
    "LEVELS", "LIFTS", "LINES", "LOCKS", "LOWERS", "MARKS", "MEASURES", "MIXES",
    "MOUNTS", "MOVES", "NAILS", "NOTCHES", "OPENS", "PACKS", "PAINTS", "PINS",
    "PLACES", "PLANES", "POSITIONS", "POURS", "PRESSES", "PULLS", "PUSHES", "RAISES",
    "RAMS", "REINFORCES", "REMOVES", "ROLLS", "SANDS", "SAWS", "SCREWS", "SEALS",
    "SECURES", "SETS", "SHAPES", "SHEATHES", "SLIDES", "SMOOTHES", "SPREADS", "STACKS",
    "STAKES", "STEPS", "STRIKES", "STRIPS", "SWINGS", "TAMPS", "TAPES", "TIGHTENS",
    "TRIMS", "TURNS", "UNROLLS", "WALKS", "WELDS", "WRAPS"
}

EXACT_GRAMMAR_EXTENSIONS = {
    1: ["securely.", "accurately.", "completely.", "methodically.", "precisely."],
    2: ["for stability.", "with precision.", "without gaps.", "for strength.", "under load."],
    3: ["for lasting strength.", "with true accuracy.", "across the frame.", "without any gaps.", "for solid support."],
    4: ["to ensure lasting stability.", "across the entire platform.", "with clean carpenter precision.", "for the next stage.", "against heavy winter loads."],
    5: ["to keep the structure level.", "across the entire wooden deck.", "for maximum strength and durability.", "before moving to the corner.", "so every joint stays tight."],
    6: ["so every joint remains rock solid.", "to keep the entire platform level.", "before fastening the next timber section.", "ensuring a tight and weatherproof fit.", "across the entire framed working deck."],
    7: ["so the entire wooden frame stays square.", "to prevent any movement under heavy loads.", "while keeping every single board completely flush.", "before securing the next upright wall post."],
    8: ["so the entire structural assembly remains completely level.", "to ensure every timber joint locks together tightly.", "while checking that each board sits perfectly flush.", "before moving along to the next framing bay."],
    9: ["so the finished timber platform supports heavy loads safely.", "to make sure every corner joint stays completely square.", "while keeping the entire wooden frame plumb and aligned."],
    10: ["so the entire wooden structure remains solid, square, and level.", "while verifying that every fastened joint holds tight under load."],
    11: ["so the entire framed timber assembly stays level, square, and rock solid."],
    12: ["so the entire wooden frame remains completely level, square, and weatherproof year round."],
    13: ["so every single fastened timber joint stays tight, level, and square across the deck."],
    14: ["so the entire framed wooden platform remains completely level, square, and solid under heavy loads."],
    15: ["so every structural timber joint locks together cleanly and keeps the entire cabin frame rock solid."]
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


def strip_non_spoken_tags_for_credit_saving(text: str, strip_all_tags: bool = True) -> str:
    s = _clean_srt_english_line(text)
    if strip_all_tags:
        s = re.sub(r'\[[^\]]+\]\s*', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def render_live_task_hud(slot_placeholder: Any, pct: float, task_title: str, step_detail: str, start_t: float):
    pct_clamped = max(1, min(100, int(round(pct * 100))))
    elapsed = round(time.time() - start_t, 1)
    bar_color = "#4ade80" if pct_clamped >= 100 else "#6366f1"
    hud_html = f"""
    <div translate="no" class="notranslate" style="background:linear-gradient(90deg,#0f172a,#312e81);border:2px solid {bar_color};border-radius:12px;padding:10px 16px;margin:8px 0;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-weight:900;color:#facc15;font-size:14px;">⚙️ {task_title}</span>
            <span style="font-weight:900;color:{bar_color};font-size:18px;font-family:Consolas,monospace;">{pct_clamped}% ({elapsed}s)</span>
        </div>
        <div style="width:100%;height:10px;background:#020617;border-radius:999px;overflow:hidden;">
            <div style="width:{pct_clamped}%;height:100%;background:linear-gradient(90deg,#4f46e5,{bar_color});"></div>
        </div>
        <div style="font-size:12.5px;color:#e2e8f0;margin-top:5px;font-weight:600;">👉 {step_detail}</div>
    </div>
    """
    slot_placeholder.markdown(hud_html, unsafe_allow_html=True)


# ==============================================================================
# KẾT NỐI ELEVENLABS API (ĐỒNG BỘ ELEVEN v4 + VOICES + SUBSCRIPTION)
# ==============================================================================

def connect_elevenlabs_api(api_key: str) -> Dict[str, Any]:
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

    # 1. Tải danh sách Giọng đọc (Voices)
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

    # 2. Tải thông tin Gói cước & Credit (Subscription)
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

    # 3. Đồng bộ danh sách Models từ API (Luôn giữ Eleven v4 & Eleven v4 Turbo ở đầu như Eleven Labstool.html)
    try:
        m_res = requests.get("https://api.elevenlabs.io/v1/models", headers=headers, timeout=10)
        if m_res.status_code == 200:
            models_data = m_res.json()
            if isinstance(models_data, list):
                tts_models = [m for m in models_data if m.get("can_do_text_to_speech")]
                synced_map: Dict[str, str] = {
                    "eleven_v4": DEFAULT_ELEVEN_MODELS["eleven_v4"],
                    "eleven_v4_turbo": DEFAULT_ELEVEN_MODELS["eleven_v4_turbo"]
                }
                for m in tts_models:
                    mid = str(m.get("model_id", "")).strip()
                    mname = str(m.get("name", mid)).strip()
                    if mid:
                        synced_map[mid] = DEFAULT_ELEVEN_MODELS.get(mid, f"🎙️ {mname} ({mid})")
                for k_def, v_def in DEFAULT_ELEVEN_MODELS.items():
                    if k_def not in synced_map:
                        synced_map[k_def] = v_def
                res_out["models"] = synced_map
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
    spoken_clean = strip_non_spoken_tags_for_credit_saving(text, strip_all_tags=True)
    if not spoken_clean:
        return ""
    for candidate_txt in (str(text).strip(), spoken_clean):
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
    """
    Tạo Voice với Eleven v4 / v4 Turbo / v3 (Hỗ trợ đầy đủ Emotive Audio Tags như [sighs], [takes a deep breath]).
    Nếu tài khoản chưa mở khóa eleven_v4, tự động chuyển mượt sang eleven_multilingual_v2 giống hệt Eleven Labstool.html.
    """
    clean_k = clean_eleven_api_key(api_key)
    if not clean_k:
        return {"ok": False, "error": "Chưa nhập ElevenLabs API Key"}

    is_emotive_model = any(tag in str(model_id).lower() for tag in ("v4", "v3"))
    should_strip = strip_tags_for_credit or (not is_emotive_model)
    api_text = strip_non_spoken_tags_for_credit_saving(text, strip_all_tags=should_strip)
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

    # Nếu phiên hiện tại đã phát hiện tài khoản chưa được cấp quyền v4 thì dùng luôn model dự phòng để không gọi đúp
    effective_model = model_id
    if "v4" in effective_model and st.session_state.get("eleven_v4_fallback_active", False):
        effective_model = "eleven_multilingual_v2"
        api_text = strip_non_spoken_tags_for_credit_saving(api_text, strip_all_tags=True)

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
        # Fallback thông minh giống hệt Eleven Labstool.html nếu tài khoản chưa mở quyền eleven_v4
        if res.status_code != 200 and ("v4" in effective_model or "v3" in effective_model):
            if "v4" in effective_model:
                st.session_state["eleven_v4_fallback_active"] = True
            payload["model_id"] = "eleven_multilingual_v2"
            payload["text"] = strip_non_spoken_tags_for_credit_saving(api_text, strip_all_tags=True) or api_text
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
        return {"ok": True, "audio_path": final_p, "real_duration": round(max(0.2, dur), 3), "cached": False, "chars_billed": len(payload["text"])}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def lock_voice_audio_to_exact_slot(input_audio_path: str, slot_duration_sec: float, stt: int) -> Tuple[str, float]:
    """Khóa cứng thời lượng file Voice khớp 100% với khung giờ trong kịch bản bằng FFmpeg atempo."""
    if not input_audio_path or not os.path.exists(input_audio_path):
        return input_audio_path, 0.0
    raw_d = AutoVideoEditor._get_media_duration_fast(input_audio_path)
    if raw_d <= 0.1:
        return input_audio_path, round(slot_duration_sec, 2)
    if abs(raw_d - slot_duration_sec) <= 0.06:
        return input_audio_path, round(slot_duration_sec, 2)

    ratio = float(np.clip(raw_d / max(0.5, slot_duration_sec), 0.72, 1.48))
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


# ==============================================================================
# ĐẾM TỪ KHỚP NHỊP 2.5 TỪ/GIÂY (TÍNH CẢ THỜI GIAN THẺ CẢM XÚC ELEVEN v4)
# ==============================================================================

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
    min_words: int = 4,
    max_words: int = 120,
    slot_fill_pct: int = 100
) -> Dict[str, Any]:
    tag_count, pause_sec, _ = compute_emotion_tags_duration(text)
    spoken_text = re.sub(r'\[[^\]]+\]', '', str(text)).strip()
    words = count_clean_english_words(spoken_text)
    word_count = len(words)

    effective_wps = max(0.6, float(target_wps) * max(0.70, min(1.20, float(eleven_speed))))
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

    is_over = word_count > (ideal_max_words + 1)
    is_under = word_count < (ideal_max_words - 1)
    tag_badge = f" | 🎭 {tag_count} thẻ v4 ({pause_sec}s)" if tag_count > 0 else ""

    if is_over:
        status = f"⚠️ Dài hơn khung (+{max(0.1, diff_sec):.1f}s | {word_count}/{ideal_max_words} từ{tag_badge})"
    elif is_under:
        status = f"⚠️ Ngắn hơn khung ({diff_sec:.1f}s | {word_count}/{ideal_max_words} từ{tag_badge})"
    else:
        status = f"✅ Chuẩn khớp 100% ({slot_duration_sec:.1f}s | {word_count}/{ideal_max_words} từ{tag_badge})"

    return {
        "word_count": word_count,
        "tag_count": tag_count,
        "emotion_sec": pause_sec,
        "est_sec": round(slot_duration_sec, 2) if (not is_over and not is_under) else est_sec,
        "ms_per_word": ms_per_word,
        "ideal_max_words": ideal_max_words,
        "status": status
    }


def _call_gemini_vision_rest(gemini_key: str, prompt: str, image_path: str = "", json_mode: bool = False) -> str:
    clean_k = re.sub(r'[^\x20-\x7E]', '', str(gemini_key or '')).strip()
    if not clean_k:
        return ""

    b64_img = ""
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            b64_img = ""

    parts_list: List[Dict[str, Any]] = [{"text": prompt}]
    if b64_img:
        parts_list.append({"inline_data": {"mime_type": "image/jpeg", "data": b64_img}})

    gen_cfg: Dict[str, Any] = {"temperature": 0.4, "maxOutputTokens": 4096}
    if json_mode:
        gen_cfg["responseMimeType"] = "application/json"

    payload = {"contents": [{"parts": parts_list}], "generationConfig": gen_cfg}
    for m_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={clean_k}"
        try:
            r = requests.post(url, json=payload, timeout=25)
            if r.status_code == 200:
                cands = r.json().get("candidates") or []
                if cands:
                    content_parts = (cands[0].get("content") or {}).get("parts") or []
                    out_txt = "\n".join(p.get("text", "") for p in content_parts if not p.get("thought") and p.get("text")).strip()
                    if out_txt:
                        return out_txt
        except Exception:
            continue
    return ""


def _refine_vietnamese_construction_terms(vi_text: str) -> str:
    s = str(vi_text or "").strip()
    if not s:
        return s
    replacements = [
        (r"\bboong\s+chính\b", "mặt sàn chính"),
        (r"\bđinh\s+tán\s+bằng\s+gỗ\b", "các thanh trụ gỗ vách"),
        (r"\bcách\s+nhiệt\s+bằng\s+bọt\b", "xốp cách nhiệt"),
        (r"\bngười\s+xây\s+dựng\b", "người thợ"),
        (r"\bPLACES\b", "đặt"), (r"\bFASTENS\b", "bắt chặt"),
        (r"\bALIGNS\b", "căn chỉnh"), (r"\bINSTALLS\b", "lắp đặt"),
        (r"\bSECURES\b", "cố định"), (r"\bCARRIES\b", "khiêng"),
        (r"\bFITS\b", "lắp khít"), (r"\bLAYERS\b", "trải đều"),
        (r"\bLOCKS\b", "khóa chặt"), (r"\bSAWS\b", "cưa")
    ]
    for pat, rep in replacements:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    s = re.sub(r'\s+', ' ', s).strip()
    if s and not s.endswith((".", "!", "?")):
        s += "."
    return s[0].upper() + s[1:] if s else s


def translate_to_vietnamese_subtitle(gemini_key: str, text_en: str, existing_vi: str = "", force_refresh: bool = False) -> str:
    clean_src = _clean_srt_english_line(re.sub(r'\[[^\]]+\]', '', str(text_en)))
    if not clean_src:
        return "Chưa có nội dung thuyết minh."
    if not force_refresh and existing_vi and _has_vn_chars(existing_vi):
        return _refine_vietnamese_construction_terms(existing_vi)
    if _has_vn_chars(clean_src):
        return _refine_vietnamese_construction_terms(clean_src)

    words_for_trans = [w.capitalize() if i == 0 else (w.lower() if w.isupper() else w) for i, w in enumerate(clean_src.split())]
    normalized_en = " ".join(words_for_trans)

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": "en", "tl": "vi", "dt": "t", "q": normalized_en},
            headers=headers, timeout=6
        )
        if r.status_code == 200:
            data = r.json()
            if data and isinstance(data[0], list):
                vi_out = "".join(part[0] for part in data[0] if part and part[0]).strip()
                if _has_vn_chars(vi_out):
                    return _refine_vietnamese_construction_terms(vi_out)
    except Exception:
        pass
    return _refine_vietnamese_construction_terms(normalized_en)


def get_video_timeline_info(video_paths: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
    return VisionAnalyzer.get_video_timeline_info(video_paths)


def extract_slot_filmstrip_from_video(video_path_or_list: Any, start_sec: float, end_sec: float, idx_tag: int) -> Tuple[str, str]:
    va = VisionAnalyzer(temp_dir=THUMB_CACHE_DIR)
    return va.extract_slot_filmstrip(video_path_or_list, start_sec, end_sec, idx_tag)


def parse_vietnamese_command_and_visuals(user_vi_cmd: str, strip_image_path: str, vo_index: int) -> Dict[str, Any]:
    cmd_low = str(user_vi_cmd or "").strip().lower()
    if cmd_low:
        is_plural = any(k in cmd_low for k in ["2 người", "hai người", "cùng nhau", "nhóm", "phối hợp", "khiêng"])
        subj_en = "Working together beside the timber frame, the two builders" if is_plural else "Working with steady focus on the wooden deck, the builder"
        subj_vi = "Phối hợp nhịp nhàng bên khung nhà gỗ, hai người thợ" if is_plural else "Làm việc tập trung trên mặt sàn gỗ, người thợ"

        if any(k in cmd_low for k in ["thang đỏ", "thang"]):
            tool_en, tool_vi = "near the red step ladder", "ngay cạnh chiếc thang chữ A màu đỏ"
        elif any(k in cmd_low for k in ["bắn vít", "máy khoan", "khoan"]):
            tool_en, tool_vi = "using a cordless power drill", "bằng máy khoan bắn vít cầm tay"
        elif any(k in cmd_low for k in ["cưa", "cắt"]):
            tool_en, tool_vi = "guiding a handheld circular saw", "bằng máy cưa đĩa cầm tay"
        else:
            tool_en, tool_vi = "with steady hands on the platform", "với đôi tay khéo léo trên mặt sàn"

        if any(k in cmd_low for k in ["xốp", "cách nhiệt", "tấm trắng"]):
            mat_en, mat_vi = "thick white foam insulation boards", "các tấm xốp cách nhiệt màu trắng dày dặn"
            purpose_en = "between the wooden floor joists so the cabin floor stays warm and dry"
            purpose_vi = "vào giữa các khoang dầm sàn để nền nhà gỗ luôn ấm áp và khô ráo"
        elif any(k in cmd_low for k in ["vách", "khung tường", "cột"]):
            mat_en, mat_vi = "upright wooden wall studs", "các thanh trụ gỗ của khung vách đứng"
            purpose_en = "along the outer deck plate so the cabin walls stand straight and solid"
            purpose_vi = "dọc theo dầm biên mặt sàn để hệ khung vách nhà đứng thẳng và vững chắc"
        else:
            mat_en, mat_vi = "heavy structural timber boards", "các cấu kiện gỗ kết cấu chính"
            purpose_en = "into exact alignment across the frame so every joint locks together tightly"
            purpose_vi = "vào đúng vị trí trên khung nhà giúp mọi mối ghép liên kết chặt chẽ"

        verbs_3 = [("PLACES", "đặt"), ("FITS", "lắp khít"), ("SECURES", "cố định")]
        opt_list = []
        for idx_v, (v_en, v_vi) in enumerate(verbs_3):
            en_s = f"{subj_en} {v_en} the {mat_en} {tool_en} {purpose_en}, checking each edge carefully to ensure a tight and accurate fit across the platform."
            vi_s = f"{subj_vi} {v_vi} {mat_vi} {tool_vi} {purpose_vi}, đồng thời kiểm tra kỹ từng mép nối để đảm bảo độ khít chuẩn xác trên mặt sàn."
            opt_list.append({"style_label": f"Bản nháp {idx_v+1}: Bám sát lệnh Tiếng Việt", "anchor_verb": v_en, "en": en_s, "vi": vi_s})
        return {"tools_vi": tool_vi, "mats_vi": mat_vi, "action_vi": user_vi_cmd.strip(), "options": opt_list}

    return {
        "tools_vi": "Máy bắn vít cầm tay, thước thủy & búa mộc",
        "mats_vi": "Cấu kiện khung gỗ & tấm sàn chịu lực",
        "action_vi": "Đo đạc, căn chỉnh và cố định các thanh gỗ vào hệ khung chính",
        "options": [
            {
                "style_label": "Phương án 1: Trực diện Hành động & Dụng cụ",
                "anchor_verb": "FASTENS",
                "en": "Standing beside the upright timber wall section, the carpenter FASTENS vertical wooden studs to the horizontal plate with a cordless drill while checking that every post stands straight and plumb.",
                "vi": "Đứng bên cạnh phần khung vách gỗ dựng đứng, người thợ mộc bắt chặt các thanh trụ gỗ dọc vào thanh dầm ngang bằng máy khoan pin đồng thời kiểm tra để mỗi cột đều thẳng đứng."
            },
            {
                "style_label": "Phương án 2: Chi tiết Kỹ thuật & Độ khít",
                "anchor_verb": "ALIGNS",
                "en": "Moving along the framed exterior wall, the builder ALIGNS the top wooden beam across the upright studs and drives heavy structural screws home to lock the cabin frame together.",
                "vi": "Di chuyển dọc theo bức tường khung ngoài, người thợ căn chỉnh thanh đà gỗ đỉnh vắt ngang qua các cột trụ đứng rồi bắn vít chịu lực để khóa chặt khung nhà lại với nhau."
            },
            {
                "style_label": "Phương án 3: Phong cách Kể chuyện Documentary",
                "anchor_verb": "SECURES",
                "en": "Across the sunlit wooden deck, the craftsman SECURES each load bearing timber post into the main wall frame, raising a sturdy skeleton that gives shape to the forest cabin.",
                "vi": "Trên mặt sàn gỗ ngập nắng, người thợ cố định từng cột gỗ chịu lực vào hệ khung tường chính, dựng nên bộ khung vững chãi định hình cho ngôi nhà gỗ giữa rừng."
            }
        ]
    }


def build_precise_action_synced_sentence(
    raw_en_text: str,
    target_words: int,
    anchor_word_pos: int,
    anchor_verb_hint: str,
    scene_idx: int,
    strip_image_path: str = "",
    variant_idx: int = 0
) -> Tuple[str, int, str]:
    tags = re.findall(r'\[[^\]]+\]', str(raw_en_text))
    spoken = _clean_srt_english_line(re.sub(r'\[[^\]]+\]', '', str(raw_en_text)))
    if _has_vn_chars(spoken):
        spoken = ""

    spoken = re.sub(r'\s+', ' ', spoken.replace("-", " ")).strip()
    tokens = [t for t in spoken.split() if t.strip()]

    if len(tokens) < 10:
        parsed_vis = parse_vietnamese_command_and_visuals("", strip_image_path, scene_idx)
        chosen_opt = parsed_vis["options"][variant_idx % len(parsed_vis["options"])]
        spoken = chosen_opt["en"].replace("-", " ")
        tokens = [t for t in spoken.split() if t.strip()]
        anchor_verb_hint = chosen_opt["anchor_verb"]

    tokens = [t.capitalize() if i == 0 else (t if t == "I" else t.lower()) for i, t in enumerate(tokens)]
    safe_anchor_pos = int(np.clip(anchor_word_pos, 4, max(4, min(12, target_words - 6))))

    while len(count_clean_english_words(" ".join(tokens))) > target_words and len(tokens) > 8:
        tokens.pop(-1)

    cur_cnt = len(count_clean_english_words(" ".join(tokens)))
    if cur_cnt < target_words:
        diff = target_words - cur_cnt
        tokens[-1] = tokens[-1].rstrip(".!?,;:")
        while diff > 0:
            step_k = min(diff, 15)
            ext_choices = EXACT_GRAMMAR_EXTENSIONS.get(step_k, EXACT_GRAMMAR_EXTENSIONS[1])
            chosen_ext = ext_choices[(scene_idx + variant_idx) % len(ext_choices)]
            ext_tokens = chosen_ext.split()
            if diff > 15:
                ext_tokens[-1] = ext_tokens[-1].rstrip(".!?") + ","
            tokens.extend(ext_tokens)
            cur_cnt = len(count_clean_english_words(" ".join(tokens)))
            diff = target_words - cur_cnt

    while len(count_clean_english_words(" ".join(tokens))) > target_words and len(tokens) > 8:
        tokens.pop(-1)

    best_verb_idx = -1
    best_dist = 999
    for idx_t, tk in enumerate(tokens):
        clean_w = re.sub(r'[^\w]', '', tk).upper()
        if clean_w in VALID_ACTION_VERBS:
            dist = abs((idx_t + 1) - safe_anchor_pos)
            if dist < best_dist:
                best_dist = dist
                best_verb_idx = idx_t

    if best_verb_idx >= 0:
        verb_clean = re.sub(r'[^\w]', '', tokens[best_verb_idx]).upper()
        tokens[best_verb_idx] = verb_clean
    else:
        verb_clean = re.sub(r'[^\w]', '', str(anchor_verb_hint or "PLACES")).upper()

    if not tokens[-1].endswith((".", "!", "?")):
        tokens[-1] = tokens[-1].rstrip(",;:-") + "."

    final_spoken = " ".join(tokens)
    final_full = f"{' '.join(tags)} {final_spoken}".strip() if tags else final_spoken
    return final_full, len(count_clean_english_words(final_spoken)), verb_clean


def generate_multiple_vo_candidates_with_ai(
    gemini_key: str,
    vo_index: int,
    total_vos: int,
    start_sec: float,
    end_sec: float,
    slot_dur: int,
    target_words: int,
    anchor_sec_x: int,
    anchor_word_pos: int,
    next_gap_sec: int,
    strip_image_path: str,
    auto_v4_tags: bool = True,
    user_custom_hint: str = ""
) -> Dict[str, Any]:
    safe_anchor_pos = int(np.clip(anchor_word_pos, 4, max(4, min(12, target_words - 6))))
    local_parsed = parse_vietnamese_command_and_visuals(user_custom_hint, strip_image_path, vo_index)
    cmd_instruction = f'\nLỆNH TIẾNG VIỆT TỪ NGƯỜI DÙNG: "{user_custom_hint}" -> Bắt buộc viết theo đúng hành động này!' if user_custom_hint.strip() else ""

    master_prompt = f"""Bạn là Đạo diễn Phim tài liệu xây dựng. Quan sát bức ảnh 6 khung hình của phân đoạn VO #{vo_index}/{total_vos} (thời lượng {slot_dur} giây).{cmd_instruction}
Hãy viết 3 phương án kịch bản tiếng Anh ("en") dài ĐÚNG {target_words} từ, có 1 ĐỘNG TỪ HÀNH ĐỘNG IN HOA ở khoảng từ thứ {safe_anchor_pos}, kèm bản dịch tiếng Việt ("vi").
Trả về JSON:
{{
  "detected_tools_vi": "Dụng cụ trong ảnh",
  "detected_materials_vi": "Vật liệu trong ảnh",
  "micro_actions_vi": "Hành động cụ thể",
  "options": [
    {{"style_label": "Bản nháp 1", "anchor_verb": "FASTENS", "en": "...", "vi": "..."}},
    {{"style_label": "Bản nháp 2", "anchor_verb": "ALIGNS", "en": "...", "vi": "..."}},
    {{"style_label": "Bản nháp 3", "anchor_verb": "SECURES", "en": "...", "vi": "..."}}
  ]
}}"""

    detected_tools_vi = local_parsed["tools_vi"]
    detected_materials_vi = local_parsed["mats_vi"]
    micro_actions_vi = local_parsed["action_vi"]
    raw_options_list: List[Dict[str, Any]] = []

    raw_resp = _call_gemini_vision_rest(gemini_key, master_prompt, strip_image_path, json_mode=True)
    if raw_resp:
        try:
            m_json = re.search(r'\{.*\}', raw_resp, re.DOTALL)
            if m_json:
                parsed = json.loads(m_json.group(0), strict=False)
                detected_tools_vi = str(parsed.get("detected_tools_vi") or detected_tools_vi)
                detected_materials_vi = str(parsed.get("detected_materials_vi") or detected_materials_vi)
                micro_actions_vi = str(parsed.get("micro_actions_vi") or micro_actions_vi)
                if isinstance(parsed.get("options"), list):
                    raw_options_list = parsed["options"]
        except Exception:
            pass

    fallback_opts = local_parsed["options"]
    final_candidates: List[Dict[str, Any]] = []
    for c_idx in range(3):
        fb_item = fallback_opts[c_idx % len(fallback_opts)]
        raw_opt = raw_options_list[c_idx] if c_idx < len(raw_options_list) and isinstance(raw_options_list[c_idx], dict) else {}
        lbl = str(raw_opt.get("style_label") or fb_item["style_label"]).strip()
        v_hint = str(raw_opt.get("anchor_verb") or fb_item["anchor_verb"]).strip().upper()
        en_raw = str(raw_opt.get("en") or fb_item["en"]).strip()
        vi_raw = str(raw_opt.get("vi") or fb_item["vi"]).strip()

        en_fixed, act_w, v_final = build_precise_action_synced_sentence(
            en_raw, target_words, safe_anchor_pos, v_hint, vo_index, strip_image_path, variant_idx=c_idx
        )
        if auto_v4_tags and not re.findall(r'\[[^\]]+\]', en_fixed):
            en_fixed = f"[calm, documentary tone] {en_fixed}"
        vi_checked = translate_to_vietnamese_subtitle(gemini_key, en_fixed, vi_raw)
        final_candidates.append({
            "option_id": c_idx + 1, "style_label": lbl, "anchor_verb": v_final,
            "actual_words": act_w, "target_words": target_words, "en_text": en_fixed, "vi_text": vi_checked
        })

    primary = final_candidates[0]
    audit_badge = f"🛠️ Dụng cụ: {detected_tools_vi} | 🧱 Vật liệu: {detected_materials_vi} | 🎬 Hành động: {micro_actions_vi}"
    formatted_block = f"{vo_index}\n{format_timestamp_srt(start_sec)} --> {format_timestamp_srt(end_sec)}\n{primary['en_text']}\n"
    return {
        "audit_summary": audit_badge, "anchor_verb": primary["anchor_verb"],
        "actual_words": primary["actual_words"], "en_text": primary["en_text"],
        "vi_text": primary["vi_text"], "candidates": final_candidates, "formatted_block": formatted_block
    }


def build_dynamic_video_vo_schedule(
    video_paths: List[str], vo_dur_mode: int = 0, min_gap_sec: float = 45.0, max_gap_sec: float = 75.0
) -> Tuple[float, List[Tuple[int, float, float, int, int]]]:
    total_vid_dur, seg_metas = get_video_timeline_info(video_paths)
    if not seg_metas or total_vid_dur <= 5.0:
        return total_vid_dur, [(1, 0.0, min(8.0, total_vid_dur), int(min(8.0, total_vid_dur)), 0)]

    eff_min_gap = max(5, int(round(total_vid_dur * 0.08))) if total_vid_dur < 180.0 else int(min(min_gap_sec, max_gap_sec))
    eff_max_gap = max(10, int(round(total_vid_dur * 0.16))) if total_vid_dur < 180.0 else int(max(min_gap_sec, max_gap_sec))

    cur_t = 2.0 if total_vid_dur < 60.0 else 5.0
    dynamic_seq = [20, 15, 20, 12, 15, 20, 10, 15, 20, 12, 8, 20]
    planned_slots: List[Tuple[int, float, float, int, int]] = []
    vo_idx = 1

    while cur_t + 8.0 <= total_vid_dur and vo_idx <= 2000:
        slot_d_int = int(vo_dur_mode) if int(vo_dur_mode) > 0 else dynamic_seq[(vo_idx - 1) % len(dynamic_seq)]
        s_t = float(int(round(cur_t)))
        e_t = float(int(round(min(total_vid_dur, s_t + slot_d_int))))
        actual_slot_d = max(5, int(round(e_t - s_t)))
        gap_chosen = int(np.clip(eff_min_gap + ((vo_idx * 7) % max(1, eff_max_gap - eff_min_gap)), eff_min_gap, eff_max_gap))
        planned_slots.append((vo_idx, s_t, e_t, actual_slot_d, gap_chosen))
        cur_t = float(int(e_t) + gap_chosen)
        vo_idx += 1
    return total_vid_dur, planned_slots


def create_synced_preview_video_clip(
    video_paths: List[str], start_sec: float, end_sec: float, voice_audio_path: str, row_id: int
) -> str:
    if not video_paths:
        return ""
    _, seg_metas = get_video_timeline_info(video_paths)
    if not seg_metas:
        return ""
    chosen_seg = next((sm for sm in seg_metas if sm["start_global"] <= start_sec <= sm["end_global"]), seg_metas[0])
    local_start = max(0.0, start_sec - chosen_seg["start_global"])
    slot_dur = max(1.0, end_sec - start_sec)
    has_voice = bool(voice_audio_path and os.path.exists(str(voice_audio_path)))
    v_tag = hashlib.md5(f"pv_{chosen_seg['video_path']}_{local_start:.2f}_{slot_dur:.2f}_{voice_audio_path if has_voice else 'no'}".encode("utf-8")).hexdigest()[:12]
    out_clip = os.path.join(CLIP_PREVIEW_DIR, f"fast_pv_row_{row_id:04d}_{v_tag}.mp4")

    if os.path.exists(out_clip) and os.path.getsize(out_clip) > 2048:
        return out_clip

    try:
        if has_voice:
            filter_str = (
                "[0:a]aresample=44100,aformat=sample_rates=44100:channel_layouts=stereo,asetpts=PTS-STARTPTS,volume=0.75[a0];"
                "[1:a]aresample=44100,aformat=sample_rates=44100:channel_layouts=stereo,asetpts=PTS-STARTPTS,volume=1.0[a1];"
                "[a0][a1]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
            )
            cmd = [
                "ffmpeg", "-y", "-ss", f"{local_start:.3f}", "-t", f"{slot_dur:.3f}",
                "-i", chosen_seg["video_path"], "-i", str(voice_audio_path),
                "-filter_complex", filter_str, "-map", "0:v:0", "-map", "[aout]",
                "-vf", "scale=640:360:flags=fast_bilinear", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "192k", "-shortest", out_clip
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-ss", f"{local_start:.3f}", "-t", f"{slot_dur:.3f}",
                "-i", chosen_seg["video_path"], "-vf", "scale=640:360:flags=fast_bilinear",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "192k", out_clip
            ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
        if os.path.exists(out_clip) and os.path.getsize(out_clip) > 2048:
            return out_clip
    except Exception:
        pass
    return ""


def import_zip_voice_pack_to_rows(zip_file_obj, existing_rows: List[Dict[str, Any]], video_paths: Any = "") -> List[Dict[str, Any]]:
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
    curr_t = 0.0
    for idx, (fname, fpath) in enumerate(extracted_files):
        r_dur = round(max(0.5, AutoVideoEditor._get_media_duration_fast(fpath) or 4.0), 3)
        if idx < len(updated_rows):
            slot_d = float(updated_rows[idx].get("Khung Giờ (s)", r_dur))
            locked_p, locked_d = lock_voice_audio_to_exact_slot(fpath, slot_d, idx + 1)
            updated_rows[idx]["audio_path"] = locked_p
            updated_rows[idx]["Voice Thực Tế (s)"] = locked_d
            updated_rows[idx]["Trạng Thái Khớp Nhịp"] = f"✅ Đã nạp & khóa từ ZIP ({locked_d}s / {slot_d}s)"
        else:
            s_sec = curr_t
            e_sec = round(s_sec + r_dur, 3)
            curr_t = e_sec + 1.0
            th_p, st_p = extract_slot_filmstrip_from_video(video_paths, s_sec, e_sec, idx + 1) if video_paths else ("", "")
            updated_rows.append({
                "STT": idx + 1, "Mốc Bắt Đầu": format_timestamp_srt(s_sec), "Mốc Kết Thúc": format_timestamp_srt(e_sec),
                "Khung Giờ (s)": round(r_dur, 2), "Điểm Chạm & Từ Neo": "Auto Sync",
                "Kiểm Tra Dụng Cụ & Hành Động": "Nạp từ gói ZIP",
                "Số Từ": 0, "Tối Đa Từ Chuẩn": 0, "Voice Ước Tính (s)": r_dur, "Voice Thực Tế (s)": r_dur,
                "Trạng Thái Khớp Nhịp": f"✅ Đã nạp từ ZIP ({fname})",
                "Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)": f"[Audio từ ZIP] {fname}",
                "Phụ Đề Tiếng Việt (Dễ Hiểu)": f"File âm thanh nạp từ gói ZIP: {fname}",
                "audio_path": fpath, "frame_path": th_p, "strip_path": st_p
            })
    return updated_rows


# ==============================================================================
# GIAO DIỆN ELEVENLABS STUDIO PRO V2.1 (ELEVEN V4 + BỘ LỌC GIỌNG + AUDIO TAGS)
# ==============================================================================

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
        act_vids = []
        if uploaded_videos:
            for vf in sorted(flatten_files_fn(uploaded_videos), key=natural_sort_key):
                pv, _ = resolve_or_cache_fn(vf)
                if pv and os.path.exists(pv):
                    act_vids.append(pv)
        return act_vids

    el_tab_native, el_tab_clone = st.tabs([
        "🔥 ElevenLabs Studio V2.1 Pro (Eleven v4, Bộ Lọc Giọng, Thẻ Cảm Xúc & Khóa Khớp Mốc Kịch Bản)",
        "🧬 Trạm Clone Giọng Nhanh (Instant Voice Cloning)"
    ])

    with el_tab_native:
        # Bố cục 2 cột chuẩn như Eleven Labstool.html: Cột trái (Chọn Eleven v4, Lọc Giọng, Thanh trượt cảm xúc) | Cột phải (Thẻ cảm xúc v4, Kịch bản)
        left_ctrl_col, right_script_col = st.columns([0.38, 0.62])

        all_el_voices = st.session_state.get("eleven_voices_list", DEFAULT_ELEVEN_VOICES)
        model_options_map = st.session_state.get("eleven_models_map", DEFAULT_ELEVEN_MODELS)

        with left_ctrl_col:
            st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
            st.markdown("##### ⚙️ 1. Chọn Model Eleven v4 & Lọc Giọng Đọc")

            m_keys = list(model_options_map.keys())
            saved_em = saved_cfg.get("eleven_model_id", "eleven_v4")
            eleven_model_id = st.selectbox(
                "Mô hình AI (Model):",
                options=m_keys,
                format_func=lambda k: model_options_map.get(k, k),
                index=m_keys.index(saved_em) if saved_em in m_keys else 0
            )
            st.caption("✅ Hỗ trợ đầy đủ Eleven v4 Emotive Tags & tự động chuyển sang Multilingual v2 nếu tài khoản chưa mở v4.")

            # Hộp Bộ Lọc Giọng Đọc Thông Minh (Smart Voice Filters từ Eleven Labstool.html)
            flt_c1, flt_c2 = st.columns(2)
            with flt_c1:
                flt_gender = st.selectbox(
                    "Giới tính:",
                    options=["all", "male", "female"],
                    format_func=lambda x: "Tất cả (Nam & Nữ)" if x == "all" else ("👨 Giọng Nam (Male)" if x == "male" else "👩 Giọng Nữ (Female)")
                )
            with flt_c2:
                flt_cat = st.selectbox(
                    "Nguồn giọng:",
                    options=["all", "cloned", "premade"],
                    format_func=lambda x: "Tất cả nguồn" if x == "all" else ("🧬 Giọng đã Clone" if x == "cloned" else "🌟 Giọng Mặc định")
                )

            flt_c3, flt_c4 = st.columns(2)
            with flt_c3:
                emotion_map = {
                    "all": "Mọi cảm xúc",
                    "narrative": "📖 Kể chuyện / Documentary",
                    "calm": "🍃 Điềm tĩnh, Trầm ấm (Calm)",
                    "deep": "🔊 Trầm dày (Deep)",
                    "raspy": "🔥 Khàn đặc trưng (Raspy)",
                    "conversational": "💬 Tự nhiên đời thường"
                }
                flt_emo = st.selectbox("Cảm xúc / Phong cách:", options=list(emotion_map.keys()), format_func=lambda k: emotion_map[k])
            with flt_c4:
                flt_search = st.text_input("🔍 Tìm tên / Vùng miền:", value="", placeholder="VD: Adam, Brian...")

            filtered_voices = [
                v for v in all_el_voices
                if (flt_gender == "all" or v.get("gender") == flt_gender)
                and (flt_cat == "all" or (flt_cat == "cloned" and v.get("category") == "cloned") or (flt_cat == "premade" and v.get("category") != "cloned"))
                and (flt_emo == "all" or flt_emo in f"{v.get('use_case', '')} {v.get('description', '')}".lower())
                and (not flt_search.strip() or flt_search.strip().lower() in f"{v.get('name', '')} {v.get('accent', '')} {v.get('description', '')} {v.get('use_case', '')}".lower())
            ]
            if not filtered_voices:
                filtered_voices = all_el_voices

            v_ids = [v["voice_id"] for v in filtered_voices]
            v_lbls = {
                v["voice_id"]: f"{'🧬 [Clone]' if v.get('category') == 'cloned' else ('👩' if v.get('gender') == 'female' else '👨')} {v['name']} ({', '.join(filter(None, [v.get('accent'), v.get('description') or v.get('use_case')])) or 'Pro'})"
                for v in filtered_voices
            }
            saved_vid = saved_cfg.get("eleven_voice_id", v_ids[0])
            eleven_voice_id = st.selectbox(
                f"Giọng đọc đang chọn ({len(filtered_voices)} giọng khớp lọc):",
                options=v_ids,
                format_func=lambda k: v_lbls.get(k, k),
                index=v_ids.index(saved_vid) if saved_vid in v_ids else 0
            )

            chosen_v_obj = next((v for v in filtered_voices if v["voice_id"] == eleven_voice_id), None)
            if chosen_v_obj:
                badges = [
                    "🧬 Giọng Clone" if chosen_v_obj.get("category") == "cloned" else "⭐ Chuẩn ElevenLabs",
                    f"Giới tính: {chosen_v_obj.get('gender')}" if chosen_v_obj.get("gender") else "",
                    f"Accent: {chosen_v_obj.get('accent')}" if chosen_v_obj.get("accent") else "",
                    f"Style: {chosen_v_obj.get('use_case')}" if chosen_v_obj.get("use_case") else ""
                ]
                badges_html = " ".join(
                    f'<span style="font-size:11px;background:#1e293b;border:1px solid #475569;padding:2px 7px;border-radius:6px;margin-right:4px;">{b}</span>'
                    for b in badges if b
                )
                st.markdown(f"<div style='margin-bottom:6px;'>{badges_html}</div>", unsafe_allow_html=True)
                if chosen_v_obj.get("preview_url"):
                    st.caption("🔊 Nghe thử mẫu giọng (0 tốn credit):")
                    st.audio(chosen_v_obj["preview_url"])

            st.markdown("##### 🎛️ 2. Tốc Độ AI & Thanh Trượt Cảm Xúc")
            eleven_speed = st.slider(
                "⚡ Tốc độ nhả chữ AI (Native Speed: 0.70x Chậm sâu | 0.92x Chuẩn Tài liệu | 1.20x Nhanh)",
                0.70, 1.20, float(saved_cfg.get("eleven_speed", 1.00)), 0.02
            )
            st_c1, st_c2, st_c3 = st.columns(3)
            with st_c1:
                eleven_stability = st.slider("Stability (%)", 0, 100, int(saved_cfg.get("eleven_stability", 45)), 5)
            with st_c2:
                eleven_similarity = st.slider("Similarity (%)", 0, 100, int(saved_cfg.get("eleven_similarity", 80)), 5)
            with st_c3:
                eleven_style = st.slider("Style Boost (%)", 0, 100, int(saved_cfg.get("eleven_style", 15)), 5)

            if st.button("🔄 Áp dụng Giọng & Tốc độ này cho toàn bộ danh sách bên phải", use_container_width=True):
                for r_item in st.session_state.get("eleven_script_rows", []):
                    r_item["row_voice_id"] = eleven_voice_id
                    r_item["row_speed"] = eleven_speed
                st.toast("✅ Đã đồng bộ Giọng đọc & Tốc độ cho toàn bộ các câu!")
                st.rerun()

            with st.expander("⏱️ Cài đặt nâng cao: Tốc độ 2.5 từ/giây & Khóa khung giờ", expanded=False):
                target_wps = st.slider("🗣️ Hệ số tốc độ từ chuẩn (Từ/giây - WPS)", 1.00, 4.50, float(saved_cfg.get("target_wps", 2.50)), 0.05)
                slot_fill_pct = st.slider("Tỷ lệ lấp đầy khung giờ (%)", 50, 100, int(saved_cfg.get("slot_fill_pct", 100)), 1)
                voice_lead_ms = st.slider("Độ trễ / sớm khi vào giọng (ms)", -500, 2000, int(saved_cfg.get("voice_lead_ms", 0)), 25)
                align_map = {
                    "start": "⏮️ Vào giọng ngay đầu khung giờ (Chuẩn SRT)",
                    "center": "⏺️ Căn câu Voice vào chính giữa khung giờ",
                    "end": "⏭️ Dồn câu Voice về sát cuối khung giờ"
                }
                al_keys = list(align_map.keys())
                saved_al = saved_cfg.get("slot_align_mode", "start")
                slot_align_mode = st.selectbox("Vị trí đặt câu Voice:", options=al_keys, format_func=lambda k: align_map[k], index=al_keys.index(saved_al) if saved_al in al_keys else 0)
                strict_credit_saver = st.checkbox("🛡️ Không tạo lại câu đã có sẵn Voice (Chống trùng Cache)", value=True)
                keep_v4_emotive_tags_in_api = st.checkbox(
                    "🎭 Giữ Thẻ Cảm Xúc khi đọc bằng Eleven v4 / v3 (Để AI diễn cảm & thở thật)",
                    value=bool(saved_cfg.get("keep_v4_emotive_tags_in_api", True))
                )
                auto_fit_voice_slot = st.checkbox("🔒 Tự động co giãn nhẹ (atempo) để Voice khớp 100% khung giờ kịch bản", value=bool(saved_cfg.get("auto_fit_voice_slot", True)))
                auto_inject_v4_tags = st.checkbox("🎭 Tự chèn [calm, documentary tone] khi AI viết kịch bản", value=bool(saved_cfg.get("auto_inject_v4_tags", True)))
            st.markdown('</div>', unsafe_allow_html=True)

        with right_script_col:
            st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
            # Thanh Chèn Nhanh Thẻ Đạo Diễn Cảm Xúc Eleven v4 / v3 (Giống hệt Eleven Labstool.html)
            st.markdown("##### 🎭 Chèn Nhanh Thẻ Đạo Diễn Cảm Xúc (Dành riêng cho Eleven v4 / v3 — nhấp để chèn):")
            tag_cols = st.columns(4)
            for idx_tg, (tag_code, tag_label) in enumerate(ELEVEN_V4_EMOTIVE_TAGS):
                with tag_cols[idx_tg % 4]:
                    if st.button(tag_label, key=f"v4_tag_btn_{idx_tg}", use_container_width=True):
                        cur_buf = st.session_state.get("eleven_raw_text_buffer", "")
                        st.session_state["eleven_raw_text_buffer"] = (cur_buf + " " + tag_code).lstrip()
                        for r_row in st.session_state.get("eleven_script_rows", []):
                            cur_s = str(r_row.get("Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)", "")).strip()
                            if tag_code.strip() not in cur_s:
                                r_row["Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)"] = f"{tag_code.strip()} {cur_s}".strip()
                        st.toast(f"🎭 Đã chèn thẻ {tag_code.strip()} vào kịch bản!")
                        st.rerun()

            raw_script_input = st.text_area(
                "📝 Dán Kịch Bản (.SRT hoặc Văn Bản) Vào Đây — Với Eleven v4 bạn có thể chèn [sighs], [takes a deep breath], [calm, documentary tone]:",
                value=st.session_state.get("eleven_raw_text_buffer", ""),
                height=155,
                placeholder="1\n00:00:05,000 --> 00:00:20,000\n[calm, documentary tone] Standing beside the timber wall, the carpenter FASTENS vertical wooden studs..."
            )
            st.session_state["eleven_raw_text_buffer"] = raw_script_input
            st.caption(f"📊 Tổng ký tự dự kiến: **{len(raw_script_input):,}** ký tự")

            _has_vids = len(_get_all_loaded_video_paths()) > 0
            _has_script_in = bool(raw_script_input.strip()) or bool(uploaded_audios)
            _has_rows = len(st.session_state.get("eleven_script_rows", [])) > 0
            _has_el_key = bool(clean_eleven_api_key(elevenlabs_api_key))
            _has_voices = any(r.get("audio_path") and os.path.exists(str(r.get("audio_path", ""))) for r in st.session_state.get("eleven_script_rows", []))

            act_c1, act_c2, act_c3 = st.columns([0.36, 0.36, 0.28])
            with act_c1:
                analyze_script_clicked = st.button(
                    "🔍 1. BÓC TÁCH KỊCH BẢN / .SRT THÀNH BẢNG" if _has_script_in else "🔒 1. DÁN KỊCH BẢN HOẶC TẢI .SRT TRƯỚC",
                    type="primary" if _has_script_in else "secondary",
                    use_container_width=True,
                    disabled=(not _has_script_in)
                )
            with act_c2:
                auto_director_master_btn = st.button(
                    "🎬 2. AI QUÉT VIDEO & TỰ VIẾT KỊCH BẢN v4" if _has_vids else "🔒 2. TẢI VIDEO Ở MỤC 1 ĐỂ AI QUÉT",
                    use_container_width=True,
                    disabled=(not _has_vids)
                )
            with act_c3:
                ai_rewrite_clicked = st.button(
                    "✨ 3. CÂN CHỈNH LẠI SỐ TỪ THEO ẢNH" if _has_rows else "🔒 3. CHƯA CÓ BẢNG KỊCH BẢN",
                    use_container_width=True,
                    disabled=(not _has_rows)
                )

            zip_col1, zip_col2 = st.columns([0.55, 0.45])
            with zip_col1:
                vo_dur_mode = st.selectbox(
                    "Độ dài mỗi câu khi AI tự tạo kịch bản:",
                    options=[0, 20, 15, 12, 10, 8],
                    format_func=lambda x: "🔄 Tự động linh hoạt (8s -> 20s)" if x == 0 else f"⏱️ Cố định {x} giây ({MANDATORY_DURATION_WORD_MAP.get(x, int(x*2.5))} từ)",
                    index=0
                )
            with zip_col2:
                uploaded_zip_pack = st.file_uploader("📦 Hoặc Nạp Gói .ZIP Voice Có Sẵn:", type=["zip"])
                if uploaded_zip_pack is not None:
                    if st.button("📥 Giải Nén & Khóa Khớp Mốc Giây Từ .ZIP", use_container_width=True):
                        p_vids = _get_all_loaded_video_paths()
                        st.session_state["eleven_script_rows"] = import_zip_voice_pack_to_rows(
                            uploaded_zip_pack, st.session_state.get("eleven_script_rows", []), video_paths=p_vids
                        )
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        step_a_hud = st.empty()

        if auto_director_master_btn:
            act_vids = _get_all_loaded_video_paths()
            t0 = time.time()
            render_live_task_hud(step_a_hud, 0.05, "ĐẠO DIỄN AI ĐANG QUÉT VIDEO", "Đang chia khung giờ thuyết minh...", t0)
            _, planned_slots = build_dynamic_video_vo_schedule(act_vids, vo_dur_mode=int(vo_dur_mode))
            rows_built, master_blocks_text = [], []
            tot_planned = max(1, len(planned_slots))
            va_helper = VisionAnalyzer(temp_dir=THUMB_CACHE_DIR)

            for vo_num, s_t, e_t, slot_d_int, gap_after in planned_slots:
                render_live_task_hud(step_a_hud, vo_num / tot_planned, f"ĐANG QUÉT 6 KHUNG HÌNH VO #{vo_num:02d}/{tot_planned}", f"Mốc {format_timestamp_mmss(s_t)} -> {format_timestamp_mmss(e_t)}", t0)
                th_p, st_p = va_helper.extract_slot_filmstrip(act_vids, s_t, e_t, vo_num)
                anchor_info = va_helper.detect_visual_anchor(act_vids, s_t, e_t, target_wps=target_wps)
                target_w_cnt = MANDATORY_DURATION_WORD_MAP.get(slot_d_int, int(round(slot_d_int * target_wps)))
                vo_res = generate_multiple_vo_candidates_with_ai(
                    gemini_key=gemini_api_key, vo_index=vo_num, total_vos=tot_planned,
                    start_sec=s_t, end_sec=e_t, slot_dur=slot_d_int, target_words=target_w_cnt,
                    anchor_sec_x=anchor_info["anchor_sec_in_clip"], anchor_word_pos=anchor_info["anchor_word_index"],
                    next_gap_sec=gap_after, strip_image_path=st_p or th_p, auto_v4_tags=auto_inject_v4_tags
                )
                master_blocks_text.append(vo_res["formatted_block"])
                m_inf = compute_script_timing_metrics(vo_res["en_text"], float(slot_d_int), eleven_speed, target_wps, 4, 120, slot_fill_pct)
                rows_built.append({
                    "STT": vo_num, "Mốc Bắt Đầu": format_timestamp_srt(s_t), "Mốc Kết Thúc": format_timestamp_srt(e_t),
                    "Khung Giờ (s)": float(slot_d_int), "Điểm Chạm & Từ Neo": f"Giây {anchor_info['anchor_sec_in_clip']}s -> ({vo_res['anchor_verb']}) | Nghỉ {gap_after}s",
                    "Kiểm Tra Dụng Cụ & Hành Động": vo_res.get("audit_summary", ""),
                    "Số Từ": vo_res["actual_words"], "Tối Đa Từ Chuẩn": target_w_cnt,
                    "Voice Ước Tính (s)": m_inf["est_sec"], "Voice Thực Tế (s)": 0.0,
                    "Trạng Thái Khớp Nhịp": m_inf["status"],
                    "Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)": vo_res["en_text"],
                    "Phụ Đề Tiếng Việt (Dễ Hiểu)": vo_res["vi_text"],
                    "candidates": vo_res.get("candidates", []), "draft_candidates": [], "user_vi_cmd": "",
                    "row_voice_id": eleven_voice_id, "row_speed": eleven_speed,
                    "audio_path": "", "frame_path": th_p, "strip_path": st_p or th_p
                })
            step_a_hud.empty()
            st.session_state["eleven_raw_text_buffer"] = "\n".join(master_blocks_text)
            st.session_state["eleven_script_rows"] = rows_built
            st.rerun()

        if analyze_script_clicked:
            t0 = time.time()
            act_vids = _get_all_loaded_video_paths()
            txt_parse = raw_script_input.strip()
            if not txt_parse and uploaded_audios:
                for af in flatten_files_fn(uploaded_audios):
                    if af.name.lower().endswith((".srt", ".txt", ".md")):
                        p_sc, _ = resolve_or_cache_fn(af)
                        with open(p_sc, "r", encoding="utf-8", errors="ignore") as f:
                            txt_parse = f.read().strip()
                        st.session_state["eleven_raw_text_buffer"] = txt_parse
                        break

            if not txt_parse:
                st.warning("⚠️ Vui lòng dán kịch bản có mốc thời gian vào ô trên hoặc tải file .SRT ở Mục 1!")
            else:
                tmp_srt = os.path.join(CACHE_DIR, "temp_parse_script.srt")
                with open(tmp_srt, "w", encoding="utf-8") as f:
                    f.write(txt_parse)
                parsed_blocks = AudioProcessor().parse_script_file(tmp_srt).get("segments", [])
                if not parsed_blocks:
                    # Nếu người dùng dán văn bản thuần chưa có mốc giờ -> Tự động chia theo dòng giống Eleven Labstool.html
                    lines = [ln.strip() for ln in re.split(r'\n+', txt_parse) if ln.strip()]
                    _, dyn_slots = build_dynamic_video_vo_schedule(act_vids, vo_dur_mode=int(vo_dur_mode))
                    for idx_l, ln in enumerate(lines):
                        if idx_l < len(dyn_slots):
                            _, s_t, e_t, d_int, _ = dyn_slots[idx_l]
                        else:
                            prev_e = parsed_blocks[-1]["end"] if parsed_blocks else 5.0
                            s_t, e_t, d_int = round(prev_e + 15.0, 3), round(prev_e + 30.0, 3), 15
                        parsed_blocks.append({"start": s_t, "end": e_t, "duration": float(d_int), "text": ln, "vi_text": ""})

                rows_built = []
                for idx_b, blk in enumerate(parsed_blocks):
                    s_t = float(blk["start"])
                    e_t = float(blk["end"])
                    slot_d = round(max(1.0, e_t - s_t), 2)
                    render_live_task_hud(step_a_hud, (idx_b + 1) / max(1, len(parsed_blocks)), f"ĐANG ĐỌC CÂU #{idx_b+1}/{len(parsed_blocks)}", f"Mốc {format_timestamp_srt(s_t)} --> {format_timestamp_srt(e_t)}", t0)
                    txt_b = _clean_srt_english_line(blk["text"])
                    th_p, st_p = extract_slot_filmstrip_from_video(act_vids, s_t, e_t, idx_b + 1) if act_vids else ("", "")
                    vi_sub = translate_to_vietnamese_subtitle(gemini_api_key, txt_b, blk.get("vi_text", ""))
                    m_inf = compute_script_timing_metrics(txt_b, slot_d, eleven_speed, target_wps, 4, 120, slot_fill_pct)
                    cached_p = get_cached_elevenlabs_mp3(txt_b, eleven_voice_id, eleven_model_id, eleven_speed, eleven_stability, eleven_similarity, eleven_style)
                    real_d_cached = 0.0
                    if cached_p:
                        cached_p, real_d_cached = lock_voice_audio_to_exact_slot(cached_p, slot_d, idx_b + 1)
                    rows_built.append({
                        "STT": idx_b + 1, "Mốc Bắt Đầu": format_timestamp_srt(s_t), "Mốc Kết Thúc": format_timestamp_srt(e_t),
                        "Khung Giờ (s)": slot_d, "Điểm Chạm & Từ Neo": "Chuẩn Kịch Bản",
                        "Kiểm Tra Dụng Cụ & Hành Động": "Theo kịch bản đã nhập",
                        "Số Từ": m_inf["word_count"], "Tối Đa Từ Chuẩn": m_inf["ideal_max_words"],
                        "Voice Ước Tính (s)": m_inf["est_sec"], "Voice Thực Tế (s)": real_d_cached,
                        "Trạng Thái Khớp Nhịp": m_inf["status"],
                        "Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)": txt_b,
                        "Phụ Đề Tiếng Việt (Dễ Hiểu)": vi_sub,
                        "candidates": [], "draft_candidates": [], "user_vi_cmd": "",
                        "row_voice_id": eleven_voice_id, "row_speed": eleven_speed,
                        "audio_path": cached_p, "frame_path": th_p, "strip_path": st_p or th_p
                    })
                step_a_hud.empty()
                st.session_state["eleven_script_rows"] = rows_built
                st.rerun()

        if ai_rewrite_clicked and st.session_state.get("eleven_script_rows"):
            cur_rows = st.session_state["eleven_script_rows"]
            act_vids = _get_all_loaded_video_paths()
            t0 = time.time()
            for ir, r_it in enumerate(cur_rows):
                s_t = parse_timestamp_srt(str(r_it.get("Mốc Bắt Đầu", "00:00:05,000")))
                e_t = parse_timestamp_srt(str(r_it.get("Mốc Kết Thúc", "00:00:20,000")))
                slot_d = int(round(max(5.0, e_t - s_t)))
                render_live_task_hud(step_a_hud, (ir + 1) / len(cur_rows), f"ĐANG CÂN CHỈNH SỐ TỪ VO #{ir+1}/{len(cur_rows)}", f"Khung {slot_d}s...", t0)
                th_p, st_p = extract_slot_filmstrip_from_video(act_vids, s_t, e_t, ir + 1) if act_vids else ("", "")
                target_w_cnt = MANDATORY_DURATION_WORD_MAP.get(slot_d, int(round(slot_d * target_wps)))
                vo_res = generate_multiple_vo_candidates_with_ai(
                    gemini_key=gemini_api_key, vo_index=ir + 1, total_vos=len(cur_rows),
                    start_sec=s_t, end_sec=e_t, slot_dur=slot_d, target_words=target_w_cnt,
                    anchor_sec_x=4, anchor_word_pos=10, next_gap_sec=45,
                    strip_image_path=st_p or th_p or r_it.get("strip_path", ""),
                    auto_v4_tags=auto_inject_v4_tags, user_custom_hint=r_it.get("user_vi_cmd", "")
                )
                m_af = compute_script_timing_metrics(vo_res["en_text"], float(slot_d), eleven_speed, target_wps, 4, 120, slot_fill_pct)
                r_it.update({
                    "Số Từ": m_af["word_count"], "Tối Đa Từ Chuẩn": m_af["ideal_max_words"],
                    "Voice Ước Tính (s)": m_af["est_sec"], "Trạng Thái Khớp Nhịp": m_af["status"],
                    "Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)": vo_res["en_text"],
                    "Phụ Đề Tiếng Việt (Dễ Hiểu)": vo_res["vi_text"],
                    "candidates": vo_res.get("candidates", [])
                })
            step_a_hud.empty()
            st.rerun()

        # ======================================================================
        # BẢNG CHỈNH SỬA KỊCH BẢN & TẠO VOICE ELEVEN v4 HÀNG LOẠT
        # ======================================================================
        if st.session_state.get("eleven_script_rows"):
            st.markdown(f"##### 🎧 Danh sách Đoạn Thoại ({len(st.session_state['eleven_script_rows'])} đoạn) — Model đang chọn: `{eleven_model_id}`")
            disp_cols = [
                "STT", "Mốc Bắt Đầu", "Mốc Kết Thúc", "Khung Giờ (s)",
                "Số Từ", "Tối Đa Từ Chuẩn", "Voice Ước Tính (s)", "Voice Thực Tế (s)",
                "Trạng Thái Khớp Nhịp", "Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)", "Phụ Đề Tiếng Việt (Dễ Hiểu)"
            ]
            clean_tb = [{k: r.get(k, "") for k in disp_cols} for r in st.session_state["eleven_script_rows"]]
            edited_sc_tb = st.data_editor(clean_tb, use_container_width=True, height=290, num_rows="fixed", key="el_script_editor_main")

            for ir, ed_r in enumerate(edited_sc_tb):
                if ir < len(st.session_state["eleven_script_rows"]):
                    orig_r = st.session_state["eleven_script_rows"][ir]
                    s_new = parse_timestamp_srt(str(ed_r.get("Mốc Bắt Đầu", "00:00:00,000")))
                    e_new = parse_timestamp_srt(str(ed_r.get("Mốc Kết Thúc", "00:00:05,000")))
                    sl_new = round(max(0.5, e_new - s_new), 2)
                    tx_new = str(ed_r.get("Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)", ""))
                    vi_new = str(ed_r.get("Phụ Đề Tiếng Việt (Dễ Hiểu)", orig_r.get("Phụ Đề Tiếng Việt (Dễ Hiểu)", "")))
                    r_spd = float(orig_r.get("row_speed") or eleven_speed)
                    m_u = compute_script_timing_metrics(tx_new, sl_new, r_spd, target_wps, 4, 120, slot_fill_pct)
                    orig_r.update({
                        "Mốc Bắt Đầu": format_timestamp_srt(s_new), "Mốc Kết Thúc": format_timestamp_srt(e_new),
                        "Khung Giờ (s)": sl_new, "Số Từ": m_u["word_count"], "Tối Đa Từ Chuẩn": m_u["ideal_max_words"],
                        "Voice Ước Tính (s)": m_u["est_sec"], "Trạng Thái Khớp Nhịp": m_u["status"],
                        "Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)": tx_new, "Phụ Đề Tiếng Việt (Dễ Hiểu)": vi_new
                    })

            dl_c1, dl_c2, dl_c3 = st.columns([0.38, 0.38, 0.24])
            with dl_c1:
                srt_export_lines = [f"{int(r['STT'])}\n{r['Mốc Bắt Đầu']} --> {r['Mốc Kết Thúc']}\n{r['Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)']}\n" for r in st.session_state["eleven_script_rows"]]
                st.download_button("📥 Tải Kịch Bản Chuẩn (.SRT)", data="\n".join(srt_export_lines).encode("utf-8"), file_name="Subtitles_Timeline.srt", mime="text/plain", use_container_width=True)
            with dl_c2:
                vi_export_lines = [f"{int(r['STT'])}\n{r['Mốc Bắt Đầu']} --> {r['Mốc Kết Thúc']}\n{r['Phụ Đề Tiếng Việt (Dễ Hiểu)']}\n" for r in st.session_state["eleven_script_rows"]]
                st.download_button("📥 Tải Phụ Đề Tiếng Việt (.SRT)", data="\n".join(vi_export_lines).encode("utf-8"), file_name="Phu_De_Tieng_Viet.srt", mime="text/plain", use_container_width=True)
            with dl_c3:
                if st.button("🗑️ Xóa Bảng Kịch Bản", use_container_width=True):
                    st.session_state["eleven_script_rows"] = []
                    st.rerun()

            bc1, bc2, bc3 = st.columns([0.44, 0.28, 0.28])
            with bc1:
                gen_all_eleven_clicked = st.button(
                    f"▶️ TẠO TẤT CẢ VOICE VỚI {eleven_model_id.upper()} (CHỐNG TRÙNG CACHE)" if (_has_el_key and _has_rows) else "🔒 TẠO VOICE (NHẬP ELEVENLABS KEY Ở THANH BÊN TRÁI)",
                    type="primary",
                    use_container_width=True,
                    disabled=(not _has_el_key or not _has_rows)
                )
            with bc2:
                ready_mp3s = [r for r in st.session_state["eleven_script_rows"] if r.get("audio_path") and os.path.exists(str(r["audio_path"]))]
                if ready_mp3s:
                    z_buf = io.BytesIO()
                    with zipfile.ZipFile(z_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for r in ready_mp3s:
                            tc_c = str(r["Mốc Bắt Đầu"]).replace(":", "-").replace(",", "-")
                            ext = os.path.splitext(str(r["audio_path"]))[1] or ".wav"
                            zf.write(r["audio_path"], arcname=f"{int(r['STT']):03d}_{tc_c}{ext}")
                    st.download_button(f"📦 Tải Toàn Bộ ({len(ready_mp3s)} Voice .ZIP)", data=z_buf.getvalue(), file_name="ElevenLabs_v4_Audio_Pack.zip", mime="application/zip", use_container_width=True)
            with bc3:
                if st.button(
                    "🚀 BƯỚC C: GHÉP VOICE VÀO VIDEO NGAY" if (_has_vids and _has_voices) else "🔒 GHÉP VIDEO (CẦN TẠO VOICE TRƯỚC)",
                    use_container_width=True,
                    disabled=(not _has_vids or not _has_voices)
                ):
                    st.session_state["trigger_dub_only_render"] = True
                    st.rerun()

            step_b_hud = st.empty()
            if gen_all_eleven_clicked:
                t0_v = time.time()
                rows_gen = st.session_state["eleven_script_rows"]
                for ig, rg in enumerate(rows_gen):
                    slot_d = float(rg["Khung Giờ (s)"])
                    cur_tx = str(rg["Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)"])
                    row_vid = rg.get("row_voice_id") or eleven_voice_id
                    row_spd = float(rg.get("row_speed") or eleven_speed)

                    if strict_credit_saver and rg.get("audio_path") and os.path.exists(str(rg.get("audio_path", ""))):
                        continue
                    render_live_task_hud(step_b_hud, (ig + 0.5) / len(rows_gen), f"ĐANG TẠO VOICE ({eleven_model_id}) CÂU #{rg['STT']}/{len(rows_gen)}", f"Khóa chuẩn khung {slot_d}s...", t0_v)
                    res_tts = generate_elevenlabs_tts_segment(
                        elevenlabs_api_key, cur_tx, row_vid, eleven_model_id,
                        row_spd, eleven_stability, eleven_similarity, eleven_style,
                        force_regenerate=False, strip_tags_for_credit=(not keep_v4_emotive_tags_in_api)
                    )
                    if res_tts.get("ok"):
                        a_path_final, real_d = lock_voice_audio_to_exact_slot(res_tts["audio_path"], slot_d, int(rg["STT"]))
                        rg.update({
                            "audio_path": a_path_final,
                            "Voice Thực Tế (s)": real_d,
                            "Trạng Thái Khớp Nhịp": f"✅ Đã khớp 100% ({real_d}s / {slot_d}s)"
                        })
                step_b_hud.empty()
                st.rerun()

            # ==================================================================
            # BẢNG ĐIỀU KHIỂN TỪNG ĐOẠN THOẠI (CHỌN GIỌNG/TỐC ĐỘ RIÊNG TỪNG CÂU NHƯ FILE HTML)
            # ==================================================================
            with st.expander("🎬 ĐIỀU KHIỂN CHI TIẾT TỪNG ĐOẠN THOẠI (CHỌN GIỌNG, TỐC ĐỘ AI, THẺ CẢM XÚC v4 & XEM THỬ VIDEO)", expanded=False):
                act_vids_pv = _get_all_loaded_video_paths()
                voice_source = filtered_voices if filtered_voices else all_el_voices
                v_row_ids = [v["voice_id"] for v in voice_source]
                v_row_names = {v["voice_id"]: v["name"] for v in voice_source}

                for ip, rp in enumerate(st.session_state["eleven_script_rows"]):
                    s_sec_p = parse_timestamp_srt(str(rp["Mốc Bắt Đầu"]))
                    e_sec_p = parse_timestamp_srt(str(rp["Mốc Kết Thúc"]))
                    slot_d_p = int(round(max(5.0, e_sec_p - s_sec_p)))
                    cv_col, ca_col, cb_col = st.columns([0.30, 0.44, 0.26])

                    with cv_col:
                        if st.session_state.get(f"show_vid_clip_{ip}", False) and act_vids_pv:
                            clip_mp4 = create_synced_preview_video_clip(act_vids_pv, s_sec_p, e_sec_p, rp.get("audio_path", ""), int(rp["STT"]))
                            if clip_mp4 and os.path.exists(clip_mp4):
                                st.video(clip_mp4)
                        else:
                            img_show = rp.get("strip_path") or rp.get("frame_path")
                            if img_show and os.path.exists(img_show):
                                st.image(img_show, use_container_width=True)
                        st.caption(f"🎯 `#{rp['STT']:02d}` | `{rp['Mốc Bắt Đầu']} --> {rp['Mốc Kết Thúc']}` ({slot_d_p}s)")

                    with ca_col:
                        st.markdown(f"**VO #{rp['STT']:02d}** (`{rp['Số Từ']}/{rp['Tối Đa Từ Chuẩn']} từ` | Thực tế: `{rp['Voice Thực Tế (s)']}s`/`{rp['Khung Giờ (s)']}s`) — {rp['Trạng Thái Khớp Nhịp']}")
                        st.markdown(f"<div translate='no' class='notranslate' style='background:#0f172a;border:1px solid #6366f1;padding:8px 12px;border-radius:8px;font-size:13.5px;'>🇬🇧 {rp['Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)']}</div>", unsafe_allow_html=True)
                        if rp.get("audio_path") and os.path.exists(str(rp["audio_path"])):
                            st.audio(rp["audio_path"])
                            with open(rp["audio_path"], "rb") as f_aud_dl:
                                tc_fn = str(rp["Mốc Bắt Đầu"]).replace(":", "-").replace(",", "-")
                                st.download_button(
                                    f"⬇️ Tải Audio Câu #{rp['STT']}",
                                    data=f_aud_dl,
                                    file_name=f"{int(rp['STT']):03d}_{tc_fn}.wav",
                                    mime="audio/wav",
                                    key=f"dl_aud_row_{ip}"
                                )
                        st.markdown(f"<div style='background:#064e3b;padding:7px 11px;border-radius:8px;margin-top:5px;font-size:13px;'>🇻🇳 {rp.get('Phụ Đề Tiếng Việt (Dễ Hiểu)', '')}</div>", unsafe_allow_html=True)

                        for d_i, d_item in enumerate(rp.get("draft_candidates") or []):
                            st.markdown(f"<div translate='no' style='background:#1e1b4b;border:1px solid #facc15;padding:7px;border-radius:8px;margin-top:5px;font-size:12.5px;'><b>{d_item['style_label']}</b><br>🇬🇧 {d_item['en_text']}<br>🇻🇳 {d_item['vi_text']}</div>", unsafe_allow_html=True)
                            if st.button(f"✅ Chọn Bản Nháp {d_i+1} cho VO #{rp['STT']}", key=f"ap_dr_{ip}_{d_i}", use_container_width=True):
                                rp["Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)"] = d_item["en_text"]
                                rp["Phụ Đề Tiếng Việt (Dễ Hiểu)"] = d_item["vi_text"]
                                rp["draft_candidates"] = []
                                st.rerun()

                    with cb_col:
                        rc_v, rc_s = st.columns([0.62, 0.38])
                        cur_r_vid = rp.get("row_voice_id") or eleven_voice_id
                        with rc_v:
                            sel_r_vid = st.selectbox(
                                f"Giọng #{rp['STT']}:",
                                options=v_row_ids,
                                format_func=lambda k: v_row_names.get(k, k),
                                index=v_row_ids.index(cur_r_vid) if cur_r_vid in v_row_ids else 0,
                                key=f"r_vid_{ip}"
                            )
                            rp["row_voice_id"] = sel_r_vid
                        with rc_s:
                            sel_r_spd = st.number_input(
                                "Tốc độ:",
                                min_value=0.70, max_value=1.20,
                                value=float(rp.get("row_speed") or eleven_speed),
                                step=0.05,
                                key=f"r_spd_{ip}"
                            )
                            rp["row_speed"] = sel_r_spd

                        u_hint = st.text_input(f"Lệnh Tiếng Việt VO #{rp['STT']}:", value=rp.get("user_vi_cmd", ""), key=f"uh_{ip}", placeholder="VD: 2 người khiêng tấm xốp...")
                        if st.button(f"🧪 Tạo 3 Bản Nháp #{rp['STT']}", key=f"bd_{ip}", use_container_width=True):
                            rp["user_vi_cmd"] = u_hint.strip()
                            tw_1 = MANDATORY_DURATION_WORD_MAP.get(slot_d_p, int(round(slot_d_p * target_wps)))
                            dr_res = generate_multiple_vo_candidates_with_ai(
                                gemini_api_key, ip + 1, len(st.session_state["eleven_script_rows"]),
                                s_sec_p, e_sec_p, slot_d_p, tw_1, 4, 10, 45, rp.get("strip_path", ""),
                                auto_v4_tags=auto_inject_v4_tags, user_custom_hint=u_hint
                            )
                            rp["draft_candidates"] = dr_res.get("candidates", [])
                            st.rerun()
                        if st.button(f"🎬 Xem Video Câu #{rp['STT']}", key=f"bv_{ip}", use_container_width=True):
                            st.session_state[f"show_vid_clip_{ip}"] = True
                            st.rerun()
                        if st.button(f"🎙️ Tạo Audio Câu #{rp['STT']}", key=f"bt_{ip}", type="primary", use_container_width=True):
                            r1 = generate_elevenlabs_tts_segment(
                                elevenlabs_api_key, rp["Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)"],
                                rp["row_voice_id"], eleven_model_id, float(rp["row_speed"]), eleven_stability,
                                eleven_similarity, eleven_style, force_regenerate=True,
                                strip_tags_for_credit=(not keep_v4_emotive_tags_in_api)
                            )
                            if r1.get("ok"):
                                a_l, d_l = lock_voice_audio_to_exact_slot(r1["audio_path"], float(rp["Khung Giờ (s)"]), int(rp["STT"]))
                                rp["audio_path"], rp["Voice Thực Tế (s)"] = a_l, d_l
                                st.rerun()
                            else:
                                st.error(r1.get("error", "Lỗi tạo Audio"))
                    st.markdown("<hr style='margin:6px 0;border-color:rgba(99,102,241,0.25);'>", unsafe_allow_html=True)

    with el_tab_clone:
        st.markdown("##### 🧬 Trạm Clone Giọng Nhanh (Instant Voice Cloning — Tích hợp từ Eleven Labstool.html)")
        clc1, clc2, clc3 = st.columns(3)
        with clc1:
            clone_name = st.text_input("Tên giọng mới (Bắt buộc) *:", placeholder="VD: Quantum Narrator - Nam Trầm Documentary")
        with clc2:
            clone_gender = st.selectbox("Nhãn Giới tính:", ["male", "female"], format_func=lambda x: "Nam (Male)" if x == "male" else "Nữ (Female)")
        with clc3:
            clone_style = st.selectbox(
                "Nhãn Phong cách / Cảm xúc:",
                ["narrative", "calm", "deep", "raspy", "conversational"],
                format_func=lambda x: {
                    "narrative": "Kể chuyện (Narrative & Story)",
                    "calm": "Điềm tĩnh (Calm)",
                    "deep": "Trầm dày (Deep)",
                    "raspy": "Khàn (Raspy)",
                    "conversational": "Tự nhiên (Conversational)"
                }.get(x, x)
            )
        clone_desc = st.text_input("Mô tả đặc điểm giọng (Tùy chọn):", placeholder="VD: Deep American male voice for survival documentary...")
        clone_rm_noise = st.checkbox("Tự động lọc sạch tạp âm / nhạc nền nhỏ (Remove Background Noise)", value=True)
        clone_files_up = st.file_uploader("Chọn File âm thanh mẫu (.mp3, .wav, .m4a - Tốt nhất dài 1-3 phút) *:", type=["mp3", "wav", "m4a"], accept_multiple_files=True)
        if st.button("🚀 Bắt đầu Clone Giọng", type="primary", use_container_width=True):
            if not clone_name.strip() or not clone_files_up:
                st.warning("⚠️ Vui lòng nhập tên giọng và tải lên ít nhất 1 file âm thanh mẫu!")
            else:
                s_paths = [resolve_or_cache_fn(cf)[0] for cf in clone_files_up]
                r_cl = clone_elevenlabs_voice_api(elevenlabs_api_key, clone_name.strip(), clone_gender, clone_style, clone_desc.strip(), s_paths, clone_rm_noise)
                if r_cl.get("ok"):
                    st.success(f"✅ Clone thành công giọng '{clone_name}'! Đang đồng bộ danh sách giọng...")
                    conn_refresh = connect_elevenlabs_api(elevenlabs_api_key)
                    if conn_refresh.get("ok"):
                        st.session_state["eleven_voices_list"] = conn_refresh["voices"]
                else:
                    st.error(f"❌ Lỗi Clone: {r_cl.get('error')}")

    return {
        "eleven_model_id": eleven_model_id,
        "eleven_voice_id": eleven_voice_id,
        "eleven_speed": eleven_speed,
        "eleven_stability": eleven_stability,
        "eleven_similarity": eleven_similarity,
        "eleven_style": eleven_style,
        "target_wps": target_wps,
        "slot_fill_pct": slot_fill_pct,
        "voice_lead_ms": voice_lead_ms,
        "slot_align_mode": slot_align_mode,
        "auto_inject_v4_tags": auto_inject_v4_tags,
        "keep_v4_emotive_tags_in_api": keep_v4_emotive_tags_in_api,
        "auto_fit_voice_slot": auto_fit_voice_slot
    }