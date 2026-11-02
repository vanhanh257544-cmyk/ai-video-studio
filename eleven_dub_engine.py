import os
import re
import io
import json
import time
import wave
import base64
import hashlib
import zipfile
import subprocess
from typing import Optional, List, Dict, Any, Callable, Tuple
from datetime import datetime
import requests  # type: ignore
import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore
from moviepy.editor import AudioFileClip, VideoFileClip  # type: ignore
import streamlit as st  # type: ignore
import streamlit.components.v1 as components  # type: ignore

from audio_processor import AudioProcessor  # type: ignore
from video_editor import AutoVideoEditor  # type: ignore

try:
    import google.generativeai as genai  # type: ignore
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

CACHE_DIR = os.path.join(os.getcwd(), "studio_cache")
THUMB_CACHE_DIR = os.path.join(CACHE_DIR, "thumbnails")
ELEVEN_CACHE_DIR = os.path.join(CACHE_DIR, "elevenlabs_audio")
CLIP_PREVIEW_DIR = os.path.join(CACHE_DIR, "clip_previews")
VI_SUB_CACHE_FILE = os.path.join(CACHE_DIR, "vi_subtitles_cache.json")

for _d in (CACHE_DIR, THUMB_CACHE_DIR, ELEVEN_CACHE_DIR, CLIP_PREVIEW_DIR):
    os.makedirs(_d, exist_ok=True)

DEFAULT_ELEVEN_VOICES = [
    {"voice_id": "pNInz6obpgDQGcFmaJgB", "name": "Adam", "category": "premade", "gender": "male", "accent": "american", "use_case": "narrative", "description": "deep, resonant documentary", "preview_url": ""},
    {"voice_id": "nPczCjzI2devNBz1zQrb", "name": "Brian", "category": "premade", "gender": "male", "accent": "american", "use_case": "narrative", "description": "deep, resonant and comforting", "preview_url": ""},
    {"voice_id": "JBFqnCBsd6RMkjVDRZzb", "name": "George", "category": "premade", "gender": "male", "accent": "british", "use_case": "narrative", "description": "warm, raspy storyteller", "preview_url": ""},
    {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "category": "premade", "gender": "female", "accent": "american", "use_case": "calm", "description": "calm, matter-of-fact", "preview_url": ""},
    {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Sarah", "category": "premade", "gender": "female", "accent": "american", "use_case": "conversational", "description": "soft, warm, natural", "preview_url": ""},
    {"voice_id": "Xb7hH8MSUJpSbSDYk0k2", "name": "Alice", "category": "premade", "gender": "female", "accent": "british", "use_case": "narrative", "description": "clear, engaging educator", "preview_url": ""}
]

PAUSE_TAG_DURATIONS = {
    "takes a deep breath": 0.75,
    "deep breath": 0.70,
    "breathes deeply": 0.75,
    "sighs": 0.60,
    "sigh": 0.55,
    "exhales": 0.55,
    "short pause": 0.45,
    "pause": 0.60,
    "long pause": 0.90,
    "clears throat": 0.60
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
    # Khôi phục thẻ tiếng Anh nếu trình duyệt Chrome từng tự dịch thẻ sang tiếng Việt
    s = re.sub(r'\[\s*giọng\s+điệu\s+bình\s+tĩnh[^\]]*\]', '[calm, documentary tone]', s, flags=re.IGNORECASE)
    s = re.sub(r'^(?:EN|VI|VO\s*\d+)\s*:\s*', '', s, flags=re.IGNORECASE).strip()
    s = re.sub(r'\(\s*(?:Số từ|Target|\d+\s*từ|Actual).*?\)\s*$', '', s, flags=re.IGNORECASE).strip()
    return s


def strip_non_spoken_tags_for_credit_saving(text: str, strip_all_tags: bool = True) -> str:
    """
    TỐI ƯU TIẾT KIỆM CREDIT ELEVENLABS PRO:
    Loại bỏ các thẻ chỉ dẫn tông giọng như [calm, documentary tone] (tiết kiệm 24 ký tự/câu = ~450 credit/video)
    trước khi gửi lên API ElevenLabs vì các thông số Stability/Similarity/Style đã điều chỉnh giọng đọc!
    """
    s = _clean_srt_english_line(text)
    if strip_all_tags:
        s = re.sub(r'\[[^\]]+\]\s*', '', s)
    else:
        s = re.sub(r'\[\s*(?:calm|focused|slow|warm)[^\]]*tone\s*\]\s*', '', s, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', s).strip()


# ==============================================================================
# 1. ĐỒNG HỒ ĐẾM NGƯỢC & BẢNG HIỂN THỊ PHẦN TRĂM TRỰC TIẾP (LIVE HUD)
# ==============================================================================

def render_auto_confirm_countdown_widget(
    target_btn_text: str,
    step_label: str,
    timeout_sec: int = 180,
    enabled: bool = True
):
    if not enabled or timeout_sec <= 0:
        return

    t_btn_js = json.dumps(target_btn_text)
    s_lbl_js = json.dumps(step_label)
    html_box = f"""
    <div translate="no" class="notranslate" style="background:linear-gradient(90deg,#0f172a,#1e3a8a);border:2px solid #facc15;border-radius:12px;padding:10px 16px;color:#fff;font-family:'Segoe UI',sans-serif;display:flex;justify-content:space-between;align-items:center;box-shadow:0 4px 16px rgba(250,204,21,0.22);">
        <div>
            <div style="font-size:13.5px;font-weight:800;color:#facc15;">
                ⏳ CHỜ BẠN XÁC NHẬN (QUÁ {int(timeout_sec)} GIÂY KHÔNG BẤM SẼ TỰ ĐỘNG CHẠY TIẾP)
            </div>
            <div id="acDesc" style="font-size:12px;color:#e2e8f0;margin-top:2px;"></div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
            <div id="acTimer" style="font-size:22px;font-weight:900;color:#4ade80;font-family:Consolas,monospace;background:#020617;padding:4px 12px;border-radius:8px;border:1px solid #38bdf8;">
                03:00
            </div>
            <button id="acPauseBtn" onclick="toggleAc()" style="background:#be123c;color:#fff;border:1px solid #fda4af;padding:6px 12px;border-radius:8px;font-weight:800;font-size:12px;cursor:pointer;">
                ⏸️ Dừng Tự Động (Để Tôi Sửa)
            </button>
        </div>
    </div>
    <script>
        const targetTxt = {t_btn_js};
        let rem = {int(timeout_sec)};
        let paused = false;
        document.getElementById("acDesc").innerText = {s_lbl_js} + " → Tự kích hoạt [" + targetTxt + "] khi về 00:00.";
        function fmt(s) {{
            return String(Math.floor(s / 60)).padStart(2, '0') + ":" + String(s % 60).padStart(2, '0') + " (" + s + "s)";
        }}
        document.getElementById("acTimer").innerText = fmt(rem);
        function toggleAc() {{
            paused = !paused;
            const b = document.getElementById("acPauseBtn");
            const d = document.getElementById("acTimer");
            b.innerText = paused ? "▶️ Tiếp Tục Đếm Ngược" : "⏸️ Dừng Tự Động (Để Tôi Sửa)";
            b.style.background = paused ? "#15803d" : "#be123c";
            d.style.color = paused ? "#facc15" : "#4ade80";
        }}
        const iv = setInterval(() => {{
            if (paused) return;
            rem -= 1;
            const d = document.getElementById("acTimer");
            if (d) d.innerText = fmt(Math.max(0, rem));
            if (rem <= 0) {{
                clearInterval(iv);
                try {{
                    const btns = Array.from(window.parent.document.querySelectorAll("button"));
                    const found = btns.find(b => b.innerText && b.innerText.includes(targetTxt));
                    if (found) found.click();
                }} catch (e) {{}}
            }}
        }}, 1000);
    </script>
    """
    components.html(html_box, height=76, scrolling=False)


def render_live_task_hud(slot_placeholder: Any, pct: float, task_title: str, step_detail: str, start_t: float):
    pct_clamped = max(1, min(100, int(round(pct * 100))))
    elapsed = round(time.time() - start_t, 1)
    bar_color = "#4ade80" if pct_clamped >= 100 else "#38bdf8"

    hud_html = f"""
    <div translate="no" class="notranslate" style="background:linear-gradient(90deg,#0f172a,#1e3a8a);border:2px solid {bar_color};border-radius:12px;padding:10px 16px;margin:8px 0;box-shadow:0 4px 20px rgba(56,189,248,0.3);">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-weight:900;color:#facc15;font-size:14px;">⚙️ {task_title}</span>
            <span style="font-weight:900;color:{bar_color};font-size:18px;font-family:Consolas,monospace;">{pct_clamped}% ({elapsed}s)</span>
        </div>
        <div style="width:100%;height:10px;background:#020617;border-radius:999px;overflow:hidden;border:1px solid rgba(56,189,248,0.4);">
            <div style="width:{pct_clamped}%;height:100%;background:linear-gradient(90deg,#0284c7,{bar_color});transition:width 0.2s ease;"></div>
        </div>
        <div style="font-size:12.5px;color:#e2e8f0;margin-top:5px;font-weight:600;">👉 {step_detail}</div>
    </div>
    """
    slot_placeholder.markdown(hud_html, unsafe_allow_html=True)


# ==============================================================================
# 2. KẾT NỐI ELEVENLABS PRO API & BẢO VỆ CREDIT TUYỆT ĐỐI
# ==============================================================================

def connect_elevenlabs_api(api_key: str) -> Dict[str, Any]:
    clean_k = clean_eleven_api_key(api_key)
    if not clean_k:
        return {"ok": False, "error": "Chưa nhập ElevenLabs API Key"}
    headers = {"xi-api-key": clean_k}
    res_out: Dict[str, Any] = {
        "ok": False, "voices": DEFAULT_ELEVEN_VOICES.copy(),
        "tier": "PRO", "char_used": 0, "char_limit": 0, "error": ""
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
    for candidate_txt in (spoken_clean, str(text).strip()):
        h_exact = f"{candidate_txt}_{voice_id}_{model_id}_{speed:.2f}_{stability}_{similarity}_{style}"
        p_exact_wav = os.path.join(ELEVEN_CACHE_DIR, f"eleven_{hashlib.md5(h_exact.encode('utf-8')).hexdigest()[:16]}.wav")
        if os.path.exists(p_exact_wav) and os.path.getsize(p_exact_wav) > 512:
            return p_exact_wav
        p_exact_mp3 = os.path.join(ELEVEN_CACHE_DIR, f"eleven_{hashlib.md5(h_exact.encode('utf-8')).hexdigest()[:16]}.mp3")
        if os.path.exists(p_exact_mp3) and os.path.getsize(p_exact_mp3) > 512:
            return p_exact_mp3
        h_text_only = f"{candidate_txt}_{voice_id}"
        p_text = os.path.join(ELEVEN_CACHE_DIR, f"eleven_txt_{hashlib.md5(h_text_only.encode('utf-8')).hexdigest()[:16]}.wav")
        if os.path.exists(p_text) and os.path.getsize(p_text) > 512:
            return p_text
    return ""


def generate_elevenlabs_tts_segment(
    api_key: str, text: str, voice_id: str, model_id: str = "eleven_v4",
    speed: float = 1.0, stability: int = 45, similarity: int = 80,
    style: int = 15, force_regenerate: bool = False, strip_tags_for_credit: bool = True
) -> Dict[str, Any]:
    clean_k = clean_eleven_api_key(api_key)
    if not clean_k:
        return {"ok": False, "error": "Chưa nhập ElevenLabs API Key"}

    # Tự động gỡ bỏ thẻ [calm, documentary tone] để tiết kiệm 24 ký tự Credit mỗi câu
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

    payload = {
        "text": api_text,
        "model_id": model_id,
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
        if res.status_code != 200 and ("v4" in model_id or "v3" in model_id):
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
# 3. ĐẾM TỪ KHỚP NHỊP 100% & BỘ PHÂN TÍCH LỆNH TIẾNG VIỆT TRỰC TIẾP
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
    slot_fill_pct: int = 100,
    breath_gap_sec: float = 0.0,
    punct_pause_ms: int = 0
) -> Dict[str, Any]:
    tag_count, pause_sec, _ = compute_emotion_tags_duration(text)
    spoken_text = re.sub(r'\[[^\]]+\]', '', str(text)).strip()
    words = count_clean_english_words(spoken_text)
    word_count = len(words)
    char_count = len(spoken_text)

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

    tag_info_str = f" | 🎭 {tag_count} thẻ ({pause_sec}s)" if tag_count > 0 else ""
    if is_over:
        status = f"⚠️ Dài hơn khung (+{max(0.1, diff_sec):.1f}s | {word_count}/{ideal_max_words} từ{tag_info_str} - Cần bớt {max(1, word_count - ideal_max_words)} từ)"
    elif is_under:
        status = f"⚠️ Ngắn hơn khung ({diff_sec:.1f}s | {word_count}/{ideal_max_words} từ{tag_info_str} - Cần thêm {max(1, ideal_max_words - word_count)} từ)"
    else:
        status = f"✅ Chuẩn khớp 100% ({slot_duration_sec:.1f}s/{slot_duration_sec:.1f}s | {word_count}/{ideal_max_words} từ{tag_info_str})"

    return {
        "word_count": word_count,
        "char_count": char_count,
        "punct_count": len(re.findall(r'[,.;:?!—\-]', spoken_text)),
        "tag_count": tag_count,
        "emotion_sec": pause_sec,
        "est_sec": round(slot_duration_sec, 2) if (not is_over and not is_under) else est_sec,
        "ms_per_word": ms_per_word,
        "ideal_max_words": ideal_max_words,
        "diff_sec": 0.0 if (not is_over and not is_under) else diff_sec,
        "is_over_time": is_over,
        "is_under_time": is_under,
        "status": status
    }


def _call_gemini_vision_rest_and_sdk(gemini_key: str, prompt: str, image_path: str = "", json_mode: bool = False) -> str:
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

    gen_cfg: Dict[str, Any] = {"temperature": 0.45, "maxOutputTokens": 8192}
    if json_mode:
        gen_cfg["responseMimeType"] = "application/json"

    payload = {
        "contents": [{"parts": parts_list}],
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ],
        "generationConfig": gen_cfg
    }

    candidate_models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.5-pro",
        "gemini-3.1-pro-preview",
        "gemini-1.5-flash"
    ]

    for m_name in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={clean_k}"
        try:
            r = requests.post(url, json=payload, timeout=28)
            if r.status_code == 200:
                data = r.json()
                cands = data.get("candidates") or []
                if cands:
                    content_parts = (cands[0].get("content") or {}).get("parts") or []
                    text_chunks = [p.get("text", "") for p in content_parts if not p.get("thought") and p.get("text")]
                    out_txt = "\n".join(text_chunks).strip()
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
        (r"\btrên\s+boong\b", "trên mặt sàn gỗ"),
        (r"\bsàn\s+công\s+tác\b", "mặt sàn thi công"),
        (r"\bboong\s+phụ\b", "mặt sàn phụ"),
        (r"\bsàn\s+phụ\b", "mặt sàn khung dưới"),
        (r"\bđinh\s+tán\s+bằng\s+gỗ\b", "các thanh trụ gỗ vách"),
        (r"\bđinh\s+tán\s+gỗ\b", "thanh trụ gỗ"),
        (r"\bcác\s+đinh\s+tán\b", "các thanh trụ đứng"),
        (r"\bcách\s+nhiệt\s+bằng\s+bọt\s+lớn\b", "xốp cách nhiệt cỡ lớn"),
        (r"\bcách\s+nhiệt\s+bằng\s+bọt\b", "xốp cách nhiệt"),
        (r"\btấm\s+bọt\b", "tấm xốp cách nhiệt"),
        (r"\bngười\s+xây\s+dựng\b", "người thợ"),
        (r"\bnhà\s+xây\s+dựng\b", "người thợ"),
        (r"\bngười\s+vận\s+hành\b", "người thợ"),
        (r"\bngười\s+thợ\s+thủ\s+công\b", "người thợ"),
        (r"\bĐẢM\s+BẢO\s+các\s+tấm\b", "bắt chặt các tấm"),
        (r"\bBẢO\s+VỆ\b", "cố định chặt"),
        (r"\bPOSITIONS\b", "đặt"), (r"\bPLACES\b", "đặt"), (r"\bFASTENS\b", "bắt chặt"),
        (r"\bALIGNS\b", "căn chỉnh"), (r"\bINSTALLS\b", "lắp đặt"), (r"\bSECURES\b", "cố định"),
        (r"\bLIFTS\b", "nâng"), (r"\bTRIMS\b", "cắt gọt"), (r"\bSEALS\b", "dán kín"),
        (r"\bFITS\b", "lắp khít"), (r"\bLAYERS\b", "trải đều"), (r"\bSCREWS\b", "bắn vít"),
        (r"\bRAISES\b", "dựng"), (r"\bLEVELS\b", "cân bằng"), (r"\bBRACES\b", "giằng cố định"),
        (r"\bMEASURES\b", "đo đạc"), (r"\bSPREADS\b", "rải đều"), (r"\bNAILS\b", "đóng đinh"),
        (r"\bLOCKS\b", "khóa chặt"), (r"\bHOISTS\b", "nâng"), (r"\bMOUNTS\b", "lắp"),
        (r"\bSAWS\b", "cưa"), (r"\bCUTS\b", "cắt"), (r"\bCARRIES\b", "khiêng"),
        (r"\bngay\s+hôm\s+nay\b", ""), (r"\bngày\s+hôm\s+nay\b", "")
    ]
    for pat, rep in replacements:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)

    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'\s+([,.;?!])', r'\1', s)
    if s and not s.endswith((".", "!", "?")):
        s += "."
    return s[0].upper() + s[1:] if s else s


def translate_to_vietnamese_subtitle(gemini_key: str, text_en: str, existing_vi: str = "", force_refresh: bool = False) -> str:
    clean_src = _clean_srt_english_line(re.sub(r'\[[^\]]+\]', '', str(text_en)))
    if not clean_src or clean_src.startswith("[Auto VO"):
        return "Chưa có nội dung thuyết minh."

    bad_vi_markers = [
        "boong chính", "đinh tán", "bằng bọt", "bảo vệ các tấm", "vật liệu chính",
        "công cụ chuyên dụng", "người vận hành", "đo đạc, căn chỉnh và cố định chặt từng thanh gỗ vào khung nhà"
    ]
    if not force_refresh and existing_vi and _has_vn_chars(existing_vi):
        if not any(bm in existing_vi.lower() for bm in bad_vi_markers):
            return _refine_vietnamese_construction_terms(existing_vi)

    if _has_vn_chars(clean_src):
        return _refine_vietnamese_construction_terms(clean_src)

    words_for_trans = []
    for idx_w, w in enumerate(clean_src.split()):
        w_alpha = re.sub(r'[^\w]', '', w)
        if w_alpha.isupper() and len(w_alpha) > 1:
            words_for_trans.append(w.capitalize() if idx_w == 0 else w.lower())
        else:
            words_for_trans.append(w)
    normalized_en = " ".join(words_for_trans)

    cache_db = {}
    if os.path.exists(VI_SUB_CACHE_FILE):
        try:
            with open(VI_SUB_CACHE_FILE, "r", encoding="utf-8") as f:
                cache_db = json.load(f)
        except Exception:
            cache_db = {}

    if not force_refresh and normalized_en in cache_db and _has_vn_chars(cache_db[normalized_en]):
        cached_v = cache_db[normalized_en]
        if not any(bm in cached_v.lower() for bm in bad_vi_markers):
            return _refine_vietnamese_construction_terms(cached_v)

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36"}
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
                    vi_refined = _refine_vietnamese_construction_terms(vi_out)
                    cache_db[normalized_en] = vi_refined
                    try:
                        with open(VI_SUB_CACHE_FILE, "w", encoding="utf-8") as f:
                            json.dump(cache_db, f, indent=2, ensure_ascii=False)
                    except Exception:
                        pass
                    return vi_refined
    except Exception:
        pass

    vi_gem = _call_gemini_vision_rest_and_sdk(
        gemini_key,
        f"Dịch câu thuyết minh xây dựng sau sang tiếng Việt tự nhiên, chuẩn xác (chỉ trả về duy nhất câu tiếng Việt): \"{normalized_en}\""
    )
    if _has_vn_chars(vi_gem):
        vi_clean = _refine_vietnamese_construction_terms(vi_gem.strip().strip('"').strip("'"))
        cache_db[normalized_en] = vi_clean
        try:
            with open(VI_SUB_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache_db, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        return vi_clean

    return _refine_vietnamese_construction_terms(normalized_en)


# ==============================================================================
# 4. BỘ PHÂN TÍCH LỆNH TIẾNG VIỆT TRỰC TIẾP & KIẾN TẠO 3 BẢN NHÁP SÁT HÀNH ĐỘNG
# ==============================================================================

def get_video_timeline_info(video_paths: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
    segments_meta = []
    cum_t = 0.0
    for vp in video_paths:
        if not vp or not os.path.exists(vp):
            continue
        dur = AutoVideoEditor._get_media_duration_fast(vp)
        if dur > 0:
            segments_meta.append({"video_path": vp, "start_global": cum_t, "end_global": cum_t + dur, "duration": dur})
            cum_t += dur
    return (cum_t if cum_t > 0 else 120.0), segments_meta


def compute_video_fingerprint_signature(video_paths: List[str]) -> str:
    if not video_paths:
        return "no_video"
    parts = []
    for p in video_paths:
        if p and os.path.exists(p):
            try:
                sz, mt = os.path.getsize(p), int(os.path.getmtime(p))
            except Exception:
                sz, mt = 0, 0
            parts.append(f"{os.path.basename(p)}_{sz}_{mt}")
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:16] if parts else "no_video"


def extract_slot_filmstrip_from_video(video_path_or_list: Any, start_sec: float, end_sec: float, idx_tag: int) -> Tuple[str, str]:
    v_list = [p for p in video_path_or_list if isinstance(p, str) and os.path.exists(p)] if isinstance(video_path_or_list, list) else ([video_path_or_list] if isinstance(video_path_or_list, str) and os.path.exists(video_path_or_list) else [])
    if not v_list:
        return "", ""
    sig_v = compute_video_fingerprint_signature(v_list)[:8]
    thumb_jpg = os.path.join(THUMB_CACHE_DIR, f"grid6_thumb_{sig_v}_{idx_tag:04d}_{int(start_sec * 10)}_{int(end_sec * 10)}.jpg")
    strip_jpg = os.path.join(THUMB_CACHE_DIR, f"grid6_strip_{sig_v}_{idx_tag:04d}_{int(start_sec * 10)}_{int(end_sec * 10)}.jpg")
    if os.path.exists(thumb_jpg) and os.path.exists(strip_jpg):
        return thumb_jpg, strip_jpg

    _, seg_metas = get_video_timeline_info(v_list)
    if not seg_metas:
        return "", ""

    dur = max(1.0, end_sec - start_sec)
    sample_times = [start_sec + dur * r for r in (0.06, 0.23, 0.41, 0.59, 0.77, 0.94)]
    frames = []
    for i_t, gt in enumerate(sample_times):
        chosen_seg = next((sm for sm in seg_metas if sm["start_global"] <= gt <= sm["end_global"]), seg_metas[-1])
        local_t = max(0.0, min(chosen_seg["duration"] - 0.05, gt - chosen_seg["start_global"]))
        tmp_f = os.path.join(THUMB_CACHE_DIR, f"_tmp_g6_{sig_v}_{idx_tag}_{i_t}.jpg")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{local_t:.3f}", "-i", chosen_seg["video_path"], "-frames:v", "1", "-vf", "scale=480:270:flags=lanczos", "-q:v", "2", tmp_f],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15
            )
            if os.path.exists(tmp_f):
                fr = cv2.imread(tmp_f)
                os.remove(tmp_f)
                if fr is not None:
                    cv2.rectangle(fr, (6, 6), (245, 36), (0, 0, 0), -1)
                    cv2.putText(fr, f"Hinh #{i_t+1} (+{gt - start_sec:.1f}s)", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)
                    frames.append(fr)
        except Exception:
            pass

    if frames:
        cv2.imwrite(thumb_jpg, frames[len(frames) // 2], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if len(frames) == 6:
            grid6 = cv2.vconcat([cv2.hconcat(frames[:3]), cv2.hconcat(frames[3:])])
            cv2.imwrite(strip_jpg, grid6, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            cv2.imwrite(strip_jpg, cv2.hconcat(frames), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return thumb_jpg, strip_jpg
    return "", ""


def parse_vietnamese_command_and_visuals(
    user_vi_cmd: str,
    strip_image_path: str,
    vo_index: int,
    start_sec: float
) -> Dict[str, Any]:
    """
    HÀM ĐỌC HIỂU CÂU LỆNH TIẾNG VIỆT CỦA NGƯỜI DÙNG + HÌNH ẢNH THỰC TẾ TRONG VIDEO:
    Khi bạn gõ lệnh tiếng Việt bên cạnh VO (VD: '2 người khiêng tấm xốp trắng đặt lên sàn cạnh thang đỏ'
    hoặc 'dùng máy khoan bắn vít thanh gỗ'), hàm này bóc tách chính xác:
    - Chủ thể (1 người thợ / 2 người phối hợp)
    - Dụng cụ & Bối cảnh (thang chữ A đỏ, máy bắn vít, cưa, búa, thước...)
    - Vật liệu (tấm xốp cách nhiệt trắng, khung vách gỗ đứng, dầm sàn, ván sàn...)
    - Động từ hành động chuẩn (CARRIES, PLACES, FASTENS, ALIGNS, SAWS, INSTALLS...)
    """
    cmd_low = str(user_vi_cmd or "").strip().lower()

    # 1. Phân tích hình ảnh 6 khung hình làm nền tảng (Không nhảy sang giai đoạn khác khi bấm tạo lại!)
    has_white_foam = False
    has_red_ladder = False
    if strip_image_path and os.path.exists(strip_image_path):
        try:
            img = cv2.imread(strip_image_path)
            if img is not None:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                white_ratio = float(np.mean((hsv[:, :, 1] < 42) & (hsv[:, :, 2] > 185)))
                red_ratio = float(np.mean(((hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 168)) & (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 100)))
                has_white_foam = (white_ratio > 0.11)
                has_red_ladder = (red_ratio > 0.010)
        except Exception:
            pass

    # 2. Xác định kịch bản gốc theo đúng diễn biến thật của cảnh (hoặc theo câu lệnh tiếng Việt bạn nhập)
    if cmd_low:
        # Bóc tách Chủ thể từ lệnh Tiếng Việt
        if any(k in cmd_low for k in ["2 người", "hai người", "cùng nhau", "phụ nữ", "vợ", "đồng đội", "nhóm", "phối hợp", "khiêng"]):
            subj_en = "working together beside the timber frame, the two builders"
            subj_vi = "Phối hợp nhịp nhàng bên khung nhà gỗ, hai người thợ"
            is_plural = True
        else:
            subj_en = "working with steady focus on the wooden deck, the builder"
            subj_vi = "Làm việc tập trung trên mặt sàn gỗ, người thợ"
            is_plural = False

        # Bóc tách Dụng cụ từ lệnh Tiếng Việt
        if any(k in cmd_low for k in ["thang đỏ", "thang chữ a", "chiếc thang", "thang"]):
            tool_en, tool_vi = "near the red step ladder", "ngay cạnh chiếc thang chữ A màu đỏ"
        elif any(k in cmd_low for k in ["bắn vít", "máy khoan", "khoan pin", "súng bắn vít", "khoan"]):
            tool_en, tool_vi = "using a cordless power drill", "bằng máy khoan bắn vít cầm tay"
        elif any(k in cmd_low for k in ["máy cưa", "cưa đĩa", "cưa gỗ", "cưa", "cắt"]):
            tool_en, tool_vi = "guiding a handheld circular saw", "bằng máy cưa đĩa cầm tay"
        elif any(k in cmd_low for k in ["búa", "đóng đinh", "gõ"]):
            tool_en, tool_vi = "swinging a heavy carpenter hammer", "bằng chiếc búa mộc cầm tay"
        elif any(k in cmd_low for k in ["thước thủy", "thước dây", "thước", "đo"]):
            tool_en, tool_vi = "checking with a carpenter level and tape", "kết hợp kiểm tra bằng thước đo chuẩn xác"
        elif any(k in cmd_low for k in ["băng keo", "dán"]):
            tool_en, tool_vi = "pressing weatherproof sealing tape by hand", "bằng cuộn băng keo chống thấm chuyên dụng"
        else:
            tool_en = "near the red step ladder" if has_red_ladder else "with steady hands on the platform"
            tool_vi = "bên cạnh chiếc thang chữ A màu đỏ" if has_red_ladder else "với đôi tay khéo léo trên mặt sàn"

        # Bóc tách Vật liệu từ lệnh Tiếng Việt
        if any(k in cmd_low for k in ["xốp", "cách nhiệt", "tấm trắng", "bọt", "panel"]):
            mat_en, mat_vi = "thick white foam insulation boards", "các tấm xốp cách nhiệt màu trắng dày dặn"
            purpose_en = "between the wooden floor joists so the cabin floor stays warm and dry"
            purpose_vi = "vào giữa các khoang dầm sàn để nền nhà gỗ luôn ấm áp và khô ráo"
        elif any(k in cmd_low for k in ["vách", "khung tường", "cột đứng", "tường gỗ", "dựng khung"]):
            mat_en, mat_vi = "upright wooden wall studs", "các thanh trụ gỗ của khung vách đứng"
            purpose_en = "along the outer deck plate so the cabin walls stand straight and solid"
            purpose_vi = "dọc theo dầm biên mặt sàn để hệ khung vách nhà đứng thẳng và vững chắc"
        elif any(k in cmd_low for k in ["cửa sổ", "khung cửa", "lỗ mở", "cửa"]):
            mat_en, mat_vi = "framed wooden window timbers", "các thanh gỗ tạo khung ô cửa sổ"
            purpose_en = "between the vertical wall posts so the window opening stays square and level"
            purpose_vi = "giữa các cột trụ vách để ô cửa sổ vuông vức và chịu lực tốt"
        elif any(k in cmd_low for k in ["ván sàn", "lát sàn", "tấm ván", "mặt sàn"]):
            mat_en, mat_vi = "wide wooden subfloor panels", "các tấm ván gỗ lát mặt sàn"
            purpose_en = "across the spaced floor joists to create a flat and sturdy working deck"
            purpose_vi = "phủ kín hệ dầm ngang nhằm tạo mặt sàn làm việc phẳng phiu, đầm chắc"
        elif any(k in cmd_low for k in ["mái", "kèo mái", "xà gồ", "đòn dông"]):
            mat_en, mat_vi = "angled wooden roof rafters", "các thanh kèo gỗ của hệ mái dốc"
            purpose_en = "high above the top wall plates to shed heavy mountain rain and snow"
            purpose_vi = "phía trên đỉnh tường để tạo độ dốc thoát nước mưa và chống tuyết đọng"
        else:
            mat_en, mat_vi = "heavy structural timber boards", "các cấu kiện gỗ kết cấu chính"
            purpose_en = "into exact alignment across the frame so every joint locks together tightly"
            purpose_vi = "vào đúng vị trí trên khung nhà giúp mọi mối ghép liên kết chặt chẽ"

        # Bóc tách Động từ hành động từ lệnh Tiếng Việt
        if any(k in cmd_low for k in ["khiêng", "bê", "vác", "mang", "di chuyển"]):
            verbs_3 = [("CARRIES", "khiêng"), ("LIFTS", "nâng"), ("PLACES", "đặt")]
        elif any(k in cmd_low for k in ["cưa", "cắt", "gọt"]):
            verbs_3 = [("SAWS", "cưa"), ("TRIMS", "cắt gọt"), ("FITS", "lắp khít")]
        elif any(k in cmd_low for k in ["bắn vít", "khoan", "bắt vít", "cố định"]):
            verbs_3 = [("FASTENS", "bắt chặt"), ("SCREWS", "bắn vít"), ("SECURES", "cố định")]
        elif any(k in cmd_low for k in ["đo", "căn", "chỉnh"]):
            verbs_3 = [("ALIGNS", "căn chỉnh"), ("MEASURES", "đo đạc"), ("LEVELS", "cân bằng")]
        else:
            verbs_3 = [("PLACES", "đặt"), ("FITS", "lắp khít"), ("SECURES", "cố định")]

        opt_list = []
        for idx_v, (v_en, v_vi) in enumerate(verbs_3):
            if idx_v == 0:
                en_s = f"{subj_en.capitalize()} {v_en} the {mat_en} {tool_en} {purpose_en}, checking each edge carefully to ensure a tight and accurate fit across the platform."
                vi_s = f"{subj_vi} {v_vi} {mat_vi} {tool_vi} {purpose_vi}, đồng thời kiểm tra kỹ từng mép nối để đảm bảo độ khít chuẩn xác trên mặt sàn."
                lbl_s = "Bản nháp 1: Bám sát 100% Lệnh Tiếng Việt của bạn (Trực diện Hành động)"
            elif idx_v == 1:
                en_s = f"Moving steadily {tool_en}, {'the two builders' if is_plural else 'the carpenter'} {v_en} each section of {mat_en} {purpose_en} while keeping every structural joint flush, square, and rock solid."
                vi_s = f"Thao tác nhịp nhàng {tool_vi}, {'hai người thợ' if is_plural else 'người thợ mộc'} {v_vi} từng phần {mat_vi} {purpose_vi}, giữ cho mọi khớp nối luôn phẳng, vuông góc và chắc chắn."
                lbl_s = "Bản nháp 2: Bám sát Lệnh Tiếng Việt (Nhấn mạnh Kỹ thuật & Độ khít)"
            else:
                en_s = f"Under the bright forest canopy {tool_en}, {'the crew' if is_plural else 'the craftsman'} {v_en} the {mat_en} firmly into place {purpose_en}, advancing the woodland cabin build with patient precision."
                vi_s = f"Dưới tán rừng xanh mát {tool_vi}, {'nhóm thợ' if is_plural else 'người thợ'} {v_vi} {mat_vi} vào đúng vị trí {purpose_vi}, từng bước hoàn thiện ngôi nhà gỗ với sự tỉ mỉ."
                lbl_s = "Bản nháp 3: Bám sát Lệnh Tiếng Việt (Phong cách Kể chuyện Documentary)"
            opt_list.append({"style_label": lbl_s, "anchor_verb": v_en, "en": en_s, "vi": vi_s})

        return {
            "tools_vi": f"Theo lệnh của bạn ({tool_vi})",
            "mats_vi": mat_vi,
            "action_vi": user_vi_cmd.strip(),
            "tool_en": tool_en,
            "mat_en": mat_en,
            "options": opt_list
        }

    # 3. Khi không nhập lệnh tay: Tự động chọn theo đúng đặc trưng thị giác của khung hình hiện tại (Không nhảy lung tung!)
    if has_white_foam and has_red_ladder:
        return {
            "tools_vi": "Thang chữ A màu đỏ, bàn kê xanh & thao tác khiêng tay",
            "mats_vi": "Tấm xốp cách nhiệt trắng cỡ lớn & khung vách gỗ đứng",
            "action_vi": "Hai người phối hợp khiêng và lắp các tấm xốp cách nhiệt trắng bên cạnh thang đỏ",
            "tool_en": "beside the red step ladder",
            "mat_en": "white foam insulation boards",
            "options": [
                {
                    "style_label": "Phương án 1: Sát thực tế từng dụng cụ & cử chỉ trong ảnh",
                    "anchor_verb": "CARRIES",
                    "en": "Working together beside the red step ladder and timber wall frame, the builder CARRIES thick white foam insulation boards onto the wooden platform while his partner guides each sheet cleanly into place.",
                    "vi": "Phối hợp cùng nhau bên cạnh chiếc thang chữ A màu đỏ và khung tường gỗ, người thợ khiêng các tấm xốp cách nhiệt màu trắng lên mặt sàn trong khi người đồng hành đỡ từng tấm vào đúng vị trí."
                },
                {
                    "style_label": "Phương án 2: Phân tích Kỹ thuật lắp ghép & Cách nhiệt",
                    "anchor_verb": "PLACES",
                    "en": "Standing on the sunlit clearing beside the framed wooden wall, the crew PLACES large rigid foam insulation panels across the deck bays to seal the cabin structure against ground cold and moisture.",
                    "vi": "Đứng dưới nắng bên cạnh khung vách gỗ vừa dựng, nhóm thợ đặt các tấm xốp cách nhiệt cứng cỡ lớn vào các khoang sàn nhằm ngăn hơi lạnh và độ ẩm từ mặt đất thấm lên nhà."
                },
                {
                    "style_label": "Phương án 3: Phong cách Kể chuyện Documentary",
                    "anchor_verb": "FITS",
                    "en": "Beneath the green summer pines near the red ladder, the craftsman FITS lightweight white insulation sheets into the timber framework with his partner, building a warm foundation for the woodland cabin.",
                    "vi": "Dưới tán thông xanh mùa hè cạnh chiếc thang đỏ, người thợ cùng bạn đồng hành lắp khít từng tấm xốp cách nhiệt trắng vào hệ khung gỗ, tạo nền tảng ấm áp cho ngôi nhà giữa rừng."
                }
            ]
        }
    elif has_white_foam:
        return {
            "tools_vi": "Dao rọc, thước đo & thao tác ép tấm xốp trực tiếp",
            "mats_vi": "Tấm xốp cách nhiệt trắng & hệ dầm sàn gỗ",
            "action_vi": "Đặt và ấn phẳng các tấm xốp cách nhiệt trắng lấp đầy các khoang dầm sàn",
            "tool_en": "across the framed floor joists",
            "mat_en": "white foam insulation panels",
            "options": [
                {
                    "style_label": "Phương án 1: Sát thực tế từng dụng cụ & cử chỉ trong ảnh",
                    "anchor_verb": "PLACES",
                    "en": "Working across the raised wooden foundation deck, the builder PLACES thick white foam insulation panels between the floor joists and presses each sheet down flat so the cabin floor stays warm and dry.",
                    "vi": "Làm việc trên mặt sàn móng gỗ nâng cao, người thợ đặt các tấm xốp cách nhiệt màu trắng dày vào giữa các thanh dầm sàn và ấn phẳng từng tấm xuống để nền nhà luôn ấm áp, khô ráo."
                },
                {
                    "style_label": "Phương án 2: Phân tích Kỹ thuật lắp ghép & Cách nhiệt",
                    "anchor_verb": "FITS",
                    "en": "Moving methodically along the open subfloor grid, the carpenter FITS rigid thermal insulation boards tightly into every joist bay by hand, blocking rising ground dampness beneath the future living space.",
                    "vi": "Di chuyển nhịp nhàng dọc theo hệ khung dầm sàn mở, người thợ mộc lắp khít các tấm xốp cách nhiệt cứng vào từng khoang dầm bằng tay, chặn đứng hơi ẩm từ mặt đất bốc lên."
                },
                {
                    "style_label": "Phương án 3: Phong cách Kể chuyện Documentary",
                    "anchor_verb": "LAYERS",
                    "en": "Under the bright forest canopy, the craftsman LAYERS reflective white insulation sheets across the wide timber platform, creating an unbroken thermal barrier that protects the woodland shelter through winter.",
                    "vi": "Dưới tán rừng rực nắng, người thợ trải đều các tấm xốp cách nhiệt trắng lên khắp mặt sàn gỗ rộng lớn, tạo lớp chắn giữ nhiệt liền mạch bảo vệ ngôi nhà gỗ suốt mùa đông."
                }
            ]
        }

    # Các cảnh khung gỗ / dựng vách / đóng sàn theo tiến trình
    stage_defaults = [
        {
            "tools_vi": "Máy bắn vít cầm tay, thước thủy & búa mộc",
            "mats_vi": "Khung cột gỗ vách đứng 2x4 & thanh dầm ngang",
            "action_vi": "Căn chỉnh các thanh trụ gỗ đứng và bắn vít cố định vào hệ khung vách",
            "tool_en": "with a cordless power drill",
            "mat_en": "upright wooden wall studs",
            "options": [
                {
                    "style_label": "Phương án 1: Sát thực tế từng dụng cụ & cử chỉ",
                    "anchor_verb": "FASTENS",
                    "en": "Standing beside the upright timber wall section, the carpenter FASTENS vertical wooden studs to the horizontal plate with a cordless drill while checking that every post stands straight and plumb.",
                    "vi": "Đứng bên cạnh phần khung vách gỗ dựng đứng, người thợ mộc bắt chặt các thanh trụ gỗ dọc vào thanh dầm ngang bằng máy khoan pin đồng thời kiểm tra để mỗi cột đều thẳng đứng."
                },
                {
                    "style_label": "Phương án 2: Phân tích Kỹ thuật & Kết cấu",
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
        },
        {
            "tools_vi": "Máy khoan bắn vít, cưa đĩa & thước dây",
            "mats_vi": "Tấm ván sàn gỗ & hệ dầm chịu lực",
            "action_vi": "Lắp ghép các tấm ván gỗ mặt sàn và bắn vít cố định xuống hệ dầm",
            "tool_en": "using a cordless impact driver",
            "mat_en": "wooden subfloor boards",
            "options": [
                {
                    "style_label": "Phương án 1: Sát thực tế từng dụng cụ & cử chỉ",
                    "anchor_verb": "SECURES",
                    "en": "Kneeling on the framed wooden platform, the builder SECURES wide subfloor boards to the underlying joists with a cordless impact driver so the entire walking deck feels solid underfoot.",
                    "vi": "Quỳ gối trên mặt sàn khung gỗ, người thợ cố định các tấm ván sàn bản rộng xuống hệ dầm bên dưới bằng máy bắn vít pin giúp toàn bộ bề mặt sàn đi lại luôn đầm chắc."
                },
                {
                    "style_label": "Phương án 2: Phân tích Kỹ thuật & Độ phẳng",
                    "anchor_verb": "LOCKS",
                    "en": "Sliding the board edges tightly together across the deck, the carpenter LOCKS each heavy wooden floor panel down along the joist lines to eliminate any gap or flex.",
                    "vi": "Đẩy sát các mép ván khít lại với nhau trên mặt sàn, người thợ mộc khóa chặt từng tấm ván sàn gỗ nặng dọc theo đường dầm nhằm loại bỏ hoàn toàn khe hở hay độ võng."
                },
                {
                    "style_label": "Phương án 3: Phong cách Kể chuyện Documentary",
                    "anchor_verb": "INSTALLS",
                    "en": "Working with steady rhythm across the clearing, the craftsman INSTALLS solid timber decking over the foundation grid, completing a clean and level stage for the cabin walls above.",
                    "vi": "Làm việc với nhịp điệu đều đặn giữa khoảng rừng thoáng đãng, người thợ lắp đặt mặt sàn gỗ đặc lên trên hệ khung móng, hoàn thiện mặt phẳng vững vàng cho phần tường phía trên."
                }
            ]
        }
    ]
    return stage_defaults[(vo_index - 1) % len(stage_defaults)]


def detect_visual_anchor_in_slot(video_paths: List[str], start_sec: float, end_sec: float, target_wps: float = 2.5) -> Dict[str, Any]:
    slot_dur = max(4.0, end_sec - start_sec)
    min_x, max_x = 2, max(2, min(8, int(round(slot_dur * 0.55))))
    default_x = max(2, min(max_x, int(round(slot_dur * 0.35))))
    _, seg_metas = get_video_timeline_info(video_paths)
    if not seg_metas:
        return {"anchor_sec_in_clip": default_x, "anchor_global_sec": round(start_sec + default_x, 3), "anchor_word_index": max(4, int(round(default_x * target_wps))), "motion_peak": 5.0}
    chosen_seg = next((sm for sm in seg_metas if sm["start_global"] <= start_sec <= sm["end_global"]), seg_metas[-1])
    local_start = max(0.0, start_sec - chosen_seg["start_global"])
    best_x, max_diff, prev_gray = default_x, -1.0, None
    try:
        cap = cv2.VideoCapture(chosen_seg["video_path"])
        for sec_offset in range(min_x, max_x + 1):
            cap.set(cv2.CAP_PROP_POS_MSEC, min(chosen_seg["duration"] - 0.05, local_start + float(sec_offset)) * 1000.0)
            ret, fr = cap.read()
            if not ret or fr is None:
                continue
            gray = cv2.cvtColor(cv2.resize(fr, (160, 90)), cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
                if diff > max_diff:
                    max_diff, best_x = diff, sec_offset
            prev_gray = gray
        cap.release()
    except Exception:
        pass
    return {"anchor_sec_in_clip": int(best_x), "anchor_global_sec": round(start_sec + best_x, 3), "anchor_word_index": max(4, int(round(best_x * target_wps))), "motion_peak": round(max(1.0, max_diff), 2)}


def build_dynamic_video_vo_schedule(
    video_paths: List[str],
    vo_dur_mode: int = 0,
    min_gap_sec: float = 45.0,
    max_gap_sec: float = 75.0
) -> Tuple[float, List[Tuple[int, float, float, int, int]]]:
    total_vid_dur, seg_metas = get_video_timeline_info(video_paths)
    if not seg_metas or total_vid_dur <= 5.0:
        return total_vid_dur, [(1, 0.0, min(8.0, total_vid_dur), int(min(8.0, total_vid_dur)), 0)]

    if total_vid_dur < 180.0:
        eff_min_gap, eff_max_gap = max(5, int(round(total_vid_dur * 0.08))), max(10, int(round(total_vid_dur * 0.16)))
    elif total_vid_dur < 480.0:
        eff_min_gap, eff_max_gap = min(int(min_gap_sec), 18), min(int(max_gap_sec), 32)
    else:
        eff_min_gap, eff_max_gap = int(min(min_gap_sec, max_gap_sec)), int(max(min_gap_sec, max_gap_sec))

    cur_t = 2.0 if total_vid_dur < 60.0 else 5.0
    dynamic_duration_sequence = [20, 15, 20, 12, 15, 20, 10, 15, 20, 12, 8, 20]
    planned_slots: List[Tuple[int, float, float, int, int]] = []
    vo_idx = 1

    while cur_t + 8.0 <= total_vid_dur and vo_idx <= 2000:
        slot_d_int = int(vo_dur_mode) if int(vo_dur_mode) > 0 else dynamic_duration_sequence[(vo_idx - 1) % len(dynamic_duration_sequence)]
        if cur_t + slot_d_int > total_vid_dur - 1.0:
            rem_d = int(total_vid_dur - cur_t)
            slot_d_int = min([8, 10, 12, 15, 20], key=lambda x: abs(x - rem_d))
            if cur_t + slot_d_int > total_vid_dur:
                slot_d_int = max(8, rem_d)

        s_t = float(int(round(cur_t)))
        e_t = float(int(round(min(total_vid_dur, s_t + slot_d_int))))
        actual_slot_d = max(5, int(round(e_t - s_t)))

        earliest_next, latest_next = e_t + eff_min_gap, min(total_vid_dur - 8.0, e_t + eff_max_gap)
        if latest_next <= earliest_next:
            planned_slots.append((vo_idx, s_t, e_t, actual_slot_d, max(0, int(round(total_vid_dur - e_t)))))
            break

        best_next_int = int(round(earliest_next + ((vo_idx * 7) % max(1, int(latest_next - earliest_next)))))
        gap_chosen = int(np.clip(best_next_int - int(e_t), eff_min_gap, eff_max_gap))
        planned_slots.append((vo_idx, s_t, e_t, actual_slot_d, gap_chosen))
        cur_t = float(int(e_t) + gap_chosen)
        vo_idx += 1

    return total_vid_dur, planned_slots


# Từ điển Mệnh đề Chuẩn Ngữ Pháp theo đúng số từ thiếu (1 -> 15 từ)
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


def build_precise_action_synced_sentence(
    raw_en_text: str,
    target_words: int,
    anchor_word_pos: int,
    anchor_verb_hint: str,
    scene_idx: int,
    strip_image_path: str = "",
    detected_tool_en: str = "a cordless drill",
    detected_material_en: str = "wooden framing boards",
    variant_idx: int = 0,
    regen_round: int = 0
) -> Tuple[str, int, str]:
    tags = re.findall(r'\[[^\]]+\]', str(raw_en_text))
    spoken = _clean_srt_english_line(re.sub(r'\[[^\]]+\]', '', str(raw_en_text)))

    # Nếu câu bị Chrome dịch sang tiếng Việt hoặc dính mẫu lỗi cũ -> Xóa để tái tạo câu tiếng Anh chuẩn
    bad_markers = [
        "primary materials", "specialized tools", "the operator", "steadily carefully",
        "carefully steadily", "right on the main deck today", "so the wooden boards stay locked",
        "throughout every changing season", "guided by taut mason lines", "at timestamp"
    ]
    if _has_vn_chars(spoken) or any(bm in spoken.lower() for bm in bad_markers):
        spoken = ""

    spoken = spoken.replace("-", " ")
    spoken = re.sub(r'\s+', ' ', spoken).strip()
    tokens = [t for t in spoken.split() if t.strip()]

    if len(tokens) < 12:
        parsed_vis = parse_vietnamese_command_and_visuals("", strip_image_path, scene_idx, 0.0)
        opts = parsed_vis["options"]
        chosen_opt = opts[variant_idx % len(opts)]
        spoken = chosen_opt["en"].replace("-", " ")
        tokens = [t for t in spoken.split() if t.strip()]
        anchor_verb_hint = chosen_opt["anchor_verb"]

    tokens = [t.capitalize() if i == 0 else (t if t in ("I",) else t.lower()) for i, t in enumerate(tokens)]
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
            chosen_ext = ext_choices[(scene_idx + variant_idx + regen_round) % len(ext_choices)]
            ext_tokens = chosen_ext.split()
            if diff > 15:
                ext_tokens[-1] = ext_tokens[-1].rstrip(".!?") + ","
            tokens.extend(ext_tokens)
            cur_cnt = len(count_clean_english_words(" ".join(tokens)))
            diff = target_words - cur_cnt

    while len(count_clean_english_words(" ".join(tokens))) > target_words and len(tokens) > 8:
        tokens.pop(-1)

    dangling = {
        "and", "or", "with", "to", "from", "in", "on", "at", "by", "for", "of",
        "the", "a", "an", "that", "which", "so", "while", "before", "using",
        "keep", "every", "each", "into", "across", "under", "over", "between", "as", "its", "their"
    }
    if len(tokens) > 6 and re.sub(r'[^\w]', '', tokens[-1]).lower() in dangling:
        tokens[-1] = "securely."

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
        m_p = re.search(r'([,.;:?!]+)$', tokens[best_verb_idx])
        suf = m_p.group(1) if m_p else ""
        verb_clean = re.sub(r'[^\w]', '', tokens[best_verb_idx]).upper()
        tokens[best_verb_idx] = verb_clean + suf
    else:
        verb_clean = re.sub(r'[^\w]', '', str(anchor_verb_hint or "PLACES")).upper()

    if not tokens[-1].endswith((".", "!", "?")):
        tokens[-1] = tokens[-1].rstrip(",;:-") + "."

    final_spoken = " ".join(tokens)
    clean_tags = ["[calm, documentary tone]" if "giọng" in tg.lower() else tg for tg in tags]
    final_full = f"{' '.join(clean_tags)} {final_spoken}".strip() if clean_tags else final_spoken
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
    total_video_dur_str: str,
    strip_image_path: str,
    auto_v4_tags: bool = False,
    prev_en_scripts: Optional[List[str]] = None,
    user_custom_hint: str = "",
    current_en_script: str = "",
    regen_round: int = 0,
    rejected_en_list: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Phân tích Lệnh Tiếng Việt trực tiếp của người dùng (`user_custom_hint`) + Lưới 6 khung hình
    để tạo 3 Bản Nháp Kịch Bản khác biệt nhau 100%, tuyệt đối không bị nhảy sang giai đoạn sai!
    """
    start_mmss = format_timestamp_mmss(start_sec)
    end_mmss = format_timestamp_mmss(end_sec)
    safe_anchor_pos = int(np.clip(anchor_word_pos, 4, max(4, min(12, target_words - 6))))

    # Bước 1: Phân tích ngay câu lệnh Tiếng Việt + hình ảnh 6 khung hình bằng bộ giải mã nội bộ
    local_parsed = parse_vietnamese_command_and_visuals(user_custom_hint, strip_image_path, vo_index, start_sec)

    recent_history = "\n".join(f"- {s}" for s in (prev_en_scripts or [])[-4:]) if prev_en_scripts else "Chưa có."
    rejected_block = "\n".join(f"- {s}" for s in (rejected_en_list or [])[-6:]) if rejected_en_list else ""
    cmd_instruction = (
        f"\n- LỆNH CHỈ ĐẠO TRỰC TIẾP BẰNG TIẾNG VIỆT TỪ NGƯỜI DÙNG: \"{user_custom_hint}\"\n"
        f"  -> BẮT BUỘC 100% cả 3 phương án kịch bản Tiếng Anh và Tiếng Việt phải miêu tả chính xác hành động, con người và dụng cụ trong câu lệnh này!"
        if user_custom_hint.strip() else ""
    )
    reject_instruction = (
        f"\n- Không lặp lại các câu cũ sau:\n{rejected_block}"
        if rejected_block else ""
    )

    master_prompt = f"""Bạn là Đạo diễn Kiểm duyệt Hình ảnh & Biên kịch Phim tài liệu thực tế.
Hãy quan sát THẬT KỸ bức ảnh LƯỚI 6 KHUNG HÌNH LIÊN TIẾP (Hinh #1 -> Hinh #6) của phân đoạn VO #{vo_index}/{total_vos} tại mốc [{start_mmss} - {end_mmss}] (thời lượng {slot_dur} giây).{cmd_instruction}{reject_instruction}

QUY TẮC BẮT BUỘC:
1. BÁM SÁT 100% HÀNH ĐỘNG THẬT TRONG ẢNH & LỆNH TIẾNG VIỆT CỦA NGƯỜI DÙNG:
   - Nhìn kỹ từ Hình #1 đến Hình #6: Có mấy người? Đang cầm dụng cụ gì? Đang thao tác với vật liệu gì?
   - Tuyệt đối KHÔNG bịa ra việc làm khung cửa sổ hay lợp mái nếu trong ảnh đang khiêng tấm xốp hoặc làm sàn!
2. VIẾT 3 BẢN NHÁP KỊCH BẢN (3 OPTIONS) KHÁC NHAU ĐỂ NGƯỜI DÙNG XEM TRƯỚC VÀ CHỌN:
   - Bản nháp 1: Miêu tả trực diện hành động, con người và dụng cụ từ Hình #1 đến Hình #6.
   - Bản nháp 2: Nhấn mạnh vào cách lắp ghép kỹ thuật, độ khít và sự chắc chắn.
   - Bản nháp 3: Lối kể chuyện Documentary mạch lạc, tự nhiên.
   - Mỗi câu tiếng Anh ("en") BẮT BUỘC có ĐÚNG {target_words} từ ({slot_dur}s x 2.5), viết bằng tiếng Anh chuẩn 100%, có 1 ĐỘNG TỪ HÀNH ĐỘNG VIẾT IN HOA ở khoảng từ thứ {safe_anchor_pos}.
   - Đi kèm bản dịch Tiếng Việt ("vi") chuẩn xác, tự nhiên.

Trả về DUY NHẤT 1 đối tượng JSON hợp lệ:
{{
  "detected_tools_vi": "Tên cụ thể dụng cụ trong ảnh",
  "detected_materials_vi": "Tên cụ thể vật liệu trong ảnh",
  "detected_tool_en": "specific English tool name",
  "detected_material_en": "specific English material name",
  "micro_actions_vi": "Diễn biến cử chỉ cụ thể",
  "options": [
    {{
      "style_label": "Bản nháp 1: Trực diện Hành động & Dụng cụ",
      "anchor_verb": "CARRIES",
      "en": "Câu tiếng Anh bản nháp 1 dài ĐÚNG {target_words} từ",
      "vi": "Phụ đề tiếng Việt bản nháp 1 chuẩn xác"
    }},
    {{
      "style_label": "Bản nháp 2: Chi tiết Kỹ thuật & Độ khít",
      "anchor_verb": "PLACES",
      "en": "Câu tiếng Anh bản nháp 2 dài ĐÚNG {target_words} từ",
      "vi": "Phụ đề tiếng Việt bản nháp 2 chuẩn xác"
    }},
    {{
      "style_label": "Bản nháp 3: Phong cách Kể chuyện Documentary",
      "anchor_verb": "FITS",
      "en": "Câu tiếng Anh bản nháp 3 dài ĐÚNG {target_words} từ",
      "vi": "Phụ đề tiếng Việt bản nháp 3 chuẩn xác"
    }}
  ]
}}"""

    detected_tools_vi = local_parsed["tools_vi"]
    detected_materials_vi = local_parsed["mats_vi"]
    detected_tool_en = local_parsed["tool_en"]
    detected_material_en = local_parsed["mat_en"]
    micro_actions_vi = local_parsed["action_vi"]
    raw_options_list: List[Dict[str, Any]] = []

    raw_resp = _call_gemini_vision_rest_and_sdk(gemini_key, master_prompt, strip_image_path, json_mode=True)
    if raw_resp:
        try:
            m_json = re.search(r'\{.*\}', raw_resp, re.DOTALL)
            if m_json:
                parsed = json.loads(m_json.group(0), strict=False)
                if parsed.get("detected_tools_vi"):
                    detected_tools_vi = str(parsed["detected_tools_vi"]).strip()
                if parsed.get("detected_materials_vi"):
                    detected_materials_vi = str(parsed["detected_materials_vi"]).strip()
                if parsed.get("detected_tool_en"):
                    detected_tool_en = str(parsed["detected_tool_en"]).strip()
                if parsed.get("detected_material_en"):
                    detected_material_en = str(parsed["detected_material_en"]).strip()
                if parsed.get("micro_actions_vi"):
                    micro_actions_vi = str(parsed["micro_actions_vi"]).strip()
                if isinstance(parsed.get("options"), list) and len(parsed["options"]) >= 1:
                    raw_options_list = parsed["options"]
        except Exception:
            pass

    fallback_opts = local_parsed["options"]
    final_candidates: List[Dict[str, Any]] = []
    seen_in_vo = set()

    for c_idx in range(3):
        fb_item = fallback_opts[c_idx % len(fallback_opts)]
        raw_opt = raw_options_list[c_idx] if c_idx < len(raw_options_list) and isinstance(raw_options_list[c_idx], dict) else {}

        lbl = str(raw_opt.get("style_label") or fb_item["style_label"]).strip()
        v_hint = str(raw_opt.get("anchor_verb") or fb_item["anchor_verb"]).strip().upper()
        en_raw = str(raw_opt.get("en") or fb_item["en"]).strip()
        vi_raw = str(raw_opt.get("vi") or fb_item["vi"]).strip()

        # Đảm bảo 3 phương án trong cùng 1 VO không bao giờ bị trùng câu nhau
        if re.sub(r'\[[^\]]+\]', '', en_raw).strip().lower() in seen_in_vo:
            en_raw = fb_item["en"]
            vi_raw = fb_item["vi"]
            v_hint = fb_item["anchor_verb"]

        en_fixed, act_w, v_final = build_precise_action_synced_sentence(
            en_raw, target_words, safe_anchor_pos, v_hint, vo_index, strip_image_path,
            detected_tool_en=detected_tool_en, detected_material_en=detected_material_en,
            variant_idx=c_idx, regen_round=regen_round
        )
        seen_in_vo.add(re.sub(r'\[[^\]]+\]', '', en_fixed).strip().lower())

        if auto_v4_tags and not re.findall(r'\[[^\]]+\]', en_fixed):
            en_fixed = f"[calm, documentary tone] {en_fixed}"

        vi_checked = translate_to_vietnamese_subtitle(gemini_key, en_fixed, vi_raw, force_refresh=(not _has_vn_chars(vi_raw)))

        final_candidates.append({
            "option_id": c_idx + 1,
            "style_label": lbl,
            "anchor_verb": v_final,
            "actual_words": act_w,
            "target_words": target_words,
            "en_text": en_fixed,
            "vi_text": vi_checked
        })

    primary = final_candidates[0]
    audit_badge = f"🛠️ Dụng cụ: {detected_tools_vi} | 🧱 Vật liệu: {detected_materials_vi} | 🎬 Hành động: {micro_actions_vi}"
    srt_start = format_timestamp_srt(start_sec)
    srt_end = format_timestamp_srt(end_sec)
    formatted_block = f"{vo_index}\n{srt_start} --> {srt_end}\n{primary['en_text']}\n"

    return {
        "visual_action_vi": micro_actions_vi,
        "audit_summary": audit_badge,
        "anchor_sec_x": anchor_sec_x,
        "anchor_word_pos": safe_anchor_pos,
        "anchor_verb": primary["anchor_verb"],
        "target_words": target_words,
        "actual_words": primary["actual_words"],
        "en_text": primary["en_text"],
        "vi_text": primary["vi_text"],
        "candidates": final_candidates,
        "next_gap_sec": next_gap_sec,
        "formatted_block": formatted_block
    }


def generate_cinematic_cabin_vo_item_with_ai(
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
    total_video_dur_str: str,
    strip_image_path: str,
    auto_v4_tags: bool = False,
    prev_en_scripts: Optional[List[str]] = None,
    user_custom_hint: str = "",
    current_en_script: str = "",
    regen_round: int = 0,
    rejected_en_list: Optional[List[str]] = None
) -> Dict[str, Any]:
    return generate_multiple_vo_candidates_with_ai(
        gemini_key=gemini_key, vo_index=vo_index, total_vos=total_vos,
        start_sec=start_sec, end_sec=end_sec, slot_dur=slot_dur, target_words=target_words,
        anchor_sec_x=anchor_sec_x, anchor_word_pos=anchor_word_pos, next_gap_sec=next_gap_sec,
        total_video_dur_str=total_video_dur_str, strip_image_path=strip_image_path,
        auto_v4_tags=auto_v4_tags, prev_en_scripts=prev_en_scripts,
        user_custom_hint=user_custom_hint, current_en_script=current_en_script,
        regen_round=regen_round, rejected_en_list=rejected_en_list
    )


def run_self_diagnostic_and_auto_repair(rows: List[Dict[str, Any]], gemini_key: str, target_wps: float = 2.50) -> List[Dict[str, Any]]:
    """
    Tự động khôi phục ngay nếu câu tiếng Anh bị Chrome dịch sang tiếng Việt ('[Giọng điệu bình tĩnh...] Làm việc xung quanh...')
    hoặc nếu 2 phương án trong cùng 1 VO bị trùng nhau!
    """
    if not rows:
        return rows

    for idx, r in enumerate(rows):
        s_sec = float(int(round(parse_timestamp_srt(str(r.get("Mốc Bắt Đầu", "00:00:05,000"))))))
        e_sec = float(int(round(parse_timestamp_srt(str(r.get("Mốc Kết Thúc", "00:00:25,000"))))))
        if e_sec <= s_sec:
            e_sec = s_sec + 15.0
        slot_d = int(round(e_sec - s_sec))
        r["Mốc Bắt Đầu"] = format_timestamp_srt(s_sec)
        r["Mốc Kết Thúc"] = format_timestamp_srt(e_sec)
        r["Khung Giờ (s)"] = float(slot_d)

        cur_en = _clean_srt_english_line(r.get("Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)", ""))
        cur_vi = str(r.get("Phụ Đề Tiếng Việt (Dễ Hiểu)", "")).strip()
        target_w = MANDATORY_DURATION_WORD_MAP.get(slot_d, int(round(slot_d * target_wps)))
        anchor_x = int(r.get("anchor_sec_x", max(2, min(8, int(round(slot_d * 0.35))))))
        safe_pos = int(np.clip(int(round(anchor_x * target_wps)), 4, max(4, min(12, target_w - 6))))
        user_cmd = str(r.get("user_vi_cmd", "")).strip()

        local_vis = parse_vietnamese_command_and_visuals(user_cmd, r.get("strip_path", ""), idx + 1, s_sec)
        if not r.get("Kiểm Tra Dụng Cụ & Hành Động") or "dụng cụ thực tế" in str(r.get("Kiểm Tra Dụng Cụ & Hành Động", "")).lower():
            r["Kiểm Tra Dụng Cụ & Hành Động"] = f"🛠️ Dụng cụ: {local_vis['tools_vi']} | 🧱 Vật liệu: {local_vis['mats_vi']} | 🎬 Hành động: {local_vis['action_vi']}"

        spoken_only = re.sub(r'\[[^\]]+\]', '', cur_en).strip()
        cur_w_cnt = len(count_clean_english_words(spoken_only))

        # Nếu câu tiếng Anh bị Chrome dịch sang tiếng Việt (_has_vn_chars) hoặc sai số từ -> Khôi phục câu tiếng Anh chuẩn
        if _has_vn_chars(spoken_only) or cur_w_cnt != target_w or "primary materials" in spoken_only.lower():
            has_tag = bool(re.findall(r'\[[^\]]+\]', cur_en))
            base_en = local_vis["options"][0]["en"] if (_has_vn_chars(spoken_only) or "primary materials" in spoken_only.lower()) else spoken_only
            fixed_en, act_w, final_v = build_precise_action_synced_sentence(
                base_en, target_w, safe_pos, local_vis["options"][0]["anchor_verb"], idx + 1, r.get("strip_path", ""), variant_idx=0
            )
            if has_tag and not re.findall(r'\[[^\]]+\]', fixed_en):
                fixed_en = f"[calm, documentary tone] {fixed_en}"
            cur_en = fixed_en
            r["Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)"] = cur_en
            r["Điểm Chạm & Từ Neo"] = f"Giây {anchor_x}s -> ({final_v})"
            cur_vi = local_vis["options"][0]["vi"]

        r["Phụ Đề Tiếng Việt (Dễ Hiểu)"] = translate_to_vietnamese_subtitle(gemini_key, cur_en, cur_vi)

        # Kiểm tra 3 phương án: Đảm bảo không có phương án nào bị Chrome dịch sang tiếng Việt ở dòng EN hoặc trùng nhau
        cands = r.get("candidates")
        need_fix_cands = (not isinstance(cands, list) or len(cands) < 3)
        if not need_fix_cands and isinstance(cands, list):
            en_set_chk = set()
            for c_it in cands:
                if not isinstance(c_it, dict):
                    need_fix_cands = True
                    break
                c_sp = re.sub(r'\[[^\]]+\]', '', str(c_it.get("en_text", ""))).strip()
                if _has_vn_chars(c_sp) or c_sp.lower() in en_set_chk or len(count_clean_english_words(c_sp)) != target_w:
                    need_fix_cands = True
                    break
                en_set_chk.add(c_sp.lower())

        if need_fix_cands:
            new_cands = []
            for v_i in range(3):
                opt_src = local_vis["options"][v_i % len(local_vis["options"])]
                c_en, c_w, c_v = build_precise_action_synced_sentence(
                    opt_src["en"], target_w, safe_pos, opt_src["anchor_verb"], idx + 1, r.get("strip_path", ""), variant_idx=v_i
                )
                if re.findall(r'\[[^\]]+\]', cur_en) and not re.findall(r'\[[^\]]+\]', c_en):
                    c_en = f"[calm, documentary tone] {c_en}"
                c_vi = translate_to_vietnamese_subtitle(gemini_key, c_en, opt_src["vi"])
                new_cands.append({
                    "option_id": v_i + 1,
                    "style_label": opt_src["style_label"],
                    "anchor_verb": c_v,
                    "actual_words": c_w,
                    "target_words": target_w,
                    "en_text": c_en,
                    "vi_text": c_vi
                })
            r["candidates"] = new_cands

        m_sync = compute_script_timing_metrics(cur_en, float(slot_d), 1.0, target_wps, 4, 120, 100, 0.0, 0)
        r.update({
            "Số Từ": m_sync["word_count"],
            "Tối Đa Từ Chuẩn": m_sync["ideal_max_words"],
            "Thời Gian Cảm Xúc (s)": m_sync["emotion_sec"],
            "Tốc Độ Từ (ms/từ)": m_sync["ms_per_word"],
            "Voice Ước Tính (s)": m_sync["est_sec"],
            "Trạng Thái Khớp Nhịp": m_sync["status"]
        })
    return rows


def _build_preview_clip_filepath(
    video_path: str,
    local_start: float,
    slot_dur: float,
    voice_audio_path: str,
    row_id: int
) -> str:
    """Tạo đường dẫn file video xem thử thống nhất 100% giữa hàm tạo và hàm kiểm tra."""
    has_voice = bool(voice_audio_path and os.path.exists(str(voice_audio_path)))
    v_tag = hashlib.md5(
        f"pv_v3_{video_path}_{local_start:.2f}_{slot_dur:.2f}_{voice_audio_path if has_voice else 'novoice'}".encode("utf-8")
    ).hexdigest()[:12]
    return os.path.join(CLIP_PREVIEW_DIR, f"fast_pv_row_{row_id:04d}_{v_tag}.mp4")


def get_existing_preview_clip_path(
    video_paths: List[str],
    start_sec: float,
    end_sec: float,
    voice_audio_path: str,
    row_id: int
) -> str:
    if not video_paths:
        return ""
    _, seg_metas = get_video_timeline_info(video_paths)
    if not seg_metas:
        return ""
    chosen_seg = next((sm for sm in seg_metas if sm["start_global"] <= start_sec <= sm["end_global"]), seg_metas[0])
    local_start = max(0.0, start_sec - chosen_seg["start_global"])
    slot_dur = max(1.0, end_sec - start_sec)
    out_clip = _build_preview_clip_filepath(chosen_seg["video_path"], local_start, slot_dur, voice_audio_path, row_id)
    return out_clip if (os.path.exists(out_clip) and os.path.getsize(out_clip) > 2048) else ""


def create_synced_preview_video_clip(
    video_paths: List[str],
    start_sec: float,
    end_sec: float,
    voice_audio_path: str,
    row_id: int
) -> str:
    """
    Cắt video xem thử chuẩn H.264 yuv420p + faststart và trộn mượt tiếng ASMR gốc [0:a] + tiếng Voice [1:a]
    với bộ đồng bộ mốc thời gian asetpts=PTS-STARTPTS.
    """
    if not video_paths:
        return ""
    _, seg_metas = get_video_timeline_info(video_paths)
    if not seg_metas:
        return ""

    chosen_seg = next((sm for sm in seg_metas if sm["start_global"] <= start_sec <= sm["end_global"]), seg_metas[0])
    local_start = max(0.0, start_sec - chosen_seg["start_global"])
    slot_dur = max(1.0, end_sec - start_sec)
    has_voice = bool(voice_audio_path and os.path.exists(str(voice_audio_path)))
    out_clip = _build_preview_clip_filepath(chosen_seg["video_path"], local_start, slot_dur, voice_audio_path, row_id)

    if os.path.exists(out_clip) and os.path.getsize(out_clip) > 2048:
        return out_clip

    try:
        if has_voice:
            # Đưa cả âm thanh gốc [0:a] và Voice [1:a] về cùng tần số 44100Hz Stereo và mốc 0s (PTS-STARTPTS)
            filter_str = (
                "[0:a]aresample=44100,aformat=sample_rates=44100:channel_layouts=stereo,asetpts=PTS-STARTPTS,volume=0.75[a0];"
                "[1:a]aresample=44100,aformat=sample_rates=44100:channel_layouts=stereo,asetpts=PTS-STARTPTS,volume=1.0[a1];"
                "[a0][a1]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
            )
            cmd_mix = [
                "ffmpeg", "-y", "-ss", f"{local_start:.3f}", "-t", f"{slot_dur:.3f}",
                "-i", chosen_seg["video_path"], "-i", str(voice_audio_path),
                "-filter_complex", filter_str,
                "-map", "0:v:0", "-map", "[aout]",
                "-vf", "scale=640:360:flags=fast_bilinear",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-c:a", "aac", "-b:a", "192k", "-shortest", out_clip
            ]
            subprocess.run(cmd_mix, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)

            # Dự phòng nếu video gốc không có kênh âm thanh [0:a]
            if not os.path.exists(out_clip) or os.path.getsize(out_clip) <= 2048:
                cmd_vo = [
                    "ffmpeg", "-y", "-ss", f"{local_start:.3f}", "-t", f"{slot_dur:.3f}",
                    "-i", chosen_seg["video_path"], "-i", str(voice_audio_path),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-vf", "scale=640:360:flags=fast_bilinear",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", out_clip
                ]
                subprocess.run(cmd_vo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
        else:
            cmd_raw = [
                "ffmpeg", "-y", "-ss", f"{local_start:.3f}", "-t", f"{slot_dur:.3f}",
                "-i", chosen_seg["video_path"],
                "-vf", "scale=640:360:flags=fast_bilinear",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-c:a", "aac", "-b:a", "192k", out_clip
            ]
            subprocess.run(cmd_raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)

        if os.path.exists(out_clip) and os.path.getsize(out_clip) > 2048:
            return out_clip
    except Exception:
        pass
    return ""


def import_zip_voice_pack_to_rows(zip_file_obj, existing_rows: List[Dict[str, Any]], video_paths: Any = "", gemini_key: str = "") -> List[Dict[str, Any]]:
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
            updated_rows[idx]["audio_path"] = fpath
            updated_rows[idx]["Voice Thực Tế (s)"] = r_dur
            slot_d = float(updated_rows[idx].get("Khung Giờ (s)", r_dur))
            updated_rows[idx]["Trạng Thái Khớp Nhịp"] = f"✅ Đã nạp từ ZIP ({r_dur}s / {slot_d}s)"
        else:
            m_tc = re.search(r'(\d{2}-\d{2}-\d{2}(?:-\d{3})?)', fname)
            s_sec = parse_timestamp_srt(m_tc.group(1)) if m_tc else curr_t
            e_sec = round(s_sec + r_dur, 3)
            curr_t = e_sec + 1.0
            th_p, st_p = extract_slot_filmstrip_from_video(video_paths, s_sec, e_sec, idx + 1) if video_paths else ("", "")
            updated_rows.append({
                "STT": idx + 1, "Mốc Bắt Đầu": format_timestamp_srt(s_sec), "Mốc Kết Thúc": format_timestamp_srt(e_sec),
                "Khung Giờ (s)": round(r_dur, 2), "Điểm Chạm & Từ Neo": "Auto Sync",
                "Kiểm Tra Dụng Cụ & Hành Động": "Nạp từ gói ZIP",
                "Số Từ": 0, "Tối Đa Từ Chuẩn": 0, "Thời Gian Cảm Xúc (s)": 0.0,
                "Tốc Độ Từ (ms/từ)": 0, "Voice Ước Tính (s)": r_dur, "Voice Thực Tế (s)": r_dur,
                "Trạng Thái Khớp Nhịp": f"✅ Đã nạp từ ZIP ({fname})",
                "Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)": f"[Audio từ ZIP] {fname}",
                "Phụ Đề Tiếng Việt (Dễ Hiểu)": f"File âm thanh nạp từ gói ZIP: {fname}",
                "audio_path": fpath, "frame_path": th_p, "strip_path": st_p
            })
    return updated_rows


# ==============================================================================
# 5. GIAO DIỆN TRẠM ĐẠO DIỄN VOICE ELEVENLABS PRO V2.1 (CÓ BẢN NHÁP & TIẾT KIỆM CREDIT)
# ==============================================================================

def render_elevenlabs_dubbing_station_ui(
    saved_cfg: dict,
    gemini_api_key: str,
    elevenlabs_api_key: str,
    whisper_model_size: str,
    auto_confirm_3min: bool,
    auto_confirm_seconds: int,
    uploaded_videos: Any,
    uploaded_audios: Any,
    resolve_or_cache_fn: Callable,
    flatten_files_fn: Callable
) -> Dict[str, Any]:
    def _get_all_loaded_video_paths() -> List[str]:
        act_vids = [p for p in st.session_state.get("direct_video_paths", []) if os.path.exists(p)]
        if not act_vids and uploaded_videos:
            for vf in flatten_files_fn(uploaded_videos):
                pv, _ = resolve_or_cache_fn(vf)
                if pv and os.path.exists(pv):
                    act_vids.append(pv)
        return act_vids

    current_vids = _get_all_loaded_video_paths()
    current_vid_sig = compute_video_fingerprint_signature(current_vids)
    prev_vid_sig = st.session_state.get("active_video_fingerprint_sig", "")
    if current_vid_sig != "no_video" and prev_vid_sig and current_vid_sig != prev_vid_sig:
        st.session_state["eleven_script_rows"] = []
        st.session_state["eleven_raw_text_buffer"] = ""
        st.session_state["eleven_step_a_confirmed"] = False
        st.session_state["eleven_step_b_ready"] = False
        st.toast("🔄 Đã nhận diện Video mới! Hệ thống đã làm sạch kịch bản cũ để phân tích riêng cho video mới.")
    if current_vid_sig != "no_video":
        st.session_state["active_video_fingerprint_sig"] = current_vid_sig

    el_tab_native, el_tab_clone, el_tab_html = st.tabs([
        "🎛️ Bàn Điều Khiển ElevenLabs v4, Lệnh Sửa Trực Tiếp Tiếng Việt (Xem Bản Nháp) & Tiết Kiệm Credit",
        "🧬 Trạm Clone Giọng Nhanh (Instant Voice Cloning)",
        "🌐 Giao Diện Web Eleven Labstool_6.html Gốc Của Bạn (Đồng Bộ 2 Chiều)"
    ])

    with el_tab_native:
        with st.expander("⚙️ CÀI ĐẶT GIỌNG ĐỌC ADAM, HỆ SỐ 2.5 TỪ/GIÂY & CHẾ ĐỘ SIÊU TIẾT KIỆM CREDIT ELEVENLABS PRO", expanded=False):
            ev_c1, ev_c2, ev_c3 = st.columns([1.05, 1.15, 1.05])
            all_el_voices = st.session_state.get("eleven_voices_list", DEFAULT_ELEVEN_VOICES)

            with ev_c1:
                st.markdown("##### ⚙️ 1. Chọn Model & Bộ Lọc Giọng Đọc Pro")
                model_options = {
                    "eleven_v4": "🔥 Eleven v4 (Mới nhất - Nhanh & Cảm xúc nhất, hỗ trợ Audio Tags)",
                    "eleven_v4_turbo": "⚡ Eleven v4 Turbo (Tốc độ cao, giữ nguyên cảm xúc v4)",
                    "eleven_multilingual_v2": "💎 Eleven Multilingual v2 (Ổn định, chuẩn Phim tài liệu dài)",
                    "eleven_v3": "🎭 Eleven v3 (Biểu cảm cao & Hỗ trợ thẻ cảm xúc)",
                    "eleven_turbo_v2_5": "🚀 Eleven Turbo v2.5 (Nhanh, mượt & Tiết kiệm 50% Credit)",
                    "eleven_flash_v2_5": "💨 Eleven Flash v2.5 (Siêu tốc, tiết kiệm 50% Credit)"
                }
                m_keys = list(model_options.keys())
                saved_em = saved_cfg.get("eleven_model_id", "eleven_v4")
                eleven_model_id = st.selectbox("Mô hình AI (Model):", options=m_keys, format_func=lambda k: model_options[k], index=m_keys.index(saved_em) if saved_em in m_keys else 0)

                fc_g, fc_c = st.columns(2)
                with fc_g:
                    flt_gender = st.selectbox("Giới tính:", ["all", "male", "female"], format_func=lambda x: "Tất cả" if x == "all" else ("👨 Nam" if x == "male" else "👩 Nữ"))
                with fc_c:
                    flt_cat = st.selectbox("Nguồn giọng:", ["all", "cloned", "premade"], format_func=lambda x: "Tất cả" if x == "all" else ("🧬 Giọng Clone" if x == "cloned" else "🌟 Mặc định"))

                fc_e, fc_s = st.columns(2)
                with fc_e:
                    flt_emo = st.selectbox("Phong cách:", ["all", "narrative", "calm", "deep", "raspy", "conversational"])
                with fc_s:
                    flt_search = st.text_input("🔍 Tìm tên giọng:", value="", placeholder="Adam, Brian...")

                filtered_el_voices = [
                    v for v in all_el_voices
                    if (flt_gender == "all" or v.get("gender") == flt_gender)
                    and (flt_cat == "all" or (flt_cat == "cloned" and v.get("category") == "cloned") or (flt_cat == "premade" and v.get("category") != "cloned"))
                    and (flt_emo == "all" or flt_emo in f"{v.get('use_case','')} {v.get('description','')}".lower())
                    and (not flt_search.strip() or flt_search.strip().lower() in f"{v.get('name','')} {v.get('accent','')} {v.get('description','')}".lower())
                ] or all_el_voices

                v_ids = [v["voice_id"] for v in filtered_el_voices]
                v_lbls = {v["voice_id"]: f"{'🧬 [Clone]' if v.get('category') == 'cloned' else ('👩' if v.get('gender') == 'female' else '👨')} {v['name']} ({v.get('accent') or v.get('use_case') or 'Pro'})" for v in filtered_el_voices}
                saved_vid = saved_cfg.get("eleven_voice_id", v_ids[0])
                eleven_voice_id = st.selectbox(f"Giọng đọc đang chọn ({len(filtered_el_voices)} giọng):", options=v_ids, format_func=lambda k: v_lbls.get(k, k), index=v_ids.index(saved_vid) if saved_vid in v_ids else 0)

                chosen_v = next((v for v in filtered_el_voices if v["voice_id"] == eleven_voice_id), None)
                if chosen_v and chosen_v.get("preview_url"):
                    st.audio(chosen_v["preview_url"])

                eleven_speed = st.slider("⚡ Tốc độ nhả chữ AI (ElevenLabs Native Speed)", 0.70, 1.20, float(saved_cfg.get("eleven_speed", 1.00)), 0.02)
                st_c1, st_c2, st_c3 = st.columns(3)
                with st_c1:
                    eleven_stability = st.slider("Stability (%)", 0, 100, int(saved_cfg.get("eleven_stability", 45)), 5)
                with st_c2:
                    eleven_similarity = st.slider("Similarity (%)", 0, 100, int(saved_cfg.get("eleven_similarity", 80)), 5)
                with st_c3:
                    eleven_style = st.slider("Style (%)", 0, 100, int(saved_cfg.get("eleven_style", 15)), 5)

            with ev_c2:
                st.markdown("##### ⏱️ 2. Công Thức Toán Học 2.5 Từ/Giây (8s=20, 10s=25, 12s=30, 15s=38, 20s=50)")
                target_wps = st.slider(
                    "🗣️ Hệ số tốc độ từ bắt buộc (Từ / giây - WPS)",
                    min_value=1.00, max_value=4.50, value=float(saved_cfg.get("target_wps", 2.50)), step=0.05
                )
                eff_wps_now = target_wps * eleven_speed
                st.caption(f"👉 Quy đổi chuẩn: **`{int(round(1000.0 / max(0.5, eff_wps_now)))} ms / từ`** | **`8s=20 | 10s=25 | 12s=30 | 15s=38 | 20s=50 từ`**")

                w_lim_c1, w_lim_c2 = st.columns(2)
                with w_lim_c1:
                    min_words_limit = st.number_input("Số từ tối thiểu / câu:", min_value=1, max_value=60, value=int(saved_cfg.get("min_words_limit", 15)), step=1)
                with w_lim_c2:
                    max_words_limit = st.number_input("Số từ tối đa / câu:", min_value=5, max_value=250, value=int(saved_cfg.get("max_words_limit", 100)), step=1)

                slot_c1, slot_c2 = st.columns(2)
                with slot_c1:
                    slot_fill_pct = st.slider("Tỷ lệ lấp đầy khung giờ (%)", 50, 100, int(saved_cfg.get("slot_fill_pct", 100)), 1)
                with slot_c2:
                    breath_gap_sec = st.slider("Khoảng nghỉ lấy hơi cuối câu (s)", 0.0, 2.5, float(saved_cfg.get("breath_gap_sec", 0.0)), 0.05)

                p_c1, p_c2 = st.columns(2)
                with p_c1:
                    punct_pause_ms = st.slider("Độ nghỉ tại dấu câu (, . ; —) [ms]", 0, 600, int(saved_cfg.get("punct_pause_ms", 0)), 10)
                with p_c2:
                    voice_lead_ms = st.slider("Độ trễ / sớm khi vào giọng (ms)", -500, 2000, int(saved_cfg.get("voice_lead_ms", 0)), 25)

                align_map = {
                    "start": "⏮️ Vào giọng ngay đầu khung giờ (Chuẩn SRT & Action-Word Sync)",
                    "center": "⏺️ Căn câu Voice vào chính giữa khung giờ",
                    "end": "⏭️ Dồn câu Voice về sát cuối khung giờ"
                }
                al_keys = list(align_map.keys())
                saved_al = saved_cfg.get("slot_align_mode", "start")
                slot_align_mode = st.selectbox("Vị trí đặt câu Voice trong khung giờ:", options=al_keys, format_func=lambda k: align_map[k], index=al_keys.index(saved_al) if saved_al in al_keys else 0)

            with ev_c3:
                st.markdown("##### 🛡️ 3. Siêu Tiết Kiệm Credit ElevenLabs Pro")
                strict_credit_saver = st.checkbox(
                    "🛡️ KHÓA BẢO VỆ CREDIT TUYỆT ĐỐI (Không bao giờ tạo lại câu đã có Voice)",
                    value=True
                )
                strip_tags_credit_saver = st.checkbox(
                    "💎 Tự gỡ thẻ [calm, documentary tone] khi gửi API (Tiết kiệm ~450 Credit/Video)",
                    value=True,
                    help="Thẻ [calm, documentary tone] dài 24 ký tự ở đầu mỗi câu sẽ bị tính tiền nếu gửi lên ElevenLabs. Bật nút này để hệ thống tự gỡ thẻ trước khi gửi API (vẫn giữ nguyên giọng trầm ấm nhờ thanh Stability/Style)!"
                )
                auto_inject_v4_tags = st.checkbox(
                    "🎭 Hiển thị nhãn [calm, documentary tone] trên bảng kịch bản",
                    value=bool(saved_cfg.get("auto_inject_v4_tags", False))
                )

                overrun_map = {
                    "ai_rewrite_and_atempo": "🪄 Chế độ Kép (Khuyên dùng): AI nhìn ảnh video bớt từ + Co nhẹ atempo khớp 100%",
                    "ai_strict_rewrite": "✍️ Thuần AI Viết Lại 100%: Tự rút gọn từ cho đến khi vừa khít",
                    "atempo_only": "⏩ Thuần DSP Atempo (Tiết kiệm Credit nhất): Giữ 100% chữ gốc, chỉ co giãn âm thanh",
                    "shift_timeline": "↔️ Tự động lùi mốc kết thúc câu vào khoảng trống B-roll phía sau"
                }
                ov_keys = list(overrun_map.keys())
                saved_ov = saved_cfg.get("overrun_policy", "ai_rewrite_and_atempo")
                overrun_policy = st.selectbox("Khi Voice DÀI HƠN khung giờ kịch bản:", options=ov_keys, format_func=lambda k: overrun_map[k], index=ov_keys.index(saved_ov) if saved_ov in ov_keys else 0)

                underrun_map = {
                    "ai_expand_action": "✨ AI Tự Động Thêm Từ: Nhìn khung hình video bồi đắp thêm chi tiết hành động",
                    "keep_and_duck": "🎵 Giữ tự nhiên & Tự động nâng nhạc nền BGM / ASMR ở khoảng dư"
                }
                un_keys = list(underrun_map.keys())
                saved_un = saved_cfg.get("underrun_policy", "ai_expand_action")
                underrun_policy = st.selectbox("Khi Voice NGẮN HƠN khung giờ kịch bản:", options=un_keys, format_func=lambda k: underrun_map[k], index=un_keys.index(saved_un) if saved_un in un_keys else 0)

                max_atempo_limit = st.slider("Giới hạn co giãn tốc độ Voice tối đa (Max Atempo)", 1.05, 1.50, float(saved_cfg.get("max_atempo_limit", 1.35)), 0.05)
                auto_fit_voice_slot = (overrun_policy in ["ai_rewrite_and_atempo", "atempo_only"])

                rw_style_map = {
                    "documentary_action": "🎬 Thực Tế Rõ Ràng, Đúng Dụng Cụ & Khớp Từng Cử Chỉ (Khuyên dùng)",
                    "concise_technical": "🛠️ Kỹ Thuật Thi Công Súc Tích (Ngắn gọn, chuẩn xác)",
                    "emotional_cinematic": "🍃 Điện Ảnh Trầm Lắng & Cảm Xúc"
                }
                rw_keys = list(rw_style_map.keys())
                saved_rw = saved_cfg.get("ai_rewrite_style", "documentary_action")
                ai_rewrite_style = st.selectbox("Phong cách AI biên tập kịch bản:", options=rw_keys, format_func=lambda k: rw_style_map[k], index=rw_keys.index(saved_rw) if saved_rw in rw_keys else 0)

        # ======================================================================
        # BƯỚC A: TRẠM ĐẠO DIỄN AI QUÉT 6 KHUNG HÌNH / CẢNH & TỰ ĐỘNG KHỚP NHỊP 100%
        # ======================================================================
        with st.expander("🎬 BƯỚC A: AI TỰ ĐỘNG QUÉT 6 KHUNG HÌNH / CẢNH, KIỂM TRA DỤNG CỤ THỰC TẾ & TẠO KỊCH BẢN CHUẨN .SRT", expanded=True):
            in_c1, in_c2 = st.columns([0.65, 0.35])
            with in_c1:
                raw_script_input = st.text_area(
                    "Kịch bản Đạo diễn chuẩn .SRT (Chỉ gồm Thời gian + Tiếng Anh để ghép Voice chuẩn 100%):",
                    value=st.session_state.get("eleven_raw_text_buffer", ""),
                    height=165
                )
                st.session_state["eleven_raw_text_buffer"] = raw_script_input
            with in_c2:
                st.markdown("##### 🎯 Thiết Lập Đạo Diễn AI Linh Hoạt Theo Video")
                vo_dur_mode = st.selectbox(
                    "Chế độ độ dài mỗi câu VO:",
                    options=[0, 20, 15, 12, 10, 8],
                    format_func=lambda x: (
                        "🔄 Tự động linh hoạt theo cảnh quay thực tế (8s=20 | 10s=25 | 12s=30 | 15s=38 | 20s=50 từ)"
                        if x == 0 else f"⏱️ Cố định {x} giây = ĐÚNG {MANDATORY_DURATION_WORD_MAP.get(x, int(x*2.5))} từ tiếng Anh"
                    ),
                    index=0
                )
                gap_c1, gap_c2 = st.columns(2)
                with gap_c1:
                    asmr_min_gap = st.number_input("Nghỉ ASMR tối thiểu (s):", min_value=4, max_value=180, value=45, step=5)
                with gap_c2:
                    asmr_max_gap = st.number_input("Nghỉ ASMR tối đa (s):", min_value=8, max_value=240, value=75, step=5)

                _has_vids = len(_get_all_loaded_video_paths()) > 0
                _has_script_in = bool(raw_script_input.strip()) or bool(st.session_state.get("direct_audio_paths")) or bool(uploaded_audios)
                _has_rows = len(st.session_state.get("eleven_script_rows", [])) > 0
                _has_el_key = bool(clean_eleven_api_key(elevenlabs_api_key))
                _has_voices = any(r.get("audio_path") and os.path.exists(str(r.get("audio_path", ""))) for r in st.session_state.get("eleven_script_rows", []))

                auto_director_master_btn = st.button(
                    "🎬 AI QUÉT VIDEO NÀY & TẠO KỊCH BẢN MỚI" if _has_vids else "🔒 [HÃY CHỌN VIDEO Ở MỤC 1 TRƯỚC] AI QUÉT VIDEO",
                    type="primary",
                    use_container_width=True,
                    disabled=(not _has_vids)
                )

                uploaded_zip_pack = st.file_uploader("📦 Hoặc Nạp Gói .ZIP Voice:", type=["zip"])
                if uploaded_zip_pack is not None:
                    if st.button("📥 Giải Nén & Khớp Mốc Giây Từ .ZIP", use_container_width=True):
                        p_vids = _get_all_loaded_video_paths()
                        st.session_state["eleven_script_rows"] = import_zip_voice_pack_to_rows(uploaded_zip_pack, st.session_state.get("eleven_script_rows", []), video_paths=p_vids, gemini_key=gemini_api_key)
                        st.session_state["eleven_step_a_confirmed"] = True
                        st.session_state["eleven_step_b_ready"] = True
                        st.rerun()

            sa1, sa2, sa3 = st.columns([0.33, 0.44, 0.23])
            with sa1:
                analyze_script_clicked = st.button(
                    "🔍 1. ĐỌC KỊCH BẢN ĐÃ DÁN & QUÉT 6 KHUNG HÌNH" if _has_script_in else "🔒 1. ĐỌC KỊCH BẢN (CHƯA DÁN KỊCH BẢN)",
                    use_container_width=True,
                    disabled=(not _has_script_in)
                )
            with sa2:
                ai_rewrite_clicked = st.button(
                    "✨ 2. RÀ SOÁT LẠI TOÀN BỘ VO THEO HÌNH ẢNH" if _has_rows else "🔒 2. RÀ SOÁT VO (CHƯA CÓ BẢNG KỊCH BẢN)",
                    use_container_width=True,
                    disabled=(not _has_rows)
                )
            with sa3:
                confirm_step_a_clicked = st.button(
                    "✅ 3. XÁC NHẬN DUYỆT KỊCH BẢN" if _has_rows else "🔒 3. DUYỆT KỊCH BẢN (CHƯA CÓ BẢNG)",
                    use_container_width=True,
                    disabled=(not _has_rows)
                )

            step_a_hud = st.empty()

            if auto_director_master_btn:
                act_vids = _get_all_loaded_video_paths()
                if not act_vids:
                    st.warning("⚠️ Vui lòng chọn Video ở Mục 1 trước khi bấm nút Đạo Diễn AI!")
                else:
                    t0 = time.time()
                    render_live_task_hud(step_a_hud, 0.03, "ĐẠO DIỄN AI ĐANG QUÉT VIDEO", "Đang dò chuyển cảnh & chia khung giờ VO động...", t0)
                    vid_tot_d, planned_slots = build_dynamic_video_vo_schedule(
                        act_vids, vo_dur_mode=int(vo_dur_mode), min_gap_sec=float(asmr_min_gap), max_gap_sec=float(asmr_max_gap)
                    )
                    total_dur_str = format_timestamp_mmss(vid_tot_d)
                    rows_built, master_blocks_text, generated_en_history = [], [], []
                    tot_planned = max(1, len(planned_slots))

                    for vo_num, s_t, e_t, slot_d_int, gap_after in planned_slots:
                        render_live_task_hud(
                            step_a_hud, (vo_num - 0.5) / tot_planned,
                            f"ĐANG KIỂM TRA 6 KHUNG HÌNH & VIẾT KỊCH BẢN VO #{vo_num:02d}/{tot_planned}",
                            f"Mốc [{format_timestamp_mmss(s_t)} -> {format_timestamp_mmss(e_t)}] ({slot_d_int}s)...",
                            t0
                        )
                        th_p, st_p = extract_slot_filmstrip_from_video(act_vids, s_t, e_t, vo_num)
                        anchor_info = detect_visual_anchor_in_slot(act_vids, s_t, e_t, target_wps=target_wps)
                        anchor_x = anchor_info["anchor_sec_in_clip"]
                        anchor_w_pos = anchor_info["anchor_word_index"]
                        target_w_cnt = MANDATORY_DURATION_WORD_MAP.get(slot_d_int, int(round(slot_d_int * target_wps)))

                        vo_res = generate_multiple_vo_candidates_with_ai(
                            gemini_key=gemini_api_key, vo_index=vo_num, total_vos=tot_planned,
                            start_sec=s_t, end_sec=e_t, slot_dur=slot_d_int, target_words=target_w_cnt,
                            anchor_sec_x=anchor_x, anchor_word_pos=anchor_w_pos, next_gap_sec=gap_after,
                            total_video_dur_str=total_dur_str, strip_image_path=st_p or th_p,
                            auto_v4_tags=auto_inject_v4_tags, prev_en_scripts=generated_en_history,
                            regen_round=0
                        )
                        generated_en_history.append(vo_res["en_text"])
                        master_blocks_text.append(vo_res["formatted_block"])

                        m_inf = compute_script_timing_metrics(
                            vo_res["en_text"], float(slot_d_int), eleven_speed, target_wps,
                            min_words_limit, max_words_limit, slot_fill_pct, breath_gap_sec, punct_pause_ms
                        )
                        rows_built.append({
                            "STT": vo_num, "Mốc Bắt Đầu": format_timestamp_srt(s_t), "Mốc Kết Thúc": format_timestamp_srt(e_t),
                            "Khung Giờ (s)": float(slot_d_int),
                            "Điểm Chạm & Từ Neo": f"Giây {anchor_x}s -> ({vo_res['anchor_verb']}) | Nghỉ {gap_after}s",
                            "Kiểm Tra Dụng Cụ & Hành Động": vo_res.get("audit_summary", ""),
                            "Số Từ": vo_res["actual_words"], "Tối Đa Từ Chuẩn": target_w_cnt,
                            "Thời Gian Cảm Xúc (s)": m_inf["emotion_sec"], "Tốc Độ Từ (ms/từ)": m_inf["ms_per_word"],
                            "Voice Ước Tính (s)": m_inf["est_sec"], "Voice Thực Tế (s)": 0.0,
                            "Trạng Thái Khớp Nhịp": m_inf["status"],
                            "Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)": vo_res["en_text"],
                            "Phụ Đề Tiếng Việt (Dễ Hiểu)": vo_res["vi_text"],
                            "candidates": vo_res.get("candidates", []),
                            "draft_candidates": [], "user_vi_cmd": "",
                            "regen_round": 0, "rejected_en_list": [],
                            "anchor_sec_x": anchor_x, "audio_path": "", "frame_path": th_p, "strip_path": st_p or th_p
                        })
                        render_live_task_hud(step_a_hud, vo_num / tot_planned, f"HOÀN TẤT VO #{vo_num:02d}/{tot_planned}", vo_res["vi_text"][:75], t0)

                    step_a_hud.empty()
                    st.session_state["eleven_raw_text_buffer"] = "\n".join(master_blocks_text)
                    st.session_state["eleven_script_rows"] = rows_built
                    st.session_state["eleven_step_a_confirmed"] = False
                    st.session_state["eleven_step_b_ready"] = False
                    st.rerun()

            if analyze_script_clicked:
                t0 = time.time()
                act_vids = _get_all_loaded_video_paths()
                vid_tot_d, _ = get_video_timeline_info(act_vids) if act_vids else (1766.0, [])
                txt_parse = raw_script_input.strip()
                if not txt_parse:
                    st.warning("⚠️ Vui lòng dán kịch bản vào ô trên hoặc bấm nút màu cam '🎬 AI Quét Video Này'!")
                else:
                    render_live_task_hud(step_a_hud, 0.05, "ĐANG ĐỌC KỊCH BẢN & QUÉT 6 KHUNG HÌNH", "Đang bóc tách mốc thời gian...", t0)
                    tmp_srt = os.path.join(CACHE_DIR, "temp_parse_script.srt")
                    with open(tmp_srt, "w", encoding="utf-8") as f:
                        f.write(txt_parse)
                    parsed_blocks = AudioProcessor(model_size=whisper_model_size).parse_script_file(tmp_srt).get("segments", [])
                    if not parsed_blocks:
                        lines = [ln.strip() for ln in re.split(r'\n+', txt_parse) if ln.strip()]
                        _, dyn_slots = build_dynamic_video_vo_schedule(act_vids, vo_dur_mode=int(vo_dur_mode), min_gap_sec=float(asmr_min_gap), max_gap_sec=float(asmr_max_gap))
                        parsed_blocks = []
                        for idx_l, ln in enumerate(lines):
                            if idx_l < len(dyn_slots):
                                _, s_t, e_t, d_int, _ = dyn_slots[idx_l]
                            else:
                                prev_e = parsed_blocks[-1]["end"] if parsed_blocks else 5.0
                                s_t, e_t, d_int = round(prev_e + float(asmr_min_gap), 3), round(min(vid_tot_d, prev_e + float(asmr_min_gap) + 15.0), 3), 15
                            parsed_blocks.append({"start": s_t, "end": e_t, "duration": float(d_int), "text": ln, "vi_text": ""})

                    rows_built = []
                    for idx_b, blk in enumerate(parsed_blocks):
                        s_t = float(int(round(float(blk["start"]))))
                        e_t = float(int(round(float(blk["end"]))))
                        slot_d = round(max(1.0, e_t - s_t), 2)
                        render_live_task_hud(
                            step_a_hud, (idx_b + 1) / len(parsed_blocks),
                            f"ĐANG QUÉT 6 KHUNG HÌNH & DỊCH PHỤ ĐỀ CÂU #{idx_b+1}/{len(parsed_blocks)}",
                            f"Mốc {format_timestamp_mmss(s_t)} -> {format_timestamp_mmss(e_t)}", t0
                        )
                        txt_b = _clean_srt_english_line(blk["text"])
                        th_p, st_p = extract_slot_filmstrip_from_video(act_vids, s_t, e_t, idx_b + 1) if act_vids else ("", "")
                        anchor_inf = detect_visual_anchor_in_slot(act_vids, s_t, e_t, target_wps=target_wps) if act_vids else {"anchor_sec_in_clip": 4, "anchor_word_index": 10}
                        vi_sub = translate_to_vietnamese_subtitle(gemini_api_key, txt_b, blk.get("vi_text", ""))
                        m_inf = compute_script_timing_metrics(txt_b, slot_d, eleven_speed, target_wps, min_words_limit, max_words_limit, slot_fill_pct, breath_gap_sec, punct_pause_ms)
                        cached_p = get_cached_elevenlabs_mp3(txt_b, eleven_voice_id, eleven_model_id, eleven_speed, eleven_stability, eleven_similarity, eleven_style)
                        real_d_cached = 0.0
                        if cached_p:
                            cached_p, real_d_cached = lock_voice_audio_to_exact_slot(cached_p, slot_d, idx_b + 1)

                        rows_built.append({
                            "STT": idx_b + 1, "Mốc Bắt Đầu": format_timestamp_srt(s_t), "Mốc Kết Thúc": format_timestamp_srt(e_t),
                            "Khung Giờ (s)": slot_d, "Điểm Chạm & Từ Neo": f"Giây {anchor_inf['anchor_sec_in_clip']}s -> Từ #{anchor_inf['anchor_word_index']}",
                            "Kiểm Tra Dụng Cụ & Hành Động": "",
                            "Số Từ": m_inf["word_count"], "Tối Đa Từ Chuẩn": m_inf["ideal_max_words"],
                            "Thời Gian Cảm Xúc (s)": m_inf["emotion_sec"], "Tốc Độ Từ (ms/từ)": m_inf["ms_per_word"],
                            "Voice Ước Tính (s)": m_inf["est_sec"], "Voice Thực Tế (s)": real_d_cached,
                            "Trạng Thái Khớp Nhịp": m_inf["status"], "Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)": txt_b,
                            "Phụ Đề Tiếng Việt (Dễ Hiểu)": vi_sub, "candidates": [], "draft_candidates": [], "user_vi_cmd": "",
                            "regen_round": 0, "rejected_en_list": [],
                            "anchor_sec_x": anchor_inf["anchor_sec_in_clip"],
                            "audio_path": cached_p, "frame_path": th_p, "strip_path": st_p or th_p
                        })
                    step_a_hud.empty()
                    st.session_state["eleven_script_rows"] = run_self_diagnostic_and_auto_repair(rows_built, gemini_api_key, target_wps=target_wps)
                    st.rerun()

            if ai_rewrite_clicked:
                cur_rows = st.session_state.get("eleven_script_rows", [])
                if cur_rows:
                    t0 = time.time()
                    act_vids = _get_all_loaded_video_paths()
                    vid_tot_d, _ = get_video_timeline_info(act_vids) if act_vids else (1766.0, [])
                    total_dur_str = format_timestamp_mmss(vid_tot_d)
                    history_en: List[str] = []
                    master_blocks_new: List[str] = []
                    for ir, r_it in enumerate(cur_rows):
                        s_t = float(int(round(parse_timestamp_srt(str(r_it.get("Mốc Bắt Đầu", "00:00:05,000"))))))
                        e_t = float(int(round(parse_timestamp_srt(str(r_it.get("Mốc Kết Thúc", "00:00:25,000"))))))
                        slot_d = int(round(max(5.0, e_t - s_t)))
                        user_cmd_now = str(st.session_state.get(f"user_hint_vo_{ir}", r_it.get("user_vi_cmd", ""))).strip()
                        r_it["user_vi_cmd"] = user_cmd_now

                        render_live_task_hud(
                            step_a_hud, (ir + 0.5) / len(cur_rows),
                            f"ĐANG PHÂN TÍCH HÌNH ẢNH & LỆNH TIẾNG VIỆT VO #{ir+1}/{len(cur_rows)}",
                            f"Mốc [{format_timestamp_mmss(s_t)} -> {format_timestamp_mmss(e_t)}]...", t0
                        )
                        th_p, st_p = extract_slot_filmstrip_from_video(act_vids, s_t, e_t, ir + 1) if act_vids else ("", "")
                        anchor_x = int(r_it.get("anchor_sec_x", max(2, min(8, int(round(slot_d * 0.35))))))
                        anchor_w_pos = max(4, int(round(anchor_x * target_wps)))
                        target_w_cnt = MANDATORY_DURATION_WORD_MAP.get(slot_d, int(round(slot_d * target_wps)))

                        vo_res = generate_multiple_vo_candidates_with_ai(
                            gemini_key=gemini_api_key, vo_index=ir + 1, total_vos=len(cur_rows),
                            start_sec=s_t, end_sec=e_t, slot_dur=slot_d, target_words=target_w_cnt,
                            anchor_sec_x=anchor_x, anchor_word_pos=anchor_w_pos, next_gap_sec=60,
                            total_video_dur_str=total_dur_str, strip_image_path=st_p or th_p or r_it.get("strip_path", ""),
                            auto_v4_tags=auto_inject_v4_tags, prev_en_scripts=history_en,
                            user_custom_hint=user_cmd_now,
                            current_en_script=r_it.get("Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)", "")
                        )
                        history_en.append(vo_res["en_text"])
                        master_blocks_new.append(vo_res["formatted_block"])
                        m_af = compute_script_timing_metrics(vo_res["en_text"], float(slot_d), eleven_speed, target_wps, min_words_limit, max_words_limit, slot_fill_pct, breath_gap_sec, punct_pause_ms)
                        r_it.update({
                            "Mốc Bắt Đầu": format_timestamp_srt(s_t), "Mốc Kết Thúc": format_timestamp_srt(e_t),
                            "Khung Giờ (s)": float(slot_d), "Điểm Chạm & Từ Neo": f"Giây {anchor_x}s -> ({vo_res['anchor_verb']})",
                            "Kiểm Tra Dụng Cụ & Hành Động": vo_res.get("audit_summary", ""),
                            "Số Từ": m_af["word_count"], "Tối Đa Từ Chuẩn": m_af["ideal_max_words"],
                            "Thời Gian Cảm Xúc (s)": m_af["emotion_sec"], "Tốc Độ Từ (ms/từ)": m_af["ms_per_word"],
                            "Voice Ước Tính (s)": m_af["est_sec"], "Trạng Thái Khớp Nhịp": m_af["status"],
                            "Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)": vo_res["en_text"],
                            "Phụ Đề Tiếng Việt (Dễ Hiểu)": vo_res["vi_text"],
                            "candidates": vo_res.get("candidates", []),
                            "frame_path": th_p or r_it.get("frame_path", ""), "strip_path": st_p or r_it.get("strip_path", "")
                        })
                    step_a_hud.empty()
                    st.session_state["eleven_raw_text_buffer"] = "\n".join(master_blocks_new)
                    st.session_state["eleven_script_rows"] = cur_rows
                    st.rerun()

            if confirm_step_a_clicked and st.session_state.get("eleven_script_rows"):
                st.session_state["eleven_step_a_confirmed"] = True
                st.rerun()

            if st.session_state.get("eleven_script_rows"):
                st.session_state["eleven_script_rows"] = run_self_diagnostic_and_auto_repair(
                    st.session_state["eleven_script_rows"], gemini_api_key, target_wps=target_wps
                )

                disp_cols = [
                    "STT", "Mốc Bắt Đầu", "Mốc Kết Thúc", "Khung Giờ (s)",
                    "Điểm Chạm & Từ Neo", "Kiểm Tra Dụng Cụ & Hành Động", "Số Từ", "Tối Đa Từ Chuẩn",
                    "Thời Gian Cảm Xúc (s)", "Tốc Độ Từ (ms/từ)", "Voice Ước Tính (s)", "Voice Thực Tế (s)",
                    "Trạng Thái Khớp Nhịp", "Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)", "Phụ Đề Tiếng Việt (Dễ Hiểu)"
                ]
                clean_tb = [{k: r.get(k, "") for k in disp_cols} for r in st.session_state["eleven_script_rows"]]
                tb_hash = hashlib.md5(json.dumps([r.get("Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)", "") + r.get("Phụ Đề Tiếng Việt (Dễ Hiểu)", "") for r in clean_tb]).encode("utf-8")).hexdigest()[:6]
                edited_sc_tb = st.data_editor(clean_tb, use_container_width=True, height=300, num_rows="fixed", key=f"el_script_editor_{tb_hash}")

                for ir, ed_r in enumerate(edited_sc_tb):
                    if ir < len(st.session_state["eleven_script_rows"]):
                        orig_r = st.session_state["eleven_script_rows"][ir]
                        s_new = parse_timestamp_srt(str(ed_r.get("Mốc Bắt Đầu", "00:00:00,000")))
                        e_new = parse_timestamp_srt(str(ed_r.get("Mốc Kết Thúc", "00:00:05,000")))
                        sl_new = round(max(0.5, e_new - s_new), 2)
                        tx_new = str(ed_r.get("Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)", ""))
                        vi_new = str(ed_r.get("Phụ Đề Tiếng Việt (Dễ Hiểu)", orig_r.get("Phụ Đề Tiếng Việt (Dễ Hiểu)", "")))
                        # Ngăn trường hợp Chrome dịch cột tiếng Anh sang tiếng Việt khi người dùng chỉnh bảng
                        if _has_vn_chars(re.sub(r'\[[^\]]+\]', '', tx_new)):
                            tx_new = orig_r.get("Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)", tx_new)
                        m_u = compute_script_timing_metrics(tx_new, sl_new, eleven_speed, target_wps, min_words_limit, max_words_limit, slot_fill_pct, breath_gap_sec, punct_pause_ms)
                        orig_r.update({
                            "Mốc Bắt Đầu": format_timestamp_srt(s_new), "Mốc Kết Thúc": format_timestamp_srt(e_new),
                            "Khung Giờ (s)": sl_new, "Số Từ": m_u["word_count"], "Tối Đa Từ Chuẩn": m_u["ideal_max_words"],
                            "Thời Gian Cảm Xúc (s)": m_u["emotion_sec"], "Tốc Độ Từ (ms/từ)": m_u["ms_per_word"],
                            "Voice Ước Tính (s)": m_u["est_sec"], "Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)": tx_new,
                            "Phụ Đề Tiếng Việt (Dễ Hiểu)": vi_new
                        })

                dl_c1, dl_c2 = st.columns(2)
                with dl_c1:
                    st.download_button("📥 Tải Kịch Bản Đạo Diễn (.TXT)", data=st.session_state.get("eleven_raw_text_buffer", "").encode("utf-8"), file_name="Kich_Ban_Dao_Dien_AI.txt", mime="text/plain", use_container_width=True)
                with dl_c2:
                    srt_export_lines = [f"{int(r['STT'])}\n{r['Mốc Bắt Đầu']} --> {r['Mốc Kết Thúc']}\n{r['Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)']}\n" for r in st.session_state["eleven_script_rows"]]
                    st.download_button("📥 Tải File Phụ Đề Chuẩn Mili-Giây (.SRT)", data="\n".join(srt_export_lines).encode("utf-8"), file_name="Subtitles_Timeline.srt", mime="text/plain", use_container_width=True)

                # ==============================================================
                # TRẠM KIỂM SOÁT & TIẾT KIỆM CREDIT ELEVENLABS PRO TRƯỚC KHI TẠO
                # ==============================================================
                rows_all = st.session_state["eleven_script_rows"]
                cached_cnt = 0
                need_gen_cnt = 0
                est_chars_to_bill = 0
                saved_tag_chars = 0
                for r_c in rows_all:
                    raw_t = str(r_c.get("Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)", ""))
                    api_t = strip_non_spoken_tags_for_credit_saving(raw_t, strip_all_tags=strip_tags_credit_saver)
                    has_aud = bool(r_c.get("audio_path") and os.path.exists(str(r_c.get("audio_path", ""))))
                    is_in_disk = bool(get_cached_elevenlabs_mp3(api_t, eleven_voice_id, eleven_model_id, eleven_speed, eleven_stability, eleven_similarity, eleven_style))
                    if (strict_credit_saver and has_aud) or is_in_disk:
                        cached_cnt += 1
                    else:
                        need_gen_cnt += 1
                        est_chars_to_bill += len(api_t)
                        saved_tag_chars += max(0, len(raw_t) - len(api_t))

                st.markdown(
                    f'<div translate="no" class="notranslate" style="background:#091326;border:1.5px solid #38bdf8;border-radius:10px;padding:10px 16px;margin:10px 0;display:flex;justify-content:space-between;align-items:center;font-size:13px;">'
                    f'<span>🛡️ <b>TRẠM BẢO VỆ CREDIT ELEVENLABS PRO:</b> Đã có sẵn Voice (0 tốn Credit): <b style="color:#4ade80;">{cached_cnt}/{len(rows_all)} câu</b> | Cần tạo mới: <b style="color:#facc15;">{need_gen_cnt} câu</b></span>'
                    f'<span>💎 Ước tính tốn: <b style="color:#facc15;">{est_chars_to_bill:,} ký tự</b> (Đã tiết kiệm <b style="color:#4ade80;">+{saved_tag_chars:,} ký tự thẻ ẩn</b>)</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                bc1, bc2, bc3 = st.columns([0.44, 0.28, 0.28])
                with bc1:
                    gen_all_eleven_clicked = st.button(
                        "🎙️ XÁC NHẬN TẠO VOICE CHO CÁC CÂU CHƯA CÓ" if (_has_el_key and _has_rows) else "🔒 TẠO VOICE (CẦN DUYỆT BẢNG & NHẬP ELEVENLABS KEY)",
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
                        st.download_button(f"📦 Tải Bộ {len(ready_mp3s)} Voice (.ZIP)", data=z_buf.getvalue(), file_name=f"ElevenLabs_Voices_{datetime.now().strftime('%H%M%S')}.zip", mime="application/zip", use_container_width=True)
                with bc3:
                    dub_now_clicked = st.button(
                        "🚀 BƯỚC C: GHÉP VOICE VÀO VIDEO ĐÃ CẮT SẴN (GIỮ 100% ASMR)" if (_has_vids and _has_voices) else "🔒 BƯỚC C: GHÉP VOICE (CẦN TẠO VOICE TRƯỚC)",
                        use_container_width=True,
                        disabled=(not _has_vids or not _has_voices)
                    )

                step_b_hud = st.empty()

                if gen_all_eleven_clicked:
                    if not clean_eleven_api_key(elevenlabs_api_key):
                        st.error("⚠️ Vui lòng nhập ElevenLabs API Key (sk_...) ở thanh bên trái!")
                    else:
                        t0_v = time.time()
                        st.session_state["eleven_step_a_confirmed"] = True
                        rows_gen = st.session_state["eleven_script_rows"]
                        ok_c = 0
                        for ig, rg in enumerate(rows_gen):
                            slot_d = float(rg["Khung Giờ (s)"])
                            cur_tx = str(rg["Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)"])

                            # Nếu bật strict_credit_saver và câu này đã có file Voice hợp lệ thì giữ nguyên 100%, không gọi API!
                            if strict_credit_saver and rg.get("audio_path") and os.path.exists(str(rg.get("audio_path", ""))):
                                ok_c += 1
                                continue

                            render_live_task_hud(
                                step_b_hud, (ig + 0.3) / len(rows_gen),
                                f"ĐANG TẠO VOICE ELEVENLABS CÂU #{rg['STT']}/{len(rows_gen)}",
                                f"Đang tổng hợp giọng đọc & khóa chuẩn khung {slot_d}s...", t0_v
                            )
                            res_tts = generate_elevenlabs_tts_segment(
                                elevenlabs_api_key, cur_tx, eleven_voice_id, eleven_model_id,
                                eleven_speed, eleven_stability, eleven_similarity, eleven_style,
                                force_regenerate=False, strip_tags_for_credit=strip_tags_credit_saver
                            )
                            if res_tts.get("ok"):
                                a_path_final, real_d = lock_voice_audio_to_exact_slot(res_tts["audio_path"], slot_d, int(rg["STT"]))
                                was_cached = bool(res_tts.get("cached", False))
                                m_done = compute_script_timing_metrics(cur_tx, slot_d, eleven_speed, target_wps, min_words_limit, max_words_limit, slot_fill_pct, breath_gap_sec, punct_pause_ms)
                                cache_tag = "💎 Cache 0 Tốn Credit" if was_cached else "✅ Mới tạo (Khớp 100%)"
                                rg.update({
                                    "audio_path": a_path_final, "Số Từ": m_done["word_count"],
                                    "Thời Gian Cảm Xúc (s)": m_done["emotion_sec"], "Voice Thực Tế (s)": real_d,
                                    "Tốc Độ Từ (ms/từ)": int(round((real_d / max(1, m_done["word_count"])) * 1000)),
                                    "Trạng Thái Khớp Nhịp": f"{cache_tag} ({real_d}s / {slot_d}s | {m_done['word_count']}/{m_done['ideal_max_words']} từ)"
                                })
                                ok_c += 1
                            render_live_task_hud(step_b_hud, (ig + 1) / len(rows_gen), f"HOÀN TẤT VOICE #{rg['STT']}/{len(rows_gen)}", f"Thời lượng thực tế: {rg.get('Voice Thực Tế (s)', 0)}s / {slot_d}s", t0_v)
                        step_b_hud.empty()
                        st.session_state["eleven_script_rows"] = rows_gen
                        st.session_state["eleven_step_b_ready"] = (ok_c > 0)
                        st.rerun()

                if dub_now_clicked:
                    st.session_state["trigger_dub_only_render"] = True

        # ======================================================================
        # XEM VIDEO XEM THỬ + LỆNH TIẾNG VIỆT TRỰC TIẾP + BẢN NHÁP XEM TRƯỚC (0 TỐN CREDIT)
        # ======================================================================
        if st.session_state.get("eleven_script_rows"):
            act_vids_preview = _get_all_loaded_video_paths()
            vid_tot_d_pv, _ = get_video_timeline_info(act_vids_preview) if act_vids_preview else (1766.0, [])
            tot_dur_str_pv = format_timestamp_mmss(vid_tot_d_pv)

            with st.expander("🎬 XEM TẤT CẢ VIDEO XEM THỬ + NHẬP LỆNH TIẾNG VIỆT SỬA TRỰC TIẾP & XEM BẢN NHÁP TRƯỚC (0 TỐN CREDIT)", expanded=True):
                pv_top1, pv_top2 = st.columns([0.55, 0.45])
                with pv_top1:
                    show_all_videos_now = st.checkbox(
                        "🎬 Hiển thị Video xem thử đã tạo cho toàn bộ các câu",
                        value=True
                    )
                with pv_top2:
                    build_all_clips_btn = st.button("⚡ TẠO ĐỒNG LOẠT VIDEO XEM THỬ TẤT CẢ CÁC CÂU (HIỆN %)", use_container_width=True)

                preview_section_hud = st.empty()

                if build_all_clips_btn and act_vids_preview:
                    t0_pv = time.time()
                    all_r = st.session_state["eleven_script_rows"]
                    for idx_p, r_p in enumerate(all_r):
                        s_p = parse_timestamp_srt(str(r_p["Mốc Bắt Đầu"]))
                        e_p = parse_timestamp_srt(str(r_p["Mốc Kết Thúc"]))
                        render_live_task_hud(
                            preview_section_hud, (idx_p + 0.3) / max(1, len(all_r)),
                            f"ĐANG CẮT & GHÉP VIDEO XEM THỬ CÂU #{idx_p+1}/{len(all_r)}",
                            f"Mốc {r_p['Mốc Bắt Đầu']} --> {r_p['Mốc Kết Thúc']} (Chuẩn MP4 FastStart)", t0_pv
                        )
                        create_synced_preview_video_clip(act_vids_preview, s_p, e_p, r_p.get("audio_path", ""), int(r_p["STT"]))
                        st.session_state[f"show_vid_clip_{idx_p}"] = True
                        render_live_task_hud(preview_section_hud, (idx_p + 1) / max(1, len(all_r)), f"ĐÃ TẠO XONG VIDEO #{idx_p+1}/{len(all_r)}", "Sẵn sàng phát trực tiếp!", t0_pv)
                    preview_section_hud.empty()
                    st.rerun()

                for ip, rp in enumerate(st.session_state["eleven_script_rows"]):
                    s_sec_p = parse_timestamp_srt(str(rp["Mốc Bắt Đầu"]))
                    e_sec_p = parse_timestamp_srt(str(rp["Mốc Kết Thúc"]))
                    slot_d_p = int(round(max(5.0, e_sec_p - s_sec_p)))
                    vi_sub_line = rp.get("Phụ Đề Tiếng Việt (Dễ Hiểu)", "")

                    cv_col, ca_col, cb_col = st.columns([0.33, 0.44, 0.23])
                    with cv_col:
                        want_video_here = bool(show_all_videos_now or st.session_state.get(f"show_vid_clip_{ip}", False))
                        existing_clip_mp4 = get_existing_preview_clip_path(
                            act_vids_preview, s_sec_p, e_sec_p, rp.get("audio_path", ""), int(rp["STT"])
                        ) if act_vids_preview else ""

                        # Nếu người dùng bật xem video mà clip của câu này chưa được cắt -> Tự động cắt ngay!
                        if want_video_here and not existing_clip_mp4 and act_vids_preview:
                            existing_clip_mp4 = create_synced_preview_video_clip(
                                act_vids_preview, s_sec_p, e_sec_p, rp.get("audio_path", ""), int(rp["STT"])
                            )

                        if want_video_here and existing_clip_mp4 and os.path.exists(existing_clip_mp4):
                            st.video(existing_clip_mp4)
                        else:
                            img_show = rp.get("strip_path") or rp.get("frame_path")
                            if img_show and os.path.exists(img_show):
                                st.image(img_show, use_container_width=True)
                        st.caption(f"🎯 Mốc chuẩn: **`{rp['Mốc Bắt Đầu']} --> {rp['Mốc Kết Thúc']}`** | {rp.get('Điểm Chạm & Từ Neo', '')}")

                    with ca_col:
                        emo_s = float(rp.get("Thời Gian Cảm Xúc (s)", 0.0))
                        st.markdown(
                            f"**VO #{rp['STT']:02d}** (`{rp['Số Từ']}/{rp['Tối Đa Từ Chuẩn']} từ` | "
                            f"🎭 Cảm xúc: `{emo_s}s` | Thực tế: `{rp['Voice Thực Tế (s)']}s`/`{rp['Khung Giờ (s)']}s`) — {rp['Trạng Thái Khớp Nhịp']}"
                        )
                        if rp.get("Kiểm Tra Dụng Cụ & Hành Động"):
                            st.markdown(
                                f'<div style="background:#1e293b;border-left:3px solid #38bdf8;padding:5px 10px;border-radius:6px;margin-bottom:6px;font-size:12.5px;color:#bae6fd;">'
                                f'{rp["Kiểm Tra Dụng Cụ & Hành Động"]}</div>',
                                unsafe_allow_html=True
                            )
                        # Khóa translate="no" để Chrome không bao giờ dịch câu tiếng Anh sang tiếng Việt trên màn hình!
                        st.markdown(
                            f'<div translate="no" class="notranslate" style="background:#0f172a;border:1px solid rgba(56,189,248,0.4);padding:9px 12px;border-radius:8px;font-size:14px;font-weight:600;color:#ffffff;margin-bottom:6px;">'
                            f'🇬🇧 {rp["Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)"]}</div>',
                            unsafe_allow_html=True
                        )
                        if rp.get("audio_path") and os.path.exists(str(rp["audio_path"])):
                            st.audio(rp["audio_path"])
                        st.markdown(
                            f'<div style="background:#064e3b;border-left:4px solid #34d399;padding:8px 12px;border-radius:8px;margin-top:6px;font-size:13.5px;color:#ecfdf5;">'
                            f'🇻🇳 <b>Phụ đề Tiếng Việt:</b> {vi_sub_line}</div>',
                            unsafe_allow_html=True
                        )

                        # ======================================================
                        # HỘP XEM TRƯỚC BẢN NHÁP (DRAFT PREVIEW) TRƯỚC KHI ÁP DỤNG
                        # ======================================================
                        draft_list = rp.get("draft_candidates") or []
                        if draft_list:
                            st.markdown(
                                f'<div style="background:#3b0764;border:2px solid #facc15;border-radius:10px;padding:8px 12px;margin-top:8px;">'
                                f'<b style="color:#facc15;font-size:13.5px;">📋 BẢN NHÁP XEM TRƯỚC CHO VO #{rp["STT"]:02d} (0 TỐN CREDIT — BẤM CHỌN ĐỂ ÁP DỤNG):</b></div>',
                                unsafe_allow_html=True
                            )
                            for d_i, d_item in enumerate(draft_list):
                                st.markdown(
                                    f'<div translate="no" class="notranslate" style="background:#090d16;border:1.5px solid #a855f7;border-radius:8px;padding:8px 11px;margin-top:5px;font-size:12.5px;">'
                                    f'<b style="color:#e879f9;">{d_item.get("style_label", f"Bản nháp {d_i+1}")}</b> '
                                    f'(`{d_item.get("actual_words", 0)}/{d_item.get("target_words", 0)} từ` | Động từ: `{d_item.get("anchor_verb", "")}`)<br>'
                                    f'🇬🇧 {d_item.get("en_text", "")}<br>'
                                    f'<span translate="yes" style="color:#6ee7b7;">🇻🇳 {d_item.get("vi_text", "")}</span></div>',
                                    unsafe_allow_html=True
                                )
                                if st.button(f"✅ Áp Dụng Bản Nháp {d_i + 1} Vào Kịch Bản VO #{rp['STT']}", key=f"apply_draft_{ip}_{d_i}", type="primary", use_container_width=True):
                                    m_dr = compute_script_timing_metrics(
                                        d_item["en_text"], float(slot_d_p), eleven_speed, target_wps,
                                        min_words_limit, max_words_limit, slot_fill_pct, breath_gap_sec, punct_pause_ms
                                    )
                                    ax_d = int(rp.get("anchor_sec_x", max(2, min(8, int(round(slot_d_p * 0.35))))))
                                    rp.update({
                                        "Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)": d_item["en_text"],
                                        "Phụ Đề Tiếng Việt (Dễ Hiểu)": d_item["vi_text"],
                                        "Số Từ": m_dr["word_count"],
                                        "Tối Đa Từ Chuẩn": m_dr["ideal_max_words"],
                                        "Voice Ước Tính (s)": m_dr["est_sec"],
                                        "Trạng Thái Khớp Nhịp": m_dr["status"],
                                        "Điểm Chạm & Từ Neo": f"Giây {ax_d}s -> ({d_item.get('anchor_verb', 'PLACES')})",
                                        "candidates": draft_list,
                                        "draft_candidates": []
                                    })
                                    st.session_state["eleven_raw_text_buffer"] = "\n".join(
                                        f"{int(_r['STT'])}\n{_r['Mốc Bắt Đầu']} --> {_r['Mốc Kết Thúc']}\n{_clean_srt_english_line(_r['Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)'])}\n"
                                        for _r in st.session_state["eleven_script_rows"]
                                    )
                                    st.rerun()
                            if st.button(f"🗑️ Đóng / Hủy Bản Nháp VO #{rp['STT']} (Giữ Nguyên Câu Cũ)", key=f"cancel_draft_{ip}", use_container_width=True):
                                rp["draft_candidates"] = []
                                st.rerun()

                        # Hiển thị 3 phương án sẵn có khi không ở chế độ nháp
                        cands_list = rp.get("candidates") or []
                        if len(cands_list) > 1 and not draft_list:
                            with st.expander(f"🎯 Xem & Đổi Nhanh 3 Phương Án Kịch Bản Của VO #{rp['STT']:02d} (Chuẩn {rp['Tối Đa Từ Chuẩn']} từ)", expanded=False):
                                for c_i, cand_item in enumerate(cands_list):
                                    is_active_opt = (cand_item.get("en_text", "").strip() == rp.get("Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)", "").strip())
                                    border_col = "#facc15" if is_active_opt else "rgba(56,189,248,0.35)"
                                    badge_active = "🌟 [ĐANG DÙNG]" if is_active_opt else ""
                                    st.markdown(
                                        f'<div translate="no" class="notranslate" style="background:#091326;border:1.5px solid {border_col};border-radius:8px;padding:7px 11px;margin-top:5px;font-size:12.5px;">'
                                        f'<b style="color:#facc15;">{cand_item.get("style_label", f"Phương án {c_i+1}")} {badge_active}</b> '
                                        f'(`{cand_item.get("actual_words", 0)}/{cand_item.get("target_words", 0)} từ` | Động từ: `{cand_item.get("anchor_verb", "")}`)<br>'
                                        f'🇬🇧 {cand_item.get("en_text", "")}<br>'
                                        f'<span translate="yes" style="color:#6ee7b7;">🇻🇳 {cand_item.get("vi_text", "")}</span></div>',
                                        unsafe_allow_html=True
                                    )
                                    if not is_active_opt:
                                        if st.button(f"✅ Chọn Phương Án {c_i + 1} cho VO #{rp['STT']}", key=f"sel_cand_{ip}_{c_i}", use_container_width=True):
                                            m_sel = compute_script_timing_metrics(
                                                cand_item["en_text"], float(slot_d_p), eleven_speed, target_wps,
                                                min_words_limit, max_words_limit, slot_fill_pct, breath_gap_sec, punct_pause_ms
                                            )
                                            ax_s = int(rp.get("anchor_sec_x", max(2, min(8, int(round(slot_d_p * 0.35))))))
                                            rp.update({
                                                "Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)": cand_item["en_text"],
                                                "Phụ Đề Tiếng Việt (Dễ Hiểu)": cand_item["vi_text"],
                                                "Số Từ": m_sel["word_count"],
                                                "Tối Đa Từ Chuẩn": m_sel["ideal_max_words"],
                                                "Voice Ước Tính (s)": m_sel["est_sec"],
                                                "Trạng Thái Khớp Nhịp": m_sel["status"],
                                                "Điểm Chạm & Từ Neo": f"Giây {ax_s}s -> ({cand_item.get('anchor_verb', 'PLACES')})"
                                            })
                                            st.session_state["eleven_raw_text_buffer"] = "\n".join(
                                                f"{int(_r['STT'])}\n{_r['Mốc Bắt Đầu']} --> {_r['Mốc Kết Thúc']}\n{_clean_srt_english_line(_r['Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)'])}\n"
                                                for _r in st.session_state["eleven_script_rows"]
                                            )
                                            st.rerun()

                    with cb_col:
                        user_hint_val = st.text_area(
                            f"📝 Nhập lệnh Tiếng Việt sửa VO #{rp['STT']}:",
                            value=st.session_state.get(f"user_hint_vo_{ip}", rp.get("user_vi_cmd", "")),
                            key=f"user_hint_vo_{ip}",
                            height=78,
                            placeholder="Ghi tiếng Việt hành động/dụng cụ bạn muốn (VD: 2 người khiêng tấm xốp trắng đặt lên sàn cạnh thang đỏ)..."
                        )
                        btn_draft_1 = st.button(
                            f"🧪 Phân Tích Lệnh & Xem Bản Nháp #{rp['STT']}",
                            key=f"btn_draft_vo_{ip}",
                            type="primary",
                            use_container_width=True,
                            help="Đọc câu lệnh Tiếng Việt bạn vừa ghi + ảnh 6 khung hình để tạo 3 Bản Nháp cho bạn xem trước (0 tốn Credit, chưa ghi đè ngay)!"
                        )
                        btn_vid_1 = st.button(f"🎬 Tạo lại Video #{rp['STT']}", key=f"btn_pv_{ip}", use_container_width=True)
                        btn_tts_1 = st.button(f"🔄 Tạo Voice Câu Này #{rp['STT']}", key=f"rg_one_{ip}", use_container_width=True)
                        row_btn_hud = st.empty()

                        if btn_draft_1:
                            t0_1 = time.time()
                            rp["user_vi_cmd"] = user_hint_val.strip()
                            new_rg_1 = int(rp.get("regen_round", 0)) + 1
                            rej_1 = list(rp.get("rejected_en_list") or [])
                            if rp.get("Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)"):
                                rej_1.append(str(rp["Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)"]))

                            render_live_task_hud(row_btn_hud, 0.25, f"VO #{rp['STT']}: ĐANG ĐỌC LỆNH TIẾNG VIỆT", "Đang đối chiếu câu lệnh với 6 khung hình...", t0_1)
                            th_1, st_1 = extract_slot_filmstrip_from_video(act_vids_preview, s_sec_p, e_sec_p, ip + 1) if act_vids_preview else ("", "")
                            ax_1 = int(rp.get("anchor_sec_x", max(2, min(8, int(round(slot_d_p * 0.35))))))
                            aw_1 = max(4, int(round(ax_1 * target_wps)))
                            tw_1 = MANDATORY_DURATION_WORD_MAP.get(slot_d_p, int(round(slot_d_p * target_wps)))
                            prev_ens = [r["Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)"] for idx_o, r in enumerate(st.session_state["eleven_script_rows"]) if idx_o != ip]

                            render_live_task_hud(row_btn_hud, 0.65, f"VO #{rp['STT']}: ĐANG TẠO 3 BẢN NHÁP XEM TRƯỚC", "Đang cân chỉnh đúng 50/50 từ & dịch Tiếng Việt...", t0_1)
                            draft_res = generate_multiple_vo_candidates_with_ai(
                                gemini_key=gemini_api_key, vo_index=ip + 1, total_vos=len(st.session_state["eleven_script_rows"]),
                                start_sec=s_sec_p, end_sec=e_sec_p, slot_dur=slot_d_p, target_words=tw_1,
                                anchor_sec_x=ax_1, anchor_word_pos=aw_1, next_gap_sec=60,
                                total_video_dur_str=tot_dur_str_pv, strip_image_path=st_1 or th_1 or rp.get("strip_path", ""),
                                auto_v4_tags=auto_inject_v4_tags, prev_en_scripts=prev_ens,
                                user_custom_hint=user_hint_val, current_en_script=rp.get("Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)", ""),
                                regen_round=new_rg_1, rejected_en_list=rej_1
                            )
                            rp.update({
                                "Kiểm Tra Dụng Cụ & Hành Động": draft_res.get("audit_summary", ""),
                                "draft_candidates": draft_res.get("candidates", []),
                                "regen_round": new_rg_1,
                                "rejected_en_list": rej_1[-10:],
                                "frame_path": th_1 or rp.get("frame_path", ""),
                                "strip_path": st_1 or rp.get("strip_path", "")
                            })
                            row_btn_hud.empty()
                            st.rerun()

                        if btn_vid_1:
                            t0_v1 = time.time()
                            render_live_task_hud(row_btn_hud, 0.35, f"ĐANG DỰNG VIDEO CẢNH #{rp['STT']}", f"Cắt mốc {rp['Mốc Bắt Đầu']} -> {rp['Mốc Kết Thúc']}...", t0_v1)
                            st.session_state[f"show_vid_clip_{ip}"] = True
                            if act_vids_preview:
                                create_synced_preview_video_clip(act_vids_preview, s_sec_p, e_sec_p, rp.get("audio_path", ""), int(rp["STT"]))
                            row_btn_hud.empty()
                            st.rerun()

                        if btn_tts_1:
                            t0_a1 = time.time()
                            render_live_task_hud(row_btn_hud, 0.40, f"ĐANG TẠO VOICE #{rp['STT']}", "Đang gọi ElevenLabs API (Đã tối ưu Credit)...", t0_a1)
                            r1 = generate_elevenlabs_tts_segment(
                                elevenlabs_api_key, rp["Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)"],
                                eleven_voice_id, eleven_model_id, eleven_speed, eleven_stability,
                                eleven_similarity, eleven_style, force_regenerate=True,
                                strip_tags_for_credit=strip_tags_credit_saver
                            )
                            if r1.get("ok"):
                                a_locked, d_locked = lock_voice_audio_to_exact_slot(r1["audio_path"], float(rp["Khung Giờ (s)"]), int(rp["STT"]))
                                rp["audio_path"], rp["Voice Thực Tế (s)"] = a_locked, d_locked
                                if act_vids_preview and st.session_state.get(f"show_vid_clip_{ip}", False):
                                    render_live_task_hud(row_btn_hud, 0.80, f"ĐANG GHÉP VOICE MỚI VÀO VIDEO #{rp['STT']}", "Đang đồng bộ lại video xem thử...", t0_a1)
                                    create_synced_preview_video_clip(act_vids_preview, s_sec_p, e_sec_p, a_locked, int(rp["STT"]))
                                row_btn_hud.empty()
                                st.rerun()
                    st.markdown("<hr style='margin:8px 0;border-color:rgba(56,189,248,0.2);'>", unsafe_allow_html=True)

    with el_tab_clone:
        st.markdown("##### 🧬 Trạm Clone Giọng Nhanh (Instant Voice Cloning)")
        clc1, clc2, clc3 = st.columns(3)
        with clc1:
            clone_name = st.text_input("Tên giọng mới (Bắt buộc):", placeholder="VD: Quantum Narrator - Nam Trầm")
        with clc2:
            clone_gender = st.selectbox("Nhãn Giới tính:", ["male", "female"])
        with clc3:
            clone_style = st.selectbox("Nhãn Phong cách:", ["narrative", "calm", "deep", "raspy", "conversational"])
        clone_desc = st.text_input("Mô tả đặc điểm giọng:", placeholder="VD: Deep American male voice for survival documentary...")
        clone_rm_noise = st.checkbox("Tự động lọc sạch tạp âm / nhạc nền nhỏ (Remove Background Noise)", value=True)
        clone_files_up = st.file_uploader("Chọn File âm thanh mẫu (.mp3, .wav, .m4a - Dài 1-3 phút):", type=["mp3", "wav", "m4a"], accept_multiple_files=True)
        if st.button("🚀 Bắt Đầu Clone Giọng Mới Lên ElevenLabs", use_container_width=True):
            if not clone_name.strip() or not clone_files_up:
                st.warning("⚠️ Vui lòng nhập tên giọng và tải lên ít nhất 1 file mẫu!")
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

    with el_tab_html:
        html_candidates = [
            "Eleven Labstool_6.html", "Eleven Labstool_5.html", "Eleven Labstool_4.html",
            "Eleven Labstool_3.html", "Eleven Labstool_2.html", "Eleven Labstool.html"
        ]
        found_html = next((os.path.join(os.getcwd(), fn) for fn in html_candidates if os.path.exists(os.path.join(os.getcwd(), fn))), "")
        if found_html:
            with open(found_html, "r", encoding="utf-8", errors="ignore") as f:
                raw_html = f.read()
            ck_inj = clean_eleven_api_key(elevenlabs_api_key)
            script_preload_js = json.dumps(st.session_state.get("eleven_raw_text_buffer", ""))
            inject_js = (
                f'<script>'
                f'if("{ck_inj}") localStorage.setItem("elevenlabs_api_key", "{ck_inj}");'
                f'window.addEventListener("DOMContentLoaded", () => {{'
                f'  const preTxt = {script_preload_js};'
                f'  if(preTxt && document.getElementById("rawSrtInput") && preTxt.includes("-->")) {{'
                f'    document.getElementById("rawSrtInput").value = preTxt;'
                f'  }} else if(preTxt && document.getElementById("rawTextInput")) {{'
                f'    document.getElementById("rawTextInput").value = preTxt;'
                f'  }}'
                f'}});'
                f'</script>'
            )
            components.html(inject_js + raw_html, height=760, scrolling=True)
        else:
            st.info("💡 Đặt file `Eleven Labstool_6.html` (hoặc `Eleven Labstool.html`) cùng thư mục với `app.py` để hiển thị trực tiếp tại đây.")

    return {
        "eleven_model_id": eleven_model_id,
        "eleven_voice_id": eleven_voice_id,
        "eleven_speed": eleven_speed,
        "eleven_stability": eleven_stability,
        "eleven_similarity": eleven_similarity,
        "eleven_style": eleven_style,
        "target_wps": target_wps,
        "min_words_limit": min_words_limit,
        "max_words_limit": max_words_limit,
        "slot_fill_pct": slot_fill_pct,
        "breath_gap_sec": breath_gap_sec,
        "punct_pause_ms": punct_pause_ms,
        "voice_lead_ms": voice_lead_ms,
        "slot_align_mode": slot_align_mode,
        "overrun_policy": overrun_policy,
        "underrun_policy": underrun_policy,
        "max_atempo_limit": max_atempo_limit,
        "ai_rewrite_style": ai_rewrite_style,
        "auto_inject_v4_tags": auto_inject_v4_tags,
        "auto_fit_voice_slot": auto_fit_voice_slot
    }