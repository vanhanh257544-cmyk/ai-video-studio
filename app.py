import os
import re
import json
import time
import wave
import struct
import shutil
import tempfile
import subprocess
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import streamlit as st  # type: ignore

from audio_processor import AudioProcessor  # type: ignore
from vision_analyzer import VisionAnalyzer  # type: ignore
from video_editor import AutoVideoEditor  # type: ignore
from eleven_dub_engine import (  # type: ignore
    DEFAULT_ELEVEN_VOICES,
     DEFAULT_ELEVEN_MODELS,
    clean_eleven_api_key,
    connect_elevenlabs_api,
    generate_elevenlabs_tts_segment,
    parse_timestamp_srt,
    render_elevenlabs_dubbing_station_ui
)

CONFIG_FILE = "settings.json"
CACHE_DIR = os.path.join(os.getcwd(), "studio_cache")
UPLOAD_CACHE_DIR = os.path.join(CACHE_DIR, "uploads")
THUMB_CACHE_DIR = os.path.join(CACHE_DIR, "thumbnails")
ELEVEN_CACHE_DIR = os.path.join(CACHE_DIR, "elevenlabs_audio")
CLIP_PREVIEW_DIR = os.path.join(CACHE_DIR, "clip_previews")
MAX_CPU = os.cpu_count() or 4

for _d in (UPLOAD_CACHE_DIR, THUMB_CACHE_DIR, ELEVEN_CACHE_DIR, CLIP_PREVIEW_DIR):
    os.makedirs(_d, exist_ok=True)

DEFAULT_SETTINGS = {
    "eleven_model_id": "eleven_v4",
    "eleven_voice_id": "pNInz6obpgDQGcFmaJgB",
    "eleven_speed": 1.00,
    "eleven_stability": 45,
    "eleven_similarity": 80,
    "eleven_style": 15,
    "target_wps": 2.50,
    "slot_fill_pct": 100,
    "voice_lead_ms": 0,
    "slot_align_mode": "start",
    "auto_inject_v4_tags": True,
    "keep_v4_emotive_tags_in_api": True,
    "auto_fit_voice_slot": True,
    "graphics_preset": "stream_copy",
    "sharpness_boost": 1.25,
    "voice_volume": 1.0,
    "keep_original_audio": True,
    "original_audio_volume": 1.0,
    "bgm_volume": 0.15,
    "bgm_spacing_mins": 1,
    "bgm_spacing_secs": 15,
    "enable_audio_ducking": True,
    "output_filename": "output/video_thi_cong_hoan_chinh.mp4"
}


def natural_sort_key(file_obj_or_path):
    """Sắp xếp thứ tự các tập video tự nhiên (Tập 1, Tập 2, ..., Tập 10)."""
    name = file_obj_or_path.name if hasattr(file_obj_or_path, "name") else os.path.basename(str(file_obj_or_path))
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def load_settings() -> dict:
    """Đọc cấu hình mặc định, không nạp API Key từ file chung để bảo mật trên Web."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                merged = DEFAULT_SETTINGS.copy()
                data = json.load(f)
                data.pop("gemini_api_key", None)
                data.pop("elevenlabs_api_key", None)
                merged.update(data)
                return merged
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()


def save_settings(data: dict):
    """Lưu thông số cấu hình nhưng loại bỏ API Key để không bị lộ cho người khác."""
    try:
        safe_data = dict(data)
        safe_data.pop("gemini_api_key", None)
        safe_data.pop("elevenlabs_api_key", None)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(safe_data, f, indent=4, ensure_ascii=False)
    except Exception:
        pass


def flatten_files(files):
    """Làm phẳng danh sách file được tải lên từ Streamlit uploader."""
    if not files:
        return []
    if not isinstance(files, list):
        return [files]
    flat = []
    for item in files:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return [f for f in flat if hasattr(f, "name")]


def resolve_or_cache_media_file(file_obj, prefix_idx: Optional[int] = None) -> Tuple[str, int]:
    """Lưu file tải lên vào bộ nhớ đệm cục bộ để FFmpeg xử lý."""
    if getattr(file_obj, "is_local_disk", False) and os.path.exists(file_obj.filepath):
        return file_obj.filepath, file_obj.size
    f_size = getattr(file_obj, "size", 0)
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", file_obj.name)
    tag = f"{prefix_idx:05d}_" if prefix_idx is not None else ""
    cached_path = os.path.join(UPLOAD_CACHE_DIR, f"{tag}{f_size}_{safe_name}")
    if os.path.exists(cached_path) and (f_size == 0 or os.path.getsize(cached_path) == f_size):
        return cached_path, f_size
    total_written = 0
    file_obj.seek(0)
    with open(cached_path, "wb") as f:
        while True:
            chunk = file_obj.read(16 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            total_written += len(chunk)
    return cached_path, total_written


def generate_silent_wav(filepath: str, duration_sec: float, sample_rate: int = 44100):
    """Tạo đoạn âm thanh im lặng dùng để giãn cách giữa các bài nhạc nền BGM."""
    if duration_sec <= 0:
        return None
    num_samples = int(sample_rate * duration_sec)
    with wave.open(filepath, "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        chunk_size = 44100
        silent_chunk = struct.pack("<" + ("h" * chunk_size * 2), *([0] * chunk_size * 2))
        for _ in range(num_samples // chunk_size):
            wav_file.writeframes(silent_chunk)
        rem = num_samples % chunk_size
        if rem > 0:
            wav_file.writeframes(struct.pack("<" + ("h" * rem * 2), *([0] * rem * 2)))
    return filepath


def build_spaced_bgm_files(bgm_paths: list, spacing_sec: float, work_dir: str) -> list:
    """Ghép các bài nhạc nền BGM kèm khoảng nghỉ im lặng ở giữa."""
    if not bgm_paths:
        return []
    if spacing_sec <= 0 and len(bgm_paths) == 1:
        return bgm_paths
    spaced_list = []
    silence_path = os.path.join(work_dir, f"silence_gap_{int(spacing_sec)}s.wav")
    if spacing_sec > 0:
        generate_silent_wav(silence_path, spacing_sec)
    for i, path in enumerate(bgm_paths):
        spaced_list.append(path)
        if spacing_sec > 0 and i < len(bgm_paths) - 1 and os.path.exists(silence_path):
            spaced_list.append(silence_path)
    merged_bgm_path = os.path.join(work_dir, "master_bgm_with_spacing.mp3")
    try:
        filter_inputs = []
        filter_concat = ""
        for idx, p in enumerate(spaced_list):
            filter_inputs.extend(["-i", p])
            filter_concat += f"[{idx}:a]"
        filter_concat += f"concat=n={len(spaced_list)}:v=0:a=1[outa]"
        cmd = ["ffmpeg", "-y"] + filter_inputs + ["-filter_complex", filter_concat, "-map", "[outa]", "-b:a", "192k", merged_bgm_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        if os.path.exists(merged_bgm_path) and os.path.getsize(merged_bgm_path) > 1024:
            return [merged_bgm_path]
    except Exception:
        pass
    return spaced_list


def clean_temp_cache() -> Tuple[int, float]:
    """Dọn dẹp thư mục tạm nhưng giữ nguyên 100% kho Voice ElevenLabs đã tạo."""
    freed_b, cnt = 0, 0
    for folder in (UPLOAD_CACHE_DIR, THUMB_CACHE_DIR, CLIP_PREVIEW_DIR):
        if os.path.exists(folder):
            try:
                for r_d, _, fs in os.walk(folder):
                    for f in fs:
                        try:
                            freed_b += os.path.getsize(os.path.join(r_d, f))
                        except Exception:
                            pass
                shutil.rmtree(folder, ignore_errors=True)
                cnt += 1
            except Exception:
                pass
    for _d in (UPLOAD_CACHE_DIR, THUMB_CACHE_DIR, ELEVEN_CACHE_DIR, CLIP_PREVIEW_DIR):
        os.makedirs(_d, exist_ok=True)
    return cnt, round(freed_b / (1024 * 1024), 2)


# ==============================================================================
# KHỞI TẠO SESSION STATE & GIAO DIỆN CHÍNH
# ==============================================================================
saved_cfg = load_settings()

for _k, _v in [
    ("session_gemini_key", ""),
    ("session_eleven_key", ""),
    ("output_filepath", saved_cfg.get("output_filename", "output/video_thi_cong_hoan_chinh.mp4")),
    ("last_system_error", ""),
    ("trigger_dub_only_render", False),
    ("last_rendered_output", ""),
    ("last_audit_report", None),
    ("last_elapsed_sec", 0.0),
    ("eleven_voices_list", DEFAULT_ELEVEN_VOICES.copy()),
    ("eleven_models_map", DEFAULT_ELEVEN_MODELS.copy()),
    ("eleven_sub_info", {"tier": "PRO", "char_used": 0, "char_limit": 0, "connected": False, "models_synced": 6}),
    ("eleven_script_rows", []),
    ("eleven_raw_text_buffer", "")
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

st.set_page_config(
    page_title="ElevenLabs AI Studio V2.1 Pro - Eleven v4, Filter & Master Dubbing",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-stale="true"], .stApp [data-stale="true"], div[data-testid="stStatusWidget"] { opacity: 1 !important; filter: none !important; pointer-events: auto !important; }
.stApp {
    background-color: #040918;
    background-image: radial-gradient(at 15% 15%, rgba(99, 102, 241, 0.16) 0px, transparent 45%),
                      radial-gradient(at 85% 20%, rgba(30, 58, 138, 0.25) 0px, transparent 50%),
                      linear-gradient(135deg, #040918 0%, #0a132c 50%, #050d1e 100%);
    font-family: 'Segoe UI', system-ui, sans-serif !important; color: #f8fafc !important;
}
[data-testid="stSidebar"] { background: #060b19 !important; border-right: 1px solid rgba(99, 102, 241, 0.35) !important; }
h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-weight: 800 !important; }
p, label, li, span { color: #f1f5f9 !important; }
.studio-banner {
    background: linear-gradient(125deg, #0f172a 0%, #312e81 52%, #0891b2 100%);
    padding: 16px 22px; border-radius: 14px; color: #ffffff;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.25); margin-bottom: 14px; border: 1px solid rgba(129, 140, 248, 0.5);
}
.studio-banner h1 { font-size: 22px !important; font-weight: 900 !important; margin: 0 0 5px 0 !important; color: #facc15 !important; }
.badge-pill {
    display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 700;
    background: #090e1c; color: #818cf8 !important; margin-right: 6px; margin-top: 5px; border: 1px solid rgba(129, 140, 248, 0.45);
}
.cyber-card {
    background: rgba(15, 23, 42, 0.92); border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 14px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
}
input, textarea, .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"], .stNumberInput input {
    background-color: #0f172a !important; color: #ffffff !important; font-weight: 700 !important;
    border: 1px solid #6366f1 !important; border-radius: 10px !important;
}
[data-testid="stFileUploadDropzone"] { background: #0f172a !important; border: 2px dashed #6366f1 !important; border-radius: 12px !important; }
button:not([kind="primary"]), div.stButton > button:not([kind="primary"]), div.stDownloadButton > button {
    background: #1e293b !important; color: #ffffff !important; border: 1px solid #6366f1 !important; font-weight: 800 !important; border-radius: 10px !important;
}
button:not([kind="primary"]):hover, div.stButton > button:not([kind="primary"]):hover {
    background: #312e81 !important; border-color: #facc15 !important;
}
div.stButton > button[kind="primary"], button[kind="primary"], div.stDownloadButton > button[kind="primary"] {
    background: linear-gradient(90deg, #4f46e5 0%, #ea580c 55%, #d97706 100%) !important; color: #ffffff !important;
    font-weight: 900 !important; font-size: 15.5px !important; padding: 11px 16px !important; border-radius: 11px !important; border: 1px solid #facc15 !important;
}
[data-testid="stExpander"] summary { background-color: #1e293b !important; color: #ffffff !important; border-radius: 10px !important; padding: 9px 14px !important; border: 1px solid #6366f1 !important; }
</style>
<div class="studio-banner">
    <h1>🎙️ ELEVENLABS STUDIO PRO V2.1 — ELEVEN v4, EMOTIVE TAGS & MASTER DUBBING</h1>
    <p>Hỗ Trợ Eleven v4 & Eleven v4 Turbo • Thanh Chèn Thẻ Cảm Xúc 1-Click • Bộ Lọc Giọng Thông Minh • Khóa Khớp Mốc Kịch Bản 100% • Giữ Nguyên Tiếng ASMR Gốc</p>
    <div>
        <span class="badge-pill">🔥 Eleven v4 + Emotive Audio Tags</span>
        <span class="badge-pill">🔍 Bộ Lọc Giọng Nam/Nữ/Clone/Cảm Xúc</span>
        <span class="badge-pill">🎙️ Khóa Cứng Khung Giờ Kịch Bản 100%</span>
        <span class="badge-pill">📥 Tải Video .MP4 Trực Tiếp Về Máy</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR: KẾT NỐI API RIÊNG BIỆT & BÀN HÒA ÂM ASMR
# ==============================================================================
with st.sidebar:
    with st.expander("🔑 1. Kết Nối API (Đồng Bộ Eleven v4)", expanded=True):
        st.caption("🔒 API Key chỉ lưu trên trình duyệt của bạn, người khác mở link sẽ không thấy.")
        gemini_api_key = st.text_input(
            "Gemini API Key (AI Quét Ảnh & Viết Kịch Bản)",
            value=st.session_state["session_gemini_key"],
            type="password"
        )
        st.session_state["session_gemini_key"] = gemini_api_key.strip()

        elevenlabs_api_key = st.text_input(
            "🎙️ ElevenLabs API Key (sk_...)",
            value=st.session_state["session_eleven_key"],
            type="password"
        )
        st.session_state["session_eleven_key"] = clean_eleven_api_key(elevenlabs_api_key)

        el_c1, el_c2 = st.columns(2)
        with el_c1:
            if st.button("🔄 Kết nối API", use_container_width=True):
                conn = connect_elevenlabs_api(st.session_state["session_eleven_key"])
                if conn.get("ok"):
                    st.session_state["eleven_voices_list"] = conn["voices"]
                    if conn.get("models"):
                        st.session_state["eleven_models_map"] = conn["models"]
                    st.session_state["eleven_sub_info"] = {
                        "tier": conn["tier"],
                        "char_used": conn["char_used"],
                        "char_limit": conn["char_limit"],
                        "connected": True,
                        "models_synced": len(conn.get("models", DEFAULT_ELEVEN_MODELS))
                    }
                    st.toast(f"✅ Đã tải {len(conn['voices'])} giọng đọc & đồng bộ Eleven v4 ({conn['tier']})!")
                    st.rerun()
                else:
                    st.error(conn.get("error", "Lỗi kết nối"))
        with el_c2:
            sub_i = st.session_state.get("eleven_sub_info", {})
            st.caption(
                f"🌟 Gói: **{sub_i.get('tier', 'PRO')}**\n\n"
                f"🔤 Đã dùng: **{sub_i.get('char_used', 0):,}** / {sub_i.get('char_limit', 0):,}"
            )

    with st.expander("🔊 2. Bàn Hòa Âm ASMR & Nhạc Nền BGM", expanded=True):
        voice_volume = st.slider("🎙️ Âm lượng Giọng đọc (Voice)", 0.2, 2.5, float(saved_cfg.get("voice_volume", 1.0)), 0.05)
        keep_original_audio = st.checkbox("🛠️ Giữ tiếng máy móc công trình gốc (ASMR)", value=bool(saved_cfg.get("keep_original_audio", True)))
        original_audio_volume = st.slider(
            "🔊 Âm lượng ASMR gốc (1.0 = 100% nguyên bản)",
            0.1, 2.0, float(saved_cfg.get("original_audio_volume", 1.0)), 0.05,
            disabled=(not keep_original_audio)
        )
        enable_audio_ducking = st.checkbox("🎚️ Tự hạ tiếng nền khi có Voice (Smart Ducking)", value=bool(saved_cfg.get("enable_audio_ducking", True)))
        bgm_volume = st.slider("🎵 Âm lượng Nhạc nền (BGM)", 0.01, 1.5, float(saved_cfg.get("bgm_volume", 0.15)), 0.02)
        bgm_spacing_mins = st.number_input("Phút giãn cách giữa 2 bài nhạc:", 0, 60, int(saved_cfg.get("bgm_spacing_mins", 1)), 1)
        bgm_spacing_secs = st.slider("Cộng thêm giây giãn cách:", 0, 59, int(saved_cfg.get("bgm_spacing_secs", 15)), 1)
        total_bgm_spacing_sec = (bgm_spacing_mins * 60) + bgm_spacing_secs

    with st.expander("🧹 3. Dọn File Tạm Trên Máy Chủ", expanded=False):
        if st.button("🧹 Dọn Rác Tạm (Giữ Nguyên Kho Voice)", use_container_width=True):
            cnt, mb = clean_temp_cache()
            st.toast(f"🧹 Đã giải phóng {mb} MB (Giữ nguyên 100% kho Voice ElevenLabs)!")

# ==============================================================================
# MỤC 1: NẠP VIDEO ĐÃ CẮT SẴN & CHỌN CHẾ ĐỘ ĐỒ HỌA XUẤT
# ==============================================================================
with st.expander("📥 MỤC 1: TẢI LÊN VIDEO ĐÃ CẮT SẴN, FILE KỊCH BẢN & CHỌN CHẤT LƯỢNG XUẤT", expanded=True):
    col_in, col_out = st.columns([1.15, 0.85])

    with col_in:
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.subheader("📥 1. Kho Tư Liệu Đầu Vào")
        uploaded_videos = st.file_uploader(
            "📹 Kéo thả 1 hoặc nhiều tập Video đã cắt sẵn (.mp4, .mov, .avi, .mkv)",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            accept_multiple_files=True
        )
        uc1, uc2 = st.columns(2)
        with uc1:
            uploaded_audios = st.file_uploader(
                "📝 File Kịch bản (.srt, .txt, .md) hoặc File Voice lẻ (.mp3, .wav)",
                type=["srt", "txt", "md", "mp3", "wav", "m4a"],
                accept_multiple_files=True
            )
        with uc2:
            uploaded_bgm = st.file_uploader(
                "🎵 Nhạc nền BGM (.mp3, .wav, .m4a - Không bắt buộc)",
                type=["mp3", "wav", "m4a"],
                accept_multiple_files=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_out:
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.subheader("📤 2. Chế Độ Đồ Họa & Đặt Tên Video Xuất Ra")
        gfx_preset_map = {
            "stream_copy": "⚡ Giữ Nguyên Bản Gốc 100% (FFmpeg Stream Copy - Xuất Siêu Tốc, Không Nén Lại)",
            "1080p_ultra_sharp": "💎 Đồ Họa 1080p Full HD Siêu Nét (Lanczos + Làm Nét Unsharp Mask + CRF 16)",
            "2k_qhd_master": "🔥 Đồ Họa 2K QHD Master Studio (2560x1440 Siêu Căng Nét + CRF 15)",
            "4k_uhd_cinema": "👑 Đồ Họa 4K UHD Điện Ảnh Cực Đại (3840x2160 Ultra-Sharp + CRF 14)"
        }
        gfx_keys = list(gfx_preset_map.keys())
        saved_gfx = saved_cfg.get("graphics_preset", "stream_copy")
        graphics_preset = st.selectbox(
            "🎨 Lựa Chọn Gói Đồ Họa Khi Xuất Video:",
            options=gfx_keys,
            format_func=lambda k: gfx_preset_map[k],
            index=gfx_keys.index(saved_gfx) if saved_gfx in gfx_keys else 0
        )
        fast_stream_copy_dub = (graphics_preset == "stream_copy")

        sharpness_boost = st.slider(
            "💎 Cường độ Làm Nét Sâu (Chỉ áp dụng khi chọn gói 1080p/2K/4K):",
            0.0, 2.5, float(saved_cfg.get("sharpness_boost", 1.25)), 0.05,
            disabled=fast_stream_copy_dub
        )

        if st.button("⚡ Đặt tên file tự động theo ngày giờ", use_container_width=True):
            st.session_state["output_filepath"] = f"output/Video_Studio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            st.rerun()

        output_filename = st.text_input("Tên file video xuất ra:", value=st.session_state["output_filepath"])
        st.session_state["output_filepath"] = output_filename
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# MỤC 2: TRẠM ELEVENLABS STUDIO V2.1 PRO (ELEVEN V4 + LỌC GIỌNG + CLONE)
# ==============================================================================
with st.expander("🎙️ MỤC 2: ELEVENLABS STUDIO PRO V2.1 — ELEVEN v4, EMOTIVE TAGS, BỘ LỌC GIỌNG & KHÓA MỐC GIỜ", expanded=True):
    eleven_cfg = render_elevenlabs_dubbing_station_ui(
        saved_cfg=saved_cfg,
        gemini_api_key=st.session_state["session_gemini_key"],
        elevenlabs_api_key=st.session_state["session_eleven_key"],
        uploaded_videos=uploaded_videos,
        uploaded_audios=uploaded_audios,
        resolve_or_cache_fn=resolve_or_cache_media_file,
        flatten_files_fn=flatten_files
    )

current_full_config = {
    **eleven_cfg,
    "graphics_preset": graphics_preset,
    "sharpness_boost": sharpness_boost,
    "voice_volume": voice_volume,
    "keep_original_audio": keep_original_audio,
    "original_audio_volume": original_audio_volume,
    "bgm_volume": bgm_volume,
    "bgm_spacing_mins": bgm_spacing_mins,
    "bgm_spacing_secs": bgm_spacing_secs,
    "enable_audio_ducking": enable_audio_ducking,
    "output_filename": output_filename
}

# ==============================================================================
# MỤC 3: KẾT XUẤT VIDEO HOÀN CHỈNH & TẢI VỀ MÁY
# ==============================================================================
with st.expander("🚀 MỤC 3: KẾT XUẤT VIDEO HOÀN CHỈNH & TẢI FILE (.MP4) VỀ MÁY", expanded=True):
    _has_vids_loaded = bool(uploaded_videos)
    ct1, ct2 = st.columns([0.25, 0.75])
    with ct1:
        if st.button("💾 Lưu Cấu Hình Mặc Định", use_container_width=True):
            save_settings(current_full_config)
            st.toast("✅ Đã lưu thông số cấu hình!")
    with ct2:
        start_clicked = st.button(
            "🚀 XÁC NHẬN GHÉP VOICE ELEVEN v4 & XUẤT VIDEO CHÍNH THỨC (GIỮ 100% ASMR GỐC)" if _has_vids_loaded else "🔒 HÃY TẢI VIDEO LÊN Ở MỤC 1 TRƯỚC KHI XUẤT",
            type="primary",
            use_container_width=True,
            disabled=(not _has_vids_loaded)
        )


def execute_dub_only_pipeline():
    """Thực thi luồng ghép Voice chuẩn mốc thời gian kịch bản vào video đã cắt sẵn."""
    v_files = flatten_files(uploaded_videos)
    if not v_files:
        st.error("⚠️ Vui lòng tải lên ít nhất 1 tập Video ở Mục 1 để bắt đầu!")
        return

    a_files = flatten_files(uploaded_audios)
    b_files = flatten_files(uploaded_bgm)
    save_settings(current_full_config)

    v_files = sorted(v_files, key=natural_sort_key)
    if a_files:
        a_files = sorted(a_files, key=natural_sort_key)

    work_dir = tempfile.mkdtemp(prefix="studio_pro_")
    percent_banner, progress_bar, status_box = st.empty(), st.progress(0.0), st.empty()
    start_wall_time = time.time()

    def set_live_progress(pct: float, stage_title: str, detail: str = ""):
        pct_c = max(0.01, min(1.0, pct))
        el_s = time.time() - start_wall_time
        progress_bar.progress(pct_c)
        percent_banner.markdown(
            f'<div class="cyber-card" style="border-left:6px solid #6366f1;padding:14px 20px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div><div style="font-size:18px;font-weight:900;color:#818cf8;">{stage_title}</div>'
            f'<div style="font-size:13.5px;color:#fff;margin-top:4px;">{detail} (⏱️ {el_s:.1f}s)</div></div>'
            f'<div style="font-size:32px;font-weight:900;color:#facc15;font-family:Consolas,monospace;">{pct_c*100:.1f}%</div>'
            f'</div></div>',
            unsafe_allow_html=True
        )

    try:
        set_live_progress(0.05, "⚡ ĐANG NẠP VIDEO CẮT SẴN", f"Đang xử lý {len(v_files)} tập video...")
        video_paths = [resolve_or_cache_media_file(vf, prefix_idx=idx + 1)[0] for idx, vf in enumerate(v_files)]
        raw_bgms = [resolve_or_cache_media_file(bf)[0] for bf in b_files]
        bgm_paths = build_spaced_bgm_files(raw_bgms, float(total_bgm_spacing_sec), work_dir) if raw_bgms else []

        editor = AutoVideoEditor(
            sharpness_boost=sharpness_boost,
            enable_audio_ducking=enable_audio_ducking,
            cpu_threads=MAX_CPU
        )

        timed_items: List[Dict[str, Any]] = []
        master_aud_p: Optional[str] = None
        lead_sec = float(eleven_cfg.get("voice_lead_ms", 0)) / 1000.0
        align_mode = str(eleven_cfg.get("slot_align_mode", "start"))
        auto_fit_slot = bool(eleven_cfg.get("auto_fit_voice_slot", True))
        keep_v4_tags = bool(eleven_cfg.get("keep_v4_emotive_tags_in_api", True))
        active_el_key = st.session_state["session_eleven_key"]

        def _compute_aligned_window(s_raw: float, e_raw: float, real_voice_dur: float) -> Tuple[float, float]:
            slot_span = max(0.3, e_raw - s_raw)
            s_adj = max(0.0, s_raw + lead_sec)
            if real_voice_dur > 0 and real_voice_dur < slot_span:
                if align_mode == "center":
                    s_adj += (slot_span - real_voice_dur) / 2.0
                elif align_mode == "end":
                    s_adj += max(0.0, slot_span - real_voice_dur - 0.1)
            return round(s_adj, 3), round(max(s_adj + 0.3, e_raw + lead_sec), 3)

        el_rows = st.session_state.get("eleven_script_rows", [])
        if el_rows:
            for ir, r in enumerate(el_rows):
                row_voice_id = r.get("row_voice_id") or eleven_cfg["eleven_voice_id"]
                row_speed = float(r.get("row_speed") or eleven_cfg["eleven_speed"])
                if (not r.get("audio_path") or not os.path.exists(str(r.get("audio_path", "")))) and active_el_key:
                    set_live_progress(
                        0.10 + (ir / max(1, len(el_rows))) * 0.25,
                        "🎙️ TỰ ĐỘNG TẠO CÁC CÂU VOICE ELEVEN v4 CÒN THIẾU",
                        f"Đang tạo câu #{r['STT']}/{len(el_rows)}..."
                    )
                    rt = generate_elevenlabs_tts_segment(
                        active_el_key,
                        r["Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)"],
                        row_voice_id,
                        eleven_cfg["eleven_model_id"],
                        row_speed,
                        eleven_cfg["eleven_stability"],
                        eleven_cfg["eleven_similarity"],
                        eleven_cfg["eleven_style"],
                        force_regenerate=False,
                        strip_tags_for_credit=(not keep_v4_tags)
                    )
                    if rt.get("ok"):
                        r["audio_path"] = rt["audio_path"]
                        r["Voice Thực Tế (s)"] = rt["real_duration"]

                if r.get("audio_path") and os.path.exists(str(r["audio_path"])):
                    s_raw = parse_timestamp_srt(str(r["Mốc Bắt Đầu"]))
                    e_raw = parse_timestamp_srt(str(r["Mốc Kết Thúc"]))
                    r_dur = float(r.get("Voice Thực Tế (s)", 0.0))
                    s_al, e_al = _compute_aligned_window(s_raw, e_raw, r_dur)
                    timed_items.append({
                        "start": s_al,
                        "end": e_al,
                        "audio_path": r["audio_path"],
                        "text": r.get("Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)", "")
                    })

        if not timed_items and a_files:
            a_paths = [resolve_or_cache_media_file(af)[0] for af in a_files]
            s_fs = [p for p in a_paths if p.lower().endswith(('.md', '.srt', '.txt'))]
            v_fs = [p for p in a_paths if not p.lower().endswith(('.md', '.srt', '.txt'))]
            ap = AudioProcessor()
            if s_fs and v_fs:
                segs = ap.parse_script_file(s_fs[0]).get("segments", [])
                if len(v_fs) > 1 and len(segs) >= len(v_fs):
                    for i_v, vp_a in enumerate(v_fs):
                        s_al, e_al = _compute_aligned_window(float(segs[i_v]["start"]), float(segs[i_v]["end"]), 0.0)
                        timed_items.append({"start": s_al, "end": e_al, "audio_path": vp_a, "text": segs[i_v].get("text", "")})
                else:
                    master_aud_p = v_fs[0]
            elif len(v_fs) >= 1:
                master_aud_p = v_fs[0]

        rep = editor.render_dub_only_video(
            video_paths=video_paths,
            output_path=output_filename,
            work_dir=work_dir,
            timed_voice_items=timed_items,
            master_audio_path=master_aud_p,
            bgm_paths=bgm_paths,
            bgm_volume=bgm_volume,
            voice_volume=voice_volume,
            keep_original_audio=keep_original_audio,
            original_audio_volume=original_audio_volume,
            auto_fit_voice_slot=auto_fit_slot,
            fast_stream_copy=fast_stream_copy_dub,
            enhance_video_sharpness=(graphics_preset != "stream_copy"),
            sharpness_amount=sharpness_boost,
            graphics_preset=graphics_preset,
            progress_callback=lambda r, msg: set_live_progress(0.35 + r * 0.65, "🎙️ ĐANG HÒA ÂM VOICE + ASMR GỐC & ĐÓNG GÓI MP4", msg)
        )
        tot_el = round(time.time() - start_wall_time, 2)
        set_live_progress(1.0, f"🎉 HOÀN TẤT GHÉP VOICE 100% ({tot_el}s)!", f"Đã xuất: {output_filename}")
        status_box.success(f"✅ Đã ghép {rep.get('total_clips_assembled', 1)} câu Voice chuẩn khung giờ vào **{rep.get('video_duration_sec', 0)} giây** Video!")
        st.session_state["last_rendered_output"] = output_filename
        st.session_state["last_audit_report"] = rep
        st.session_state["last_elapsed_sec"] = tot_el
    except Exception as e:
        st.session_state["last_system_error"] = str(e)
        st.error(f"❌ Đã xảy ra lỗi khi xuất video: {e}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if start_clicked or st.session_state.get("trigger_dub_only_render", False):
    st.session_state["trigger_dub_only_render"] = False
    execute_dub_only_pipeline()

if st.session_state.get("last_rendered_output") and os.path.exists(st.session_state["last_rendered_output"]):
    out_done = st.session_state["last_rendered_output"]
    rep_d = st.session_state.get("last_audit_report") or {}
    st.markdown("---")
    st.subheader("🎬 KẾT QUẢ VIDEO HOÀN CHỈNH (XEM TRỰC TIẾP & TẢI VỀ MÁY)")

    m1, m2, m3 = st.columns(3)
    m1.metric("Thời gian xử lý", f"{st.session_state.get('last_elapsed_sec', 0.0)} giây")
    m2.metric("Tổng thời lượng Video", f"{rep_d.get('video_duration_sec', 0)} giây")
    m3.metric("Số câu Voice đã ghép", f"{rep_d.get('total_clips_assembled', 1)} câu")

    with open(out_done, "rb") as f_mp4:
        st.download_button(
            label=f"📥 BẤM ĐỂ TẢI VIDEO HOÀN CHỈNH ({os.path.basename(out_done)}) VỀ MÁY TÍNH / ĐIỆN THOẠI",
            data=f_mp4,
            file_name=os.path.basename(out_done),
            mime="video/mp4",
            type="primary",
            use_container_width=True
        )

    st.video(out_done)