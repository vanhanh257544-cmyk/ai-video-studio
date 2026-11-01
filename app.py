import os
import re
import json
import time
import wave
import struct
import shutil
import inspect
import tempfile
import subprocess
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, date
try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None
    filedialog = None
import streamlit as st  # type: ignore

from audio_processor import AudioProcessor  # type: ignore
from vision_analyzer import VisionAnalyzer  # type: ignore
from video_editor import AutoVideoEditor  # type: ignore
from eleven_dub_engine import (  # type: ignore
    DEFAULT_ELEVEN_VOICES,
    clean_eleven_api_key,
    connect_elevenlabs_api,
    generate_elevenlabs_tts_segment,
    render_auto_confirm_countdown_widget,
    render_elevenlabs_dubbing_station_ui
)

try:
    import psutil  # type: ignore
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

CONFIG_FILE = "settings.json"
QUOTA_FILE = "gemini_quota_tracker.json"
CACHE_DIR = os.path.join(os.getcwd(), "studio_cache")
UPLOAD_CACHE_DIR = os.path.join(CACHE_DIR, "uploads")
THUMB_CACHE_DIR = os.path.join(CACHE_DIR, "thumbnails")
ELEVEN_CACHE_DIR = os.path.join(CACHE_DIR, "elevenlabs_audio")
MAX_CPU = os.cpu_count() or 8

for _d in (UPLOAD_CACHE_DIR, THUMB_CACHE_DIR, ELEVEN_CACHE_DIR):
    os.makedirs(_d, exist_ok=True)


def ensure_streamlit_server_config():
    st_dir = os.path.join(os.getcwd(), ".streamlit")
    st_cfg_path = os.path.join(st_dir, "config.toml")
    try:
        os.makedirs(st_dir, exist_ok=True)
        with open(st_cfg_path, "w", encoding="utf-8") as f:
            f.write(
                "[server]\nmaxUploadSize = 102400\nmaxMessageSize = 4096\n"
                "enableXsrfProtection = false\nenableCORS = false\n"
                "enableWebsocketCompression = false\n\n[runner]\nfastReruns = true\n"
            )
    except Exception:
        pass


ensure_streamlit_server_config()

DEFAULT_SETTINGS = {
    "gemini_api_key": "",
    "elevenlabs_api_key": "",
    "eleven_model_id": "eleven_v4",
    "eleven_voice_id": "pNInz6obpgDQGcFmaJgB",
    "eleven_speed": 1.00,
    "eleven_stability": 45,
    "eleven_similarity": 80,
    "eleven_style": 15,
    "target_wps": 2.50,
    "min_words_limit": 10,
    "max_words_limit": 80,
    "slot_fill_pct": 100,
    "voice_lead_ms": 0,
    "breath_gap_sec": 0.0,
    "punct_pause_ms": 0,
    "slot_align_mode": "start",
    "overrun_policy": "ai_rewrite_and_atempo",
    "underrun_policy": "ai_expand_action",
    "max_atempo_limit": 1.35,
    "ai_rewrite_style": "documentary_action",
    "auto_inject_v4_tags": True,
    "auto_fit_voice_slot": True,
    "fast_stream_copy_dub": True,
    "graphics_preset": "stream_copy",
    "sharpness_boost": 1.25,
    "auto_confirm_3min": True,
    "auto_confirm_seconds": 180,
    "use_ai_vision": True,
    "whisper_model_size": "base",
    "edit_mode": "dub_only",
    "aspect_ratio": "16:9 Full HD (1920x1080 - YouTube/TV)",
    "fit_mode": "crop",
    "target_fps": 60,
    "video_codec": "libx264",
    "render_preset": "superfast",
    "crf_quality": 17,
    "cpu_threads": MAX_CPU,
    "color_style": "none",
    "lut_opacity": 0.06,
    "brightness": 0.0,
    "contrast": 1.0,
    "saturation": 1.05,
    "enable_vignette": False,
    "enable_letterbox": False,
    "letterbox_ratio": 0.065,
    "enable_crossfade": False,
    "crossfade_duration": 0.2,
    "enable_fade_inout": True,
    "watermark_text": "",
    "watermark_pos": "bottom_right",
    "zoom_factor": 1.10,
    "default_speed": 3.0,
    "min_speed": 0.5,
    "max_speed": 25.0,
    "flip_prob": 0.15,
    "color_segments": 6,
    "enable_shuffle": True,
    "auto_clip_duration": 3.0,
    "auto_total_duration": 0,
    "target_duration_mins": 0,
    "voice_volume": 1.0,
    "keep_original_audio": True,
    "original_audio_volume": 0.30,
    "bgm_volume": 0.15,
    "bgm_spacing_mins": 1,
    "bgm_spacing_secs": 15,
    "enable_audio_ducking": True,
    "use_smart_cache": True,
    "strict_sequential_video": True,
    "require_preview_confirm": True,
    "auto_heal_enabled": True,
    "daily_quota_limit": 1500,
    "output_filename": "output/video_thi_cong_hoan_chinh.mp4"
}


class LocalDiskFileProxy:
    def __init__(self, filepath: str):
        self.filepath = os.path.abspath(filepath)
        self.name = os.path.basename(filepath)
        try:
            self.size = os.path.getsize(self.filepath)
        except Exception:
            self.size = 0
        self.is_local_disk = True


def natural_sort_key(file_obj_or_path):
    name = file_obj_or_path.name if hasattr(file_obj_or_path, "name") else os.path.basename(str(file_obj_or_path))
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def pick_local_files_dialog(title: str, filetypes: list, multiple: bool = True) -> List[str]:
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        selected = filedialog.askopenfilenames(title=title, filetypes=filetypes) if multiple else filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        if not selected:
            return []
        return list(selected) if multiple else [selected]
    except Exception:
        return []


def pick_local_folder_videos_dialog(title: str = "Chọn thư mục chứa các tập Video") -> List[str]:
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        folder = filedialog.askdirectory(title=title)
        root.destroy()
        if not folder or not os.path.isdir(folder):
            return []
        exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts) and os.path.isfile(os.path.join(folder, f))]
        return sorted(files, key=natural_sort_key)
    except Exception:
        return []


def open_file_dialog(current_path: str) -> str:
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        init_dir = os.path.dirname(os.path.abspath(current_path)) if current_path else os.getcwd()
        init_file = os.path.basename(current_path) if current_path else f"Video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        selected = filedialog.asksaveasfilename(
            title="Chọn thư mục và đặt tên file video xuất ra",
            initialdir=init_dir, initialfile=init_file,
            defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4"), ("All Files", "*.*")]
        )
        root.destroy()
        return selected
    except Exception:
        return current_path


def load_settings() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                merged = DEFAULT_SETTINGS.copy()
                merged.update(json.load(f))
                return merged
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()


def save_settings(data: dict):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        st.sidebar.error(f"Lỗi lưu cài đặt: {e}")


def load_quota_tracker() -> dict:
    today_str = str(date.today())
    default_q = {"date": today_str, "requests_used": 0, "tokens_estimated": 0, "last_error": ""}
    if os.path.exists(QUOTA_FILE):
        try:
            with open(QUOTA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("date") == today_str:
                    return data
        except Exception:
            pass
    return default_q


def update_quota_tracker(add_requests: int = 0, add_tokens: int = 0, error_msg: Optional[str] = None) -> dict:
    q = load_quota_tracker()
    q["requests_used"] = max(0, q.get("requests_used", 0) + add_requests)
    q["tokens_estimated"] = max(0, q.get("tokens_estimated", 0) + add_tokens)
    if error_msg is not None:
        q["last_error"] = error_msg
    try:
        with open(QUOTA_FILE, "w", encoding="utf-8") as f:
            json.dump(q, f, indent=4, ensure_ascii=False)
    except Exception:
        pass
    return q


def flatten_files(files):
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


def format_timestamp_srt(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000))
    total_sec = int(seconds)
    return f"{total_sec // 3600:02d}:{(total_sec % 3600) // 60:02d}:{total_sec % 60:02d},{ms:03d}"


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


def resolve_or_cache_media_file(file_obj, prefix_idx: Optional[int] = None) -> Tuple[str, int]:
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


def export_edl_to_srt_string(edl: list) -> str:
    lines = []
    for idx, item in enumerate(edl, 1):
        v_seg = item.get("voice_segment", {})
        lines.append(
            f"{idx}\n{format_timestamp_srt(float(v_seg.get('start', 0.0)))} --> "
            f"{format_timestamp_srt(float(v_seg.get('end', 0.0)))}\n{v_seg.get('text', f'Phân cảnh {idx}')}\n"
        )
    return "\n".join(lines)


def export_edl_to_fcpxml_string(edl: list, fps: int = 60) -> str:
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="4">\n  <sequence>',
        f'    <name>AI_Video_Studio_Pro_Max</name>\n    <rate><timebase>{fps}</timebase><ntsc>FALSE</ntsc></rate>',
        '    <media><video><track>'
    ]
    for idx, item in enumerate(edl, 1):
        v_seg = item.get("voice_segment", {})
        sc = item.get("scene", {})
        ep_name = sc.get("episode_name", os.path.basename(sc.get("video_path", f"clip_{idx}.mp4")))
        xml_lines.append(
            f'      <clipitem id="clip-{idx}"><name>{ep_name}</name>'
            f'<start>{int(float(v_seg.get("start", 0))*fps)}</start><end>{int(float(v_seg.get("end", 0))*fps)}</end>'
            f'<in>{int(float(sc.get("start", 0))*fps)}</in><out>{int(float(sc.get("end", 5))*fps)}</out>'
            f'<file id="f-{idx}"><name>{ep_name}</name><pathurl>{sc.get("video_path", ep_name)}</pathurl></file></clipitem>'
        )
    xml_lines.append('    </track></video></media>\n  </sequence>\n</xmeml>')
    return "\n".join(xml_lines)


def build_visual_timeline_html(edl: list) -> str:
    if not edl:
        return ""
    total_dur = max(1.0, float(edl[-1].get("voice_segment", {}).get("end", 10.0)))
    blocks_html = []
    curr_t = 0.0
    for idx, item in enumerate(edl):
        v_seg = item.get("voice_segment", {})
        s_t = float(v_seg.get("start", 0.0))
        e_t = float(v_seg.get("end", s_t + 3.0))
        if s_t - curr_t > 0.6:
            gap_w = max(0.8, ((s_t - curr_t) / total_dur) * 100.0)
            blocks_html.append(f'<div title="B-Roll: {curr_t:.1f}s->{s_t:.1f}s" style="width:{gap_w:.2f}%;background:#065f46;height:100%;border-right:1px solid #020617;font-size:10px;color:#a7f3d0;display:flex;align-items:center;justify-content:center;overflow:hidden;">B-Roll</div>')
        seg_w = max(1.2, ((e_t - s_t) / total_dur) * 100.0)
        bg_col = "#0284c7" if idx % 2 == 0 else "#2563eb"
        blocks_html.append(f'<div title="Câu #{idx+1}: {s_t:.1f}s->{e_t:.1f}s" style="width:{seg_w:.2f}%;background:{bg_col};height:100%;border-right:1px solid #facc15;font-size:10.5px;font-weight:800;color:#fff;display:flex;align-items:center;justify-content:center;overflow:hidden;">#{idx+1}</div>')
        curr_t = e_t
    return (
        '<div style="margin:10px 0 14px 0;"><div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px;font-weight:700;">'
        '<span>🎞️ SƠ ĐỒ BĂNG CHUYỀN TIMELINE (PREMIERE / DAVINCI VIEW)</span><span>🟦 Đoạn có Voice/Kịch bản | 🟩 Khoảng nghỉ B-Roll (Tự nâng nhạc nền)</span></div>'
        f'<div style="width:100%;height:32px;background:#020617;border:2px solid #38bdf8;border-radius:8px;display:flex;overflow:hidden;">{"".join(blocks_html)}</div></div>'
    )


def generate_silent_wav(filepath: str, duration_sec: float, sample_rate: int = 44100):
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


def get_realtime_io_speed() -> str:
    now = time.time()
    prev_t = st.session_state.get("io_prev_time", now - 1.0)
    dt = max(0.05, now - prev_t)
    if HAS_PSUTIL and psutil is not None:
        try:
            net = psutil.net_io_counters()
            disk = psutil.disk_io_counters()
            total_cur = (net.bytes_sent + net.bytes_recv) + ((disk.read_bytes + disk.write_bytes) if disk else 0)
            speed = max(0.0, (total_cur - st.session_state.get("io_prev_bytes", total_cur)) / dt)
            st.session_state["io_prev_time"] = now
            st.session_state["io_prev_bytes"] = total_cur
            if speed > 0:
                return f"{speed / (1024 * 1024):.2f} MB/s" if speed >= 1048576 else f"{speed / 1024:.1f} KB/s"
        except Exception:
            pass
    proc_b = st.session_state.get("processed_bytes_total", 0)
    avg_s = max(1024 * 1024 * 12, proc_b / max(0.1, now - st.session_state.get("process_start_time", now - 1.0)))
    return f"{avg_s / (1024 * 1024):.2f} MB/s"


def analyze_text_visual_intent(text: str) -> dict:
    t_low = text.lower()
    closeup_kw = ["nail", "hammer", "chisel", "knot", "joint", "twine", "stake", "mortar", "clay", "seam", "plumb", "level", "hinge", "chain", "hook", "candle", "brush", "đóng đinh", "mối nối", "chi tiết", "cận cảnh", "bắt vít", "đục", "vữa"]
    wide_kw = ["woods", "forest", "sanctuary", "cabin", "roofline", "canopy", "hills", "terrace", "veranda", "porch", "facade", "walks into", "leaving behind", "toàn cảnh", "khu rừng", "ngôi nhà", "mái nhà", "tổng thể", "hoàn thiện"]
    motion_kw = ["drives", "pounds", "chops", "hammers", "rams", "saws", "strips", "hauling", "carving", "crushes", "dig", "axe", "shovel", "đập", "đào", "chặt", "cưa", "kéo"]
    pref_shot = "Wide shot" if any(k in t_low for k in wide_kw) else ("Close-up" if any(k in t_low for k in closeup_kw) else "Close-up")
    high_m = len(re.findall(r'\b[A-Z]{4,}\b', text)) > 0 or any(k in t_low for k in motion_kw)
    words = [w for w in re.findall(r'[a-zA-ZÀ-ỹ]{4,}', t_low) if w not in {"with", "from", "that", "this", "into", "over", "after", "while", "their"}]
    return {"preferred_shot": pref_shot, "target_motion": "high" if high_m else "medium", "keywords": set(words)}


def optimize_scene_subclip_window(scene: dict, needed_duration: float, default_spd: float, usage_count: int = 0) -> dict:
    sc_copy = dict(scene)
    s_start = float(sc_copy.get("start", sc_copy.get("start_time", 0.0)))
    s_end = float(sc_copy.get("end", sc_copy.get("end_time", s_start + float(sc_copy.get("duration", 5.0)))))
    avail_dur = max(0.5, s_end - s_start)
    raw_needed = max(1.2, needed_duration * max(1.0, min(default_spd, 4.0)))
    if avail_dur > raw_needed * 1.30:
        margin = avail_dur - raw_needed
        shift_ratio = 0.25 if usage_count == 0 else min(0.85, 0.20 + (usage_count * 0.35))
        new_start = s_start + (margin * shift_ratio)
        new_end = min(s_end, new_start + raw_needed)
        sc_copy.update({"start": round(new_start, 3), "end": round(new_end, 3), "start_time": round(new_start, 3), "end_time": round(new_end, 3), "duration": round(new_end - new_start, 3)})
    else:
        sc_copy.update({"start": round(s_start, 3), "end": round(s_end, 3), "start_time": round(s_start, 3), "end_time": round(s_end, 3)})
    return sc_copy


def smart_director_build_edl(
    mode: str, valid_scenes: list, voice_segments: Optional[list] = None,
    ref_scenes_dna: Optional[list] = None, strict_sequential: bool = True,
    target_total_duration: float = 0.0, default_clip_duration: float = 3.5, default_spd: float = 3.0
) -> list:
    if not valid_scenes:
        return []
    total_raw = len(valid_scenes)
    used_counts: Dict[str, int] = {}
    edl, slots = [], []

    if mode == "voice_sync" and voice_segments:
        total_v = max(1, len(voice_segments))
        for idx, seg in enumerate(voice_segments):
            intent = analyze_text_visual_intent(seg.get("text", ""))
            p_r = idx / max(1, total_v - 1) if total_v > 1 else 0.0
            w_st = int(seg.get("workflow_stage", 1 if p_r <= 0.15 else (4 if p_r >= 0.85 else (3 if p_r >= 0.5 else 2))))
            slots.append({
                "index": idx, "start": float(seg.get("start", 0.0)), "end": float(seg.get("end", 0.0)),
                "duration": max(0.5, float(seg.get("duration", 3.0))), "text": seg.get("text", f"Phân đoạn {idx + 1}"),
                "preferred_shot": seg.get("preferred_shot") or intent["preferred_shot"],
                "content_type": seg.get("content_type", "Thao tác"), "workflow_stage": w_st,
                "target_motion": intent["target_motion"], "keywords": intent["keywords"]
            })
    elif ref_scenes_dna:
        cur_t, r_idx = 0.0, 0
        ref_tot = sum(max(0.8, float(s.get("duration", 3.0))) for s in ref_scenes_dna)
        limit_t = float(target_total_duration) if target_total_duration > 0 else ref_tot
        while cur_t < limit_t and r_idx < max(len(ref_scenes_dna) * 20, 50000):
            ref_sc = ref_scenes_dna[r_idx % len(ref_scenes_dna)]
            dur = max(0.8, float(ref_sc.get("duration", 3.0)))
            if target_total_duration > 0 and cur_t + dur > limit_t:
                dur = limit_t - cur_t
            if dur <= 0.15:
                break
            slots.append({
                "index": r_idx, "start": round(cur_t, 3), "end": round(cur_t + dur, 3), "duration": round(dur, 3),
                "text": f"[Clone ADN Mẫu #{r_idx + 1} | {ref_sc.get('shot_type', 'Auto')}]",
                "preferred_shot": ref_sc.get("shot_type", "Close-up"), "content_type": ref_sc.get("content_type", "Thao tác"),
                "workflow_stage": int(ref_sc.get("workflow_stage", 2)),
                "target_motion": "high" if float(ref_sc.get("motion_score", 2.0)) > 3.5 else "medium", "keywords": set()
            })
            cur_t += dur
            r_idx += 1
            if target_total_duration <= 0 and r_idx >= len(ref_scenes_dna):
                break
    else:
        cur_t, idx = 0.0, 0
        limit_t = float(target_total_duration) if target_total_duration > 0 else (total_raw * default_clip_duration)
        while cur_t < limit_t and idx < max(total_raw * 10, 50000):
            dur = float(default_clip_duration)
            if target_total_duration > 0 and cur_t + dur > limit_t:
                dur = limit_t - cur_t
            if dur <= 0.15:
                break
            slots.append({
                "index": idx, "start": round(cur_t, 3), "end": round(cur_t + dur, 3), "duration": round(dur, 3),
                "text": f"Phân cảnh tự động #{idx + 1}", "preferred_shot": "Close-up" if (idx % 3 != 0) else "Wide shot",
                "content_type": "Thao tác", "workflow_stage": 2, "target_motion": "high" if (idx % 2 == 0) else "medium", "keywords": set()
            })
            cur_t += dur
            idx += 1

    total_slots = max(1, len(slots))
    last_selected_idx = 0

    for s_idx, slot in enumerate(slots):
        progress = s_idx / max(1, total_slots - 1) if total_slots > 1 else 0.0
        stage_num = int(round(1 + progress * 3))
        if strict_sequential:
            ideal_center = int(round(progress * (total_raw - 1)))
            win_rad = max(2, int(total_raw / max(4, total_slots)) + 1)
            win_s = max(last_selected_idx, max(0, ideal_center - win_rad))
            win_e = min(total_raw - 1, max(win_s + win_rad, ideal_center + win_rad))
            if win_s > win_e:
                win_s, win_e = max(0, ideal_center - win_rad), min(total_raw - 1, ideal_center + win_rad)
            candidate_indices = list(range(win_s, win_e + 1))
        else:
            candidate_indices = list(range(total_raw))

        best_idx = candidate_indices[0] if candidate_indices else 0
        best_score = -1e9
        for c_idx in candidate_indices:
            sc = valid_scenes[c_idx]
            sc_key = f"{sc.get('video_path', '')}_{sc.get('start', sc.get('start_time', 0))}"
            u_cnt = used_counts.get(sc_key, 0)
            score = 120.0 if u_cnt == 0 else -(u_cnt * 150.0)
            if strict_sequential:
                score -= abs(c_idx - (progress * (total_raw - 1))) * (18.0 / max(1.0, total_raw / 12.0))
                if c_idx >= last_selected_idx:
                    score += 25.0
            if str(slot.get("preferred_shot", "Close-up")).lower() in str(sc.get("shot_type", "")).lower():
                score += 40.0
            m_val = float(sc.get("motion_score", 2.0))
            score += min(20.0, float(sc.get("sharpness_score", 50.0)) / 15.0)
            score += min(30.0, m_val * 4.5) if slot.get("target_motion") == "high" else (18.0 - min(18.0, abs(m_val - 2.5) * 3.0))
            if float(sc.get("duration", 3.0)) >= min(8.0, slot["duration"] * 0.6):
                score += 25.0
            if slot.get("keywords"):
                sc_desc = f"{sc.get('description', '')} {sc.get('content_type', '')}".lower()
                score += sum(1 for kw in slot["keywords"] if kw in sc_desc) * 30.0
            if score > best_score:
                best_score, best_idx = score, c_idx

        chosen_raw = valid_scenes[best_idx]
        chosen_key = f"{chosen_raw.get('video_path', '')}_{chosen_raw.get('start', chosen_raw.get('start_time', 0))}"
        cur_u = used_counts.get(chosen_key, 0)
        used_counts[chosen_key] = cur_u + 1
        if strict_sequential and best_idx > last_selected_idx:
            last_selected_idx = min(best_idx, int(round(((s_idx + 1) / total_slots) * (total_raw - 1))))

        edl.append({
            "voice_segment": {
                "index": slot["index"], "start": slot["start"], "end": slot["end"], "duration": slot["duration"],
                "text": slot["text"], "preferred_shot": slot["preferred_shot"], "content_type": slot["content_type"], "workflow_stage": stage_num
            },
            "scene": optimize_scene_subclip_window(chosen_raw, slot["duration"], default_spd, cur_u),
            "assigned_stage": stage_num
        })
    return edl


def apply_user_table_edits_to_edl(edited_rows: List[Dict[str, Any]], original_edl: List[Dict[str, Any]], valid_scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not edited_rows or not original_edl:
        return original_edl
    ep_lookup = {sc.get("episode_name", os.path.basename(sc.get("video_path", ""))): sc for sc in reversed(valid_scenes)}
    updated = []
    for idx, row in enumerate(edited_rows):
        base = dict(original_edl[idx]) if idx < len(original_edl) else dict(original_edl[-1])
        v_seg, sc = dict(base.get("voice_segment", {})), dict(base.get("scene", {}))
        new_s = parse_timestamp_srt(str(row.get("Mốc Bắt Đầu", "00:00:00,000")))
        new_e = parse_timestamp_srt(str(row.get("Mốc Kết Thúc", "00:00:05,000")))
        if new_e <= new_s:
            new_e = new_s + max(0.5, float(row.get("Thời Lượng (s)", 3.0)))
        v_seg.update({
            "start": round(new_s, 3), "end": round(new_e, 3), "duration": round(new_e - new_s, 3),
            "text": str(row.get("Phụ Đề Tiếng Việt (Chỉ Đối Soát - Không Nhúng Video)", v_seg.get("text", ""))),
            "preferred_shot": str(row.get("Góc Máy", sc.get("shot_type", "Wide shot")))
        })
        req_ep = str(row.get("Tập Video Nguồn (Chuẩn Thứ Tự)", sc.get("episode_name", ""))).strip()
        if req_ep in ep_lookup and req_ep != sc.get("episode_name", ""):
            m_sc = ep_lookup[req_ep]
            sc.update({"video_path": m_sc.get("video_path", sc.get("video_path")), "episode_name": req_ep, "thumb_path": m_sc.get("thumb_path", sc.get("thumb_path"))})
        nums = re.findall(r'\d+(?:\.\d+)?', str(row.get("Khung Giờ Cảnh Gốc", "")))
        if len(nums) >= 2:
            s_o, e_o = float(nums[0]), max(float(nums[0]) + 0.5, float(nums[1]))
            sc.update({"start": round(s_o, 3), "end": round(e_o, 3), "start_time": round(s_o, 3), "end_time": round(e_o, 3), "duration": round(e_o - s_o, 3)})
        sc["shot_type"] = str(row.get("Góc Máy", sc.get("shot_type", "Wide shot")))
        base["voice_segment"], base["scene"] = v_seg, sc
        updated.append(base)
    return updated


def clean_temp_cache() -> Tuple[int, float]:
    freed_b, cnt = 0, 0
    tmp_root = tempfile.gettempdir()
    try:
        for item in os.listdir(tmp_root):
            if item.startswith("studio_pro_"):
                fp = os.path.join(tmp_root, item)
                if os.path.isdir(fp):
                    for r_d, _, fs in os.walk(fp):
                        for f in fs:
                            try:
                                freed_b += os.path.getsize(os.path.join(r_d, f))
                            except Exception:
                                pass
                    shutil.rmtree(fp, ignore_errors=True)
                    cnt += 1
    except Exception:
        pass
    for folder in (UPLOAD_CACHE_DIR, THUMB_CACHE_DIR, os.path.join(CACHE_DIR, "clip_previews"), os.path.join(os.getcwd(), "__pycache__")):
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
    for _d in (UPLOAD_CACHE_DIR, THUMB_CACHE_DIR, ELEVEN_CACHE_DIR):
        os.makedirs(_d, exist_ok=True)
    return cnt, round(freed_b / (1024 * 1024), 2)


def run_one_click_ai_fix(current_cfg: dict) -> list:
    fixed = []
    va_path = os.path.join(os.getcwd(), "vision_analyzer.py")
    if os.path.exists(va_path):
        try:
            with open(va_path, "r", encoding="utf-8") as f:
                c = f.read()
            if "gemini-2.5-pro" in c:
                with open(va_path, "w", encoding="utf-8") as f:
                    f.write(c.replace("gemini-2.5-pro", "gemini-3.1-pro-preview"))
                fixed.append("✅ Đã nâng cấp model trong `vision_analyzer.py` lên `gemini-3.1-pro-preview`.")
            else:
                fixed.append("✅ Model Gemini đã ở phiên bản chuẩn `gemini-3.1-pro-preview`.")
        except Exception as e:
            fixed.append(f"⚠️ Lỗi kiểm tra vision_analyzer.py: {e}")
    q_data = load_quota_tracker()
    if q_data.get("requests_used", 0) >= int(current_cfg.get("daily_quota_limit", 1500) * 0.9):
        current_cfg["use_ai_vision"] = False
        fixed.append("✅ Hạn mức Gemini gần đầy: Đã chuyển sang chế độ OpenCV nội bộ.")
    out_dir = os.path.dirname(os.path.abspath(current_cfg.get("output_filename", "output/video.mp4")))
    os.makedirs(out_dir, exist_ok=True)
    current_cfg["video_codec"] = "libx264"
    update_quota_tracker(error_msg="")
    save_settings(current_cfg)
    fixed.append("🛡️ Đã kích hoạt Auto-Heal & đưa Encoder về `libx264` an toàn 100%.")
    return fixed


# ==============================================================================
# KHỞI TẠO TRẠNG THÁI (SESSION STATE) & GIAO DIỆN CYBERPUNK
# ==============================================================================
saved_cfg = load_settings()

for _k, _v in [
    ("output_filepath", saved_cfg.get("output_filename", "output/video_thi_cong_hoan_chinh.mp4")),
    ("preview_ready", False), ("preview_edl", []), ("edl_objects", []), ("cached_valid_scenes", []),
    ("preview_summary", {}), ("last_system_error", ""), ("trigger_confirm_render", False),
    ("trigger_dub_only_render", False), ("last_rendered_output", ""), ("last_audit_report", None),
    ("last_elapsed_sec", 0.0), ("direct_video_paths", []), ("direct_audio_paths", []),
    ("direct_bgm_paths", []), ("direct_ref_path", ""),
    ("eleven_voices_list", DEFAULT_ELEVEN_VOICES.copy()),
    ("eleven_sub_info", {"tier": "PRO", "char_used": 0, "char_limit": 0, "connected": False}),
    ("eleven_script_rows", []), ("eleven_step_a_confirmed", False), ("eleven_step_b_ready", False),
    ("eleven_raw_text_buffer", "")
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

if saved_cfg.get("elevenlabs_api_key") and not st.session_state["eleven_sub_info"].get("connected"):
    _conn = connect_elevenlabs_api(saved_cfg["elevenlabs_api_key"])
    if _conn.get("ok"):
        st.session_state["eleven_voices_list"] = _conn["voices"]
        st.session_state["eleven_sub_info"] = {"tier": _conn["tier"], "char_used": _conn["char_used"], "char_limit": _conn["char_limit"], "connected": True}

st.set_page_config(page_title="AI Video Studio Pro Max - ElevenLabs v4 & Master Dubbing 10/10", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-stale="true"], .stApp [data-stale="true"], div[data-testid="stStatusWidget"] { opacity: 1 !important; filter: none !important; pointer-events: auto !important; }
.stApp {
    background-color: #040918;
    background-image: radial-gradient(at 15% 15%, rgba(6, 182, 212, 0.16) 0px, transparent 45%),
                      radial-gradient(at 85% 20%, rgba(30, 58, 138, 0.28) 0px, transparent 50%),
                      linear-gradient(135deg, #040918 0%, #0a132c 50%, #050d1e 100%);
    font-family: 'Segoe UI', system-ui, sans-serif !important; color: #f8fafc !important;
}
[data-testid="stSidebar"] { background: #060b19 !important; border-right: 1px solid rgba(56, 189, 248, 0.3) !important; }
h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-weight: 800 !important; }
p, label, li, span { color: #f1f5f9 !important; }
.studio-banner {
    background: linear-gradient(125deg, #0f172a 0%, #1e3a8a 52%, #0891b2 100%);
    padding: 18px 24px; border-radius: 16px; color: #ffffff;
    box-shadow: 0 10px 30px rgba(6, 182, 212, 0.22); margin-bottom: 14px; border: 1px solid rgba(56, 189, 248, 0.5);
}
.studio-banner h1 { font-size: 25px !important; font-weight: 900 !important; margin: 0 0 6px 0 !important; color: #facc15 !important; }
.badge-pill {
    display: inline-block; padding: 4px 11px; border-radius: 999px; font-size: 12px; font-weight: 700;
    background: #090e1c; color: #38bdf8 !important; margin-right: 6px; margin-top: 6px; border: 1px solid rgba(56, 189, 248, 0.45);
}
.cyber-card {
    background: rgba(15, 23, 42, 0.92); border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 14px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 6px 24px rgba(0, 0, 0, 0.4);
}
input, textarea, .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"], .stNumberInput input {
    background-color: #0f172a !important; color: #ffffff !important; font-weight: 700 !important;
    border: 1px solid #38bdf8 !important; border-radius: 10px !important;
}
.stSelectbox span, .stSelectbox div { color: #ffffff !important; }
div[data-baseweb="popover"], ul[data-baseweb="menu"], li[role="option"] { background-color: #0f172a !important; color: #ffffff !important; }
li[role="option"]:hover, li[aria-selected="true"] { background-color: #1e3a8a !important; color: #facc15 !important; }
div[data-baseweb="tooltip"], div[data-baseweb="tooltip"] *, div[role="tooltip"], div[role="tooltip"] *, [data-testid="stTooltipContent"], [data-testid="stTooltipContent"] * {
    background-color: #0f172a !important; color: #ffffff !important; font-weight: 700 !important;
}
[data-testid="stTooltipContent"], div[data-baseweb="tooltip"] > div {
    border: 1.5px solid #38bdf8 !important; border-radius: 10px !important; padding: 8px 12px !important; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.9) !important;
}
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="input"] > div {
    background-color: #0f172a !important; color: #ffffff !important; border-color: #38bdf8 !important;
}
.stSlider div[data-testid="thumbValue"] { color: #ffffff !important; font-weight: 900 !important; background: #0284c7 !important; padding: 2px 8px !important; border-radius: 6px !important; }
[data-testid="stFileUploadDropzone"] { background: #0f172a !important; border: 2px dashed #38bdf8 !important; border-radius: 14px !important; }
[data-testid="stFileUploadDropzone"] * { color: #f8fafc !important; font-weight: 600 !important; }
button:not([kind="primary"]), div.stButton > button:not([kind="primary"]), div.stDownloadButton > button,
[data-testid="stFileUploadDropzone"] button, [data-testid="stFileUploader"] button, [data-testid="baseButton-secondary"], .stNumberInput button {
    background: #1e3a8a !important; background-color: #1e3a8a !important; color: #ffffff !important;
    border: 1px solid #38bdf8 !important; font-weight: 800 !important; border-radius: 10px !important;
}
button p, button span, div.stButton > button p, div.stButton > button span, div.stDownloadButton > button p { color: #ffffff !important; }
button:not([kind="primary"]):hover, div.stButton > button:not([kind="primary"]):hover, div.stDownloadButton > button:hover {
    background: #0284c7 !important; background-color: #0284c7 !important; color: #ffffff !important; border-color: #facc15 !important;
}
div.stButton > button[kind="primary"], button[kind="primary"], [data-testid="baseButton-primary"] {
    background: linear-gradient(90deg, #b91c1c 0%, #ea580c 50%, #d97706 100%) !important; color: #ffffff !important;
    font-weight: 900 !important; font-size: 16px !important; padding: 12px 18px !important; border-radius: 12px !important; border: 1px solid #facc15 !important;
}
[data-testid="stExpander"] summary { background-color: #1e3a8a !important; color: #ffffff !important; border-radius: 10px !important; padding: 10px 14px !important; border: 1px solid #38bdf8 !important; }
[data-baseweb="tab-list"] { background: #0f172a !important; border-radius: 12px 12px 0 0 !important; padding: 6px 10px 0 10px !important; border-bottom: 2px solid rgba(56, 189, 248, 0.3) !important; }
[data-baseweb="tab"] { background-color: #0f172a !important; color: #cbd5e1 !important; font-weight: 800 !important; margin-right: 6px !important; border-radius: 8px 8px 0 0 !important; border: 1px solid rgba(56, 189, 248, 0.25) !important; }
[aria-selected="true"] { background-color: #1e3a8a !important; color: #facc15 !important; border-bottom: 3px solid #facc15 !important; }
.terminal-box { background: #020617; border: 1px solid #38bdf8; border-radius: 14px; padding: 14px 18px; margin-top: 10px; font-family: Consolas, monospace; font-size: 13px; }
.terminal-header { color: #38bdf8; font-weight: 800; margin-bottom: 8px; border-bottom: 1px solid rgba(56, 189, 248, 0.3); padding-bottom: 4px; }
.log-line { margin: 4px 0; } .log-time { color: #94a3b8; margin-right: 8px; } .log-new { color: #4ade80; font-weight: 700; } .log-old { color: #cbd5e1; opacity: 0.75; }
button:disabled, div.stButton > button:disabled, button[disabled], div.stButton > button[disabled] {
    background: #1e293b !important; background-color: #1e293b !important;
    color: #64748b !important; border: 1.5px dashed #475569 !important;
    opacity: 0.50 !important; cursor: not-allowed !important; box-shadow: none !important;
}
button:disabled p, button:disabled span, div.stButton > button:disabled p { color: #64748b !important; }
</style>
<div class="studio-banner">
    <h1>🎬 AI AUTO VIDEO STUDIO PRO MAX — ELEVENLABS v4 & MASTER DUBBING 10/10</h1>
    <p>Mượt Mà Từ Video 1 Phút Đến 3 Tiếng • Chống Lặp Từ Cuối Voice 100% • Tự Chèn Cảm Xúc & Tính Giây Nghỉ • Đồ Họa Ultra-Sharp • Xem Thử Tất Cả Video + Phụ Đề Việt</p>
    <div>
        <span class="badge-pill">🎙️ Chống Lặp Từ Cuối 100%</span>
        <span class="badge-pill">🎭 Cảm Xúc Tự Cân Nhịp (Không Đếm Từ Thẻ)</span>
        <span class="badge-pill">💎 Đồ Họa Ultra-Sharp 1080p/2K/4K</span>
        <span class="badge-pill">🇻🇳 Phụ Đề Việt Dưới Mọi Voice</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR: TẤT CẢ ĐỀU THU GỌN ĐƯỢC
# ==============================================================================
with st.sidebar:
    with st.expander("🔑 1. Kết Nối AI & ElevenLabs Pro", expanded=True):
        gemini_api_key = st.text_input("Gemini API Key (Tự động lưu)", value=saved_cfg.get("gemini_api_key", ""), type="password")
        elevenlabs_api_key = st.text_input("🎙️ ElevenLabs API Key (sk_...)", value=saved_cfg.get("elevenlabs_api_key", ""), type="password")

        el_c1, el_c2 = st.columns(2)
        with el_c1:
            if st.button("🔄 Kết nối ElevenLabs", use_container_width=True):
                conn = connect_elevenlabs_api(elevenlabs_api_key)
                if conn.get("ok"):
                    st.session_state["eleven_voices_list"] = conn["voices"]
                    st.session_state["eleven_sub_info"] = {"tier": conn["tier"], "char_used": conn["char_used"], "char_limit": conn["char_limit"], "connected": True}
                    saved_cfg["elevenlabs_api_key"] = clean_eleven_api_key(elevenlabs_api_key)
                    save_settings(saved_cfg)
                    st.toast(f"✅ Đã tải {len(conn['voices'])} giọng đọc ({conn['tier']})!")
                    st.rerun()
                else:
                    st.error(conn.get("error", "Lỗi kết nối"))
        with el_c2:
            sub_i = st.session_state.get("eleven_sub_info", {})
            st.caption(f"🌟 Gói: **{sub_i.get('tier', 'PRO')}**\n\n🔤 Đã dùng: **{sub_i.get('char_used', 0):,}**")

    with st.expander("⏳ 2. Chế Độ Tự Chạy 3 Phút & AI Vision", expanded=True):
        auto_confirm_3min = st.toggle("⏳ Bật Tự Động Chạy Tiếp Sau 3 Phút Không Bấm", value=bool(saved_cfg.get("auto_confirm_3min", True)))
        auto_confirm_seconds = st.slider("Thời gian chờ xác nhận (giây: 180s = 3 phút)", min_value=30, max_value=600, value=int(saved_cfg.get("auto_confirm_seconds", 180)), step=10)
        use_ai_vision = st.toggle("Bật Gemini 3.1 Pro (3-Frame Đa Luồng)", value=bool(saved_cfg.get("use_ai_vision", True)))
        use_smart_cache = st.toggle("⚡ Bật Smart Cache 0.5s (Hỗ trợ video 3 tiếng)", value=bool(saved_cfg.get("use_smart_cache", True)))
        whisper_options = ["base", "small", "medium", "large"]
        saved_w = saved_cfg.get("whisper_model_size", "base")
        whisper_model_size = st.selectbox("Model Whisper", options=whisper_options, index=whisper_options.index(saved_w) if saved_w in whisper_options else 0)

    with st.expander("📊 3. Hạn Mức Gemini & Dọn Rác An Toàn", expanded=False):
        q_info = load_quota_tracker()
        d_lim = int(saved_cfg.get("daily_quota_limit", 1500))
        r_used = int(q_info.get("requests_used", 0))
        st.caption(f"⚡ Quota Gemini: **{r_used}/{d_lim}** | 🪙 Token: **{int(q_info.get('tokens_estimated', 0)):,}**")
        qc1, qc2 = st.columns(2)
        with qc1:
            if st.button("🔄 Reset Quota", use_container_width=True):
                with open(QUOTA_FILE, "w", encoding="utf-8") as f:
                    json.dump({"date": str(date.today()), "requests_used": 0, "tokens_estimated": 0, "last_error": ""}, f)
                st.rerun()
        with qc2:
            if st.button("🛠️ 1-Click Fix", use_container_width=True, key="sb_fix"):
                st.session_state["fix_report"] = run_one_click_ai_fix(saved_cfg)
                st.rerun()
        if st.button("🧹 Dọn Rác Tạm (Giữ Nguyên Voice Cache)", use_container_width=True):
            cnt, mb = clean_temp_cache()
            st.toast(f"🧹 Đã giải phóng {mb} MB (Giữ nguyên 100% kho Voice ElevenLabs)!")

if st.session_state.get("fix_report"):
    with st.expander("🛠️ Báo cáo AI Auto-Fix", expanded=True):
        for ln in st.session_state["fix_report"]:
            st.write(ln)
        if st.button("Đóng"):
            st.session_state["fix_report"] = None
            st.rerun()

# ==============================================================================
# MỤC 1 & 2: ĐẶT TRONG EXPANDER THU GỌN ĐƯỢC + LỰA CHỌN ĐỒ HỌA SIÊU NÉT
# ==============================================================================
with st.expander("📥 MỤC 1 & 2: KHO TƯ LIỆU ĐẦU VÀO (1 PHÚT -> 3 TIẾNG), CHẾ ĐỘ DỰNG & LỰA CHỌN ĐỒ HỌA SIÊU NÉT (THU GỌN / MỞ RỘNG)", expanded=True):
    col_in, col_out = st.columns([1.15, 0.85])

    with col_in:
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.subheader("📥 1. Kho Tư Liệu Đầu Vào (Nạp Siêu Tốc 0 Giây Hoặc Kéo Thả)")
        fc1, fc2, fc3 = st.columns([0.42, 0.38, 0.20])
        with fc1:
            if st.button("📂 Chọn Cả Thư Mục Video (0 Giây)", use_container_width=True):
                pv = pick_local_folder_videos_dialog()
                if pv:
                    st.session_state["direct_video_paths"] = pv
                    st.rerun()
        with fc2:
            if st.button("🎞️ Chọn Video Đã Cắt Sẵn / Lẻ (0s)", use_container_width=True):
                pv = pick_local_files_dialog("Chọn Video (Hỗ trợ 1 phút đến 3 tiếng)", [("Video", "*.mp4 *.mov *.avi *.mkv *.webm")], multiple=True)
                if pv:
                    st.session_state["direct_video_paths"] = sorted(pv, key=natural_sort_key)
                    st.rerun()
        with fc3:
            if st.button("🗑️ Xóa Chọn", use_container_width=True):
                st.session_state["direct_video_paths"], st.session_state["direct_audio_paths"], st.session_state["direct_bgm_paths"], st.session_state["direct_ref_path"] = [], [], [], ""
                st.rerun()

        fs1, fs2, fs3 = st.columns(3)
        with fs1:
            if st.button("🎙️ Chọn Nhanh Voice/Note (0s)", use_container_width=True):
                pa = pick_local_files_dialog("Chọn Kịch bản (.srt/.txt) hoặc Voice (.mp3/.wav)", [("Script & Audio", "*.srt *.txt *.md *.mp3 *.wav *.m4a")], multiple=True)
                if pa:
                    st.session_state["direct_audio_paths"] = pa
                    st.rerun()
        with fs2:
            if st.button("🎵 Chọn Nhanh Nhạc BGM (0s)", use_container_width=True):
                pb = pick_local_files_dialog("Chọn Nhạc nền BGM", [("Audio", "*.mp3 *.wav *.m4a")], multiple=True)
                if pb:
                    st.session_state["direct_bgm_paths"] = pb
                    st.rerun()
        with fs3:
            if st.button("🎬 Chọn Nhanh Video Mẫu (0s)", use_container_width=True):
                pr = pick_local_files_dialog("Chọn 1 Video Mẫu", [("Video", "*.mp4 *.mov *.avi *.mkv")], multiple=False)
                if pr:
                    st.session_state["direct_ref_path"] = pr[0]
                    st.rerun()

        if st.session_state["direct_video_paths"] or st.session_state["direct_audio_paths"] or st.session_state["direct_bgm_paths"]:
            st.success(f"⚡ **Đã liên kết ổ cứng (0 giây chờ):** `{len(st.session_state['direct_video_paths'])}` Video | `{len(st.session_state['direct_audio_paths'])}` Voice/Note | `{len(st.session_state['direct_bgm_paths'])}` BGM")

        uploaded_videos = st.file_uploader("📹 Hoặc Kéo Thả Video đã cắt sẵn / Video thô (.mp4, .mov, .avi, .mkv)", type=["mp4", "mov", "avi", "mkv", "webm"], accept_multiple_files=True)
        uc1, uc2, uc3 = st.columns(3)
        with uc1:
            uploaded_audios = st.file_uploader("🎙️ Voice / Note Kịch bản (.srt, .txt, .mp3, .wav)", type=["mp3", "wav", "m4a", "md", "srt", "txt"], accept_multiple_files=True)
        with uc2:
            uploaded_bgm = st.file_uploader("🎵 Nhạc nền BGM (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"], accept_multiple_files=True)
        with uc3:
            uploaded_ref_video = st.file_uploader("🎞️ Video Mẫu (Clone ADN)", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_out:
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.subheader("📤 2. Thiết Lập Chế Độ Dựng & Chất Lượng Đồ Họa")
        edit_mode_map = {
            "dub_only": "🎙️ Chế độ 5 (CHUẨN): Chỉ Ghép Voice & Nhạc Vào Video Đã Cắt/Tua Sẵn (1 Phút -> 3 Tiếng)",
            "voice_sync": "🤖 Chế độ 1: AI Đồng Bộ Cắt Cảnh Chuẩn 100% Theo Voice / Kịch Bản Note",
            "highlight": "🎬 Chế độ 2: Clone Video Mẫu / Auto Highlight Công Trình",
            "sequential": "⏩ Chế độ 3: Nối Tuần Tự Chuẩn (Tập 1 → Tập 2 → Tập Cuối)",
            "random": "🎲 Chế độ 4: Trộn Ngẫu Nhiên Toàn Bộ (Random Remix)"
        }
        mode_keys = list(edit_mode_map.keys())
        saved_m = saved_cfg.get("edit_mode", "dub_only")
        edit_mode = st.selectbox("Chọn Chế Độ Xử Lý Video:", options=mode_keys, format_func=lambda k: edit_mode_map[k], index=mode_keys.index(saved_m) if saved_m in mode_keys else 0)

        gfx_preset_map = {
            "stream_copy": "⚡ Giữ Nguyên Bản Gốc 100% (FFmpeg Stream Copy - Xuất 3 Giây, Không Nén Lại)",
            "1080p_ultra_sharp": "💎 Đồ Họa 1080p Full HD Siêu Nét (Lanczos + Khử Mờ Unsharp Mask + CRF 16)",
            "2k_qhd_master": "🔥 Đồ Họa 2K QHD Master Studio (2560x1440 Siêu Căng Nét + CRF 15 - Chuẩn YouTube Pro)",
            "4k_uhd_cinema": "👑 Đồ Họa 4K UHD Điện Ảnh Cực Đại (3840x2160 Ultra-Sharp + CRF 14)"
        }
        gfx_keys = list(gfx_preset_map.keys())
        saved_gfx = saved_cfg.get("graphics_preset", "stream_copy")
        graphics_preset = st.selectbox(
            "🎨 Lựa Chọn Gói Đồ Họa Chống Vỡ Nét (Graphics Quality):",
            options=gfx_keys,
            format_func=lambda k: gfx_preset_map[k],
            index=gfx_keys.index(saved_gfx) if saved_gfx in gfx_keys else 0,
            help="Chọn 'Giữ Nguyên Bản Gốc' để xuất siêu tốc 3 giây, hoặc chọn '1080p/2K/4K Siêu Nét' nếu muốn bộ lọc AI Unsharp Mask tăng độ nét từng chi tiết gỗ/đá!"
        )
        fast_stream_copy_dub = (graphics_preset == "stream_copy")

        is_dub_mode = (edit_mode == "dub_only")
        dc1, dc2 = st.columns(2)
        with dc1:
            target_duration_mins = st.number_input("⏱️ Thời lượng mong muốn (Chỉ dùng cho Chế độ 1-4):", min_value=0, max_value=240, value=int(saved_cfg.get("target_duration_mins", 0)), step=1, disabled=is_dub_mode)
        with dc2:
            auto_total_duration_sec = st.number_input("⏱️ Cộng thêm số Giây lẻ (Chỉ dùng cho Chế độ 1-4):", min_value=0, max_value=7200, value=int(saved_cfg.get("auto_total_duration", 0)) if int(saved_cfg.get("target_duration_mins", 0)) == 0 else 0, step=5, disabled=is_dub_mode)
        auto_total_duration = (target_duration_mins * 60) + auto_total_duration_sec

        strict_sequential_video = st.checkbox("🔢 Ép luồng xử lý chạy đúng thứ tự từng tập (Tập 1 → Tập Cuối)", value=bool(saved_cfg.get("strict_sequential_video", True)))

        cb1, cb2 = st.columns(2)
        with cb1:
            if st.button("📁 Chọn thư mục & Đặt tên file...", use_container_width=True):
                ch = open_file_dialog(st.session_state["output_filepath"])
                if ch:
                    st.session_state["output_filepath"] = ch
                    st.rerun()
        with cb2:
            if st.button("⚡ Đặt tên tự động theo giờ", use_container_width=True):
                st.session_state["output_filepath"] = f"output/Video_Studio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                st.rerun()

        output_filename = st.text_input("Đường dẫn lưu video xuất ra:", value=st.session_state["output_filepath"])
        st.session_state["output_filepath"] = output_filename
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# MỤC 3: TRẠM ĐẠO DIỄN VOICE ELEVENLABS PRO V2.1 (ĐẶT TRONG EXPANDER THU GỌN ĐƯỢC)
# ==============================================================================
with st.expander("🎙️ MỤC 3: TRẠM ĐẠO DIỄN VOICE ELEVENLABS PRO V2.1, THẺ CẢM XÚC & XEM THỬ TẤT CẢ VIDEO + PHỤ ĐỀ VIỆT (THU GỌN / MỞ RỘNG)", expanded=(edit_mode == "dub_only")):
    eleven_cfg = render_elevenlabs_dubbing_station_ui(
        saved_cfg=saved_cfg,
        gemini_api_key=gemini_api_key,
        elevenlabs_api_key=elevenlabs_api_key,
        whisper_model_size=whisper_model_size,
        auto_confirm_3min=auto_confirm_3min,
        auto_confirm_seconds=auto_confirm_seconds,
        uploaded_videos=uploaded_videos,
        uploaded_audios=uploaded_audios,
        resolve_or_cache_fn=resolve_or_cache_media_file,
        flatten_files_fn=flatten_files
    )

# ==============================================================================
# MỤC 4: BẢNG ĐIỀU KHIỂN KỸ THUẬT CHUYÊN SÂU 5 TAB (ĐẶT TRONG EXPANDER THU GỌN ĐƯỢC)
# ==============================================================================
with st.expander("🎛️ MỤC 4: BẢNG ĐIỀU KHIỂN KỸ THUẬT CHUYÊN SÂU — ĐỒ HỌA SIÊU NÉT, RENDER & HÒA ÂM ASMR (THU GỌN / MỞ RỘNG)", expanded=False):
    tab_speed, tab_graphics, tab_render, tab_audio, tab_watermark = st.tabs([
        "⏱️ Tốc Độ & Nhịp Cắt", "🎨 Đồ Họa Siêu Nét & Màu Sắc", "🚀 Tối Ưu Render & Phần Cứng", "🔊 Bàn Hòa Âm, Giãn Cách & Auto Ducking", "🏷️ Đóng Dấu Watermark"
    ])

    with tab_speed:
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            default_speed = st.slider("Tốc độ mặc định video thô (Speed)", 0.1, 50.0, float(saved_cfg.get("default_speed", 3.0)), 0.1)
            enable_shuffle = st.checkbox("Đảo cảnh thông minh trong cùng công đoạn", value=False if strict_sequential_video else bool(saved_cfg.get("enable_shuffle", True)))
        with sc2:
            min_speed = st.slider("Giới hạn tốc độ tối thiểu (Min Speed)", 0.1, 10.0, float(saved_cfg.get("min_speed", 0.5)), 0.1)
            max_speed = st.slider("Giới hạn tốc độ tối đa (Max Speed)", 1.0, 50.0, float(saved_cfg.get("max_speed", 25.0)), 0.5)
        with sc3:
            auto_clip_duration = st.slider("Độ dài mỗi cảnh khi dựng tự động (s)", 0.5, 60.0, float(saved_cfg.get("auto_clip_duration", 3.0)), 0.5)
            require_preview_confirm = st.checkbox("🛡️ Bắt buộc Xem Trước & Xác nhận trước khi xuất (Có tự chạy sau 3 phút)", value=bool(saved_cfg.get("require_preview_confirm", True)))

    with tab_graphics:
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            aspect_options = [
                "16:9 Full HD (1920x1080 - YouTube/TV)", "9:16 TikTok / Reels / Shorts (1080x1920)",
                "1:1 Facebook / Instagram Vuông (1080x1080)", "4:5 Instagram / Feed Dọc (1080x1350)",
                "16:9 HD Nhẹ (1280x720 - Siêu tốc)", "16:9 2K QHD (2560x1440 - Siêu nét)", "16:9 4K UHD (3840x2160 - Cực đại)"
            ]
            saved_asp = saved_cfg.get("aspect_ratio", aspect_options[0])
            aspect_ratio = st.selectbox("Độ phân giải & Tỷ lệ khung hình:", options=aspect_options, index=aspect_options.index(saved_asp) if saved_asp in aspect_options else 0)
            sharpness_boost = st.slider("💎 Cường độ Làm Nét Sâu Chống Vỡ Hình (Unsharp Boost)", 0.0, 2.5, float(saved_cfg.get("sharpness_boost", 1.25)), 0.05)
            fit_map = {"crop": "🔍 Cắt & Phóng đầy khung hình", "blur_bg": "🌫️ Làm mờ nền (Blur Background)"}
            fit_mode = st.selectbox("Xử lý viền khung hình:", options=list(fit_map.keys()), format_func=lambda k: fit_map[k], index=0 if saved_cfg.get("fit_mode", "crop") == "crop" else 1)
            zoom_factor = st.slider("Hệ số phóng đại (Zoom Factor)", 1.00, 3.00, float(saved_cfg.get("zoom_factor", 1.10)), 0.02)
            flip_prob = st.slider("Tỷ lệ lật gương (Mirror Flip)", 0.0, 1.0, float(saved_cfg.get("flip_prob", 0.15)), 0.05)
        with gc2:
            color_map = {"none": "⚡ Nguyên bản 100% (Nhanh nhất - Chuẩn Stream Copy)", "auto_lut": "🌈 Đa sắc tự nhiên (Auto LUT)", "cinematic": "🎬 Điện ảnh (Cinematic)", "warm": "☀️ Nắng ấm (Warm)", "cool": "🧊 Kỹ thuật sắc nét (Cool)", "vibrant": "🔥 Rực rỡ (Vibrant)", "vintage": "🎞️ Cổ điển (Vintage)"}
            c_keys = list(color_map.keys())
            saved_col = saved_cfg.get("color_style", "none")
            color_style = st.selectbox("Bộ lọc màu (Color Grading):", options=c_keys, format_func=lambda k: color_map[k], index=c_keys.index(saved_col) if saved_col in c_keys else 0)
            lut_opacity = st.slider("Cường độ phủ màu LUT", 0.00, 0.40, float(saved_cfg.get("lut_opacity", 0.06)), 0.01)
            color_segments = st.slider("Số phân đoạn đổi màu", 1, 30, int(saved_cfg.get("color_segments", 6)), 1)
        with gc3:
            brightness = st.slider("Độ sáng (Brightness)", -80.0, 80.0, float(saved_cfg.get("brightness", 0.0)), 2.0)
            contrast = st.slider("Độ tương phản (Contrast)", 0.50, 2.20, float(saved_cfg.get("contrast", 1.0)), 0.05)
            saturation = st.slider("Độ rực màu (Saturation)", 0.00, 2.50, float(saved_cfg.get("saturation", 1.05)), 0.05)
            enable_letterbox = st.checkbox("🎬 Viền đen Điện ảnh (Letterbox)", value=bool(saved_cfg.get("enable_letterbox", False)))
            letterbox_ratio = st.slider("Độ dày viền đen", 0.02, 0.18, float(saved_cfg.get("letterbox_ratio", 0.065)), 0.005)
            enable_vignette = st.checkbox("🔍 Làm tối 4 góc (Vignette)", value=bool(saved_cfg.get("enable_vignette", False)))

    with tab_render:
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            codec_map = {"libx264": "💻 CPU Đa Nhân (libx264 - Tương thích 100%)", "h264_nvenc": "🚀 NVIDIA GPU (h264_nvenc)", "h264_qsv": "⚡ Intel QuickSync GPU", "h264_amf": "🔥 AMD Radeon GPU"}
            cd_keys = list(codec_map.keys())
            saved_cd = saved_cfg.get("video_codec", "libx264")
            video_codec = st.selectbox("Bộ mã hóa phần cứng:", options=cd_keys, format_func=lambda k: codec_map[k], index=cd_keys.index(saved_cd) if saved_cd in cd_keys else 0)
            preset_list = ["ultrafast", "superfast", "veryfast", "fast", "medium"]
            saved_pr = saved_cfg.get("render_preset", "superfast")
            render_preset = st.selectbox("Tốc độ nén FFmpeg:", options=preset_list, index=preset_list.index(saved_pr) if saved_pr in preset_list else 1)
        with rc2:
            fps_list = [24, 25, 30, 50, 60, 120]
            saved_f = int(saved_cfg.get("target_fps", 60))
            target_fps = st.selectbox("Tốc độ khung hình (FPS):", options=fps_list, index=fps_list.index(saved_f) if saved_f in fps_list else 4)
            crf_quality = st.slider("Chất lượng CRF (Số nhỏ càng nét - Chuẩn 16-18):", 14, 35, int(saved_cfg.get("crf_quality", 17)), 1)
        with rc3:
            cpu_threads = st.slider(f"Số luồng CPU song song (Tối đa {MAX_CPU}):", 1, 32, int(saved_cfg.get("cpu_threads", MAX_CPU)), 1)
            enable_crossfade = st.checkbox("✨ Chuyển cảnh mờ đè (Crossfade)", value=bool(saved_cfg.get("enable_crossfade", False)))
            crossfade_duration = st.slider("Thời gian Crossfade (s)", 0.1, 1.5, float(saved_cfg.get("crossfade_duration", 0.2)), 0.05)
            enable_fade_inout = st.checkbox("🎬 Fade In/Out đầu cuối", value=bool(saved_cfg.get("enable_fade_inout", True)))

    with tab_audio:
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            voice_volume = st.slider("🎙️ Âm lượng Giọng thuyết minh (Voice)", 0.1, 3.0, float(saved_cfg.get("voice_volume", 1.0)), 0.05)
            bgm_volume = st.slider("🎵 Âm lượng Nhạc nền (BGM)", 0.01, 2.0, float(saved_cfg.get("bgm_volume", 0.15)), 0.02)
            enable_audio_ducking = st.checkbox("🎚️ Tự động Nâng/Hạ âm nền theo Voice (Smart Ducking)", value=bool(saved_cfg.get("enable_audio_ducking", True)))
        with ac2:
            bgm_spacing_mins = st.number_input("Số Phút giãn cách giữa 2 bài nhạc (0 - 60p):", 0, 60, int(saved_cfg.get("bgm_spacing_mins", 1)), 1)
            bgm_spacing_secs = st.slider("Cộng thêm số Giây giãn cách (0 - 59s):", 0, 59, int(saved_cfg.get("bgm_spacing_secs", 15)), 1)
            total_bgm_spacing_sec = (bgm_spacing_mins * 60) + bgm_spacing_secs
        with ac3:
            # Luôn bật giữ tiếng ASMR gốc mặc định = True và âm lượng = 1.0 (100% nguyên gốc) cho Chế độ 5
            keep_original_audio = st.checkbox(
                "🛠️ Giữ tiếng máy móc công trình gốc (ASMR Nguyên Bản)",
                value=True if edit_mode == "dub_only" else bool(saved_cfg.get("keep_original_audio", True))
            )
            saved_orig_vol = float(saved_cfg.get("original_audio_volume", 1.0))
            if edit_mode == "dub_only" and saved_orig_vol < 0.8:
                saved_orig_vol = 1.0
            original_audio_volume = st.slider(
                "🔊 Âm lượng tiếng ASMR công trình gốc (1.0 = 100% Nguyên gốc)",
                0.05, 2.0, saved_orig_vol, 0.05,
                disabled=(not keep_original_audio)
            )
    with tab_watermark:
        wc1, wc2 = st.columns(2)
        with wc1:
            watermark_text = st.text_input("🏷️ Nội dung chữ đóng dấu lên Video:", value=saved_cfg.get("watermark_text", ""))
        with wc2:
            wm_pos_map = {"bottom_right": "↘️ Góc dưới phải", "bottom_left": "↙️ Góc dưới trái", "top_right": "↗️ Góc trên phải", "top_left": "↖️ Góc trên trái"}
            wm_keys = list(wm_pos_map.keys())
            saved_wp = saved_cfg.get("watermark_pos", "bottom_right")
            watermark_pos = st.selectbox("Vị trí hiển thị chữ:", options=wm_keys, format_func=lambda k: wm_pos_map[k], index=wm_keys.index(saved_wp) if saved_wp in wm_keys else 0)

current_full_config = {
    "gemini_api_key": gemini_api_key,
    "elevenlabs_api_key": clean_eleven_api_key(elevenlabs_api_key),
    **eleven_cfg,
    "fast_stream_copy_dub": fast_stream_copy_dub,
    "graphics_preset": graphics_preset,
    "sharpness_boost": sharpness_boost,
    "auto_confirm_3min": auto_confirm_3min,
    "auto_confirm_seconds": auto_confirm_seconds,
    "use_ai_vision": use_ai_vision,
    "use_smart_cache": use_smart_cache,
    "whisper_model_size": whisper_model_size,
    "edit_mode": edit_mode,
    "aspect_ratio": aspect_ratio,
    "fit_mode": fit_mode,
    "target_fps": target_fps,
    "video_codec": video_codec,
    "render_preset": render_preset,
    "crf_quality": crf_quality,
    "cpu_threads": cpu_threads,
    "color_style": color_style,
    "lut_opacity": lut_opacity,
    "brightness": brightness,
    "contrast": contrast,
    "saturation": saturation,
    "enable_vignette": enable_vignette,
    "enable_letterbox": enable_letterbox,
    "letterbox_ratio": letterbox_ratio,
    "enable_crossfade": enable_crossfade,
    "crossfade_duration": crossfade_duration,
    "enable_fade_inout": enable_fade_inout,
    "watermark_text": watermark_text,
    "watermark_pos": watermark_pos,
    "zoom_factor": zoom_factor,
    "default_speed": default_speed,
    "min_speed": min_speed,
    "max_speed": max_speed,
    "flip_prob": flip_prob,
    "color_segments": color_segments,
    "enable_shuffle": False if strict_sequential_video else enable_shuffle,
    "auto_clip_duration": auto_clip_duration,
    "auto_total_duration": auto_total_duration,
    "target_duration_mins": target_duration_mins,
    "voice_volume": voice_volume,
    "keep_original_audio": keep_original_audio,
    "original_audio_volume": original_audio_volume,
    "bgm_volume": bgm_volume,
    "bgm_spacing_mins": bgm_spacing_mins,
    "bgm_spacing_secs": bgm_spacing_secs,
    "enable_audio_ducking": enable_audio_ducking,
    "strict_sequential_video": strict_sequential_video,
    "require_preview_confirm": require_preview_confirm,
    "output_filename": output_filename
}

# ==============================================================================
# MỤC 5: TRUNG TÂM ĐIỀU KHIỂN XEM TRƯỚC & XUẤT BẢN VIDEO (THU GỌN ĐƯỢC)
# ==============================================================================
with st.expander("🚀 MỤC 5: TRUNG TÂM ĐIỀU KHIỂN XEM TRƯỚC & XUẤT BẢN VIDEO CHÍNH THỨC (THU GỌN / MỞ RỘNG)", expanded=True):
    _has_vids_loaded = bool(st.session_state.get("direct_video_paths") or uploaded_videos)
    _is_dub_only = (edit_mode == "dub_only")
    ct1, ct2, ct3, ct4 = st.columns([0.18, 0.36, 0.31, 0.15])
    with ct1:
        if st.button("💾 Lưu Cấu Hình", use_container_width=True):
            save_settings(current_full_config)
            st.toast("✅ Đã lưu toàn bộ thông số vào settings.json!")
    with ct2:
        preview_clicked = st.button(
            "🔒 BƯỚC 1: TẠO TIMELINE CẮT CẢNH (CHỈ DÙNG CHO CHẾ ĐỘ 1-4)" if _is_dub_only else ("👁️ BƯỚC 1: PHÂN TÍCH & TẠO BẢN XEM TRƯỚC" if _has_vids_loaded else "🔒 BƯỚC 1: CHỌN VIDEO Ở MỤC 1 TRƯỚC"),
            use_container_width=True,
            disabled=(_is_dub_only or not _has_vids_loaded)
        )
    with ct3:
        start_clicked = st.button(
            "🚀 BƯỚC 2: XÁC NHẬN & XUẤT VIDEO CHÍNH THỨC (GIỮ 100% ASMR GỐC)" if _has_vids_loaded else "🔒 BƯỚC 2: HÃY CHỌN VIDEO Ở MỤC 1 TRƯỚC KHI XUẤT",
            type="primary",
            use_container_width=True,
            disabled=(not _has_vids_loaded)
        )
    with ct4:
        if st.button("🛠️ Sửa Lỗi 1-Click", use_container_width=True, key="main_fix_btn"):
            st.session_state["fix_report"] = run_one_click_ai_fix(current_full_config)
            st.rerun()


def execute_studio_pipeline(is_preview_only: bool = False, use_manual_edited_edl: bool = False, force_dub_only: bool = False):
    v_files = [LocalDiskFileProxy(p) for p in st.session_state.get("direct_video_paths", []) if os.path.exists(p)] or flatten_files(uploaded_videos)
    if not v_files:
        st.error("⚠️ Vui lòng chọn Video (0 Giây) hoặc kéo thả Video ở Mục 1 để bắt đầu!")
        return

    a_files = [LocalDiskFileProxy(p) for p in st.session_state.get("direct_audio_paths", []) if os.path.exists(p)] or flatten_files(uploaded_audios)
    b_files = [LocalDiskFileProxy(p) for p in st.session_state.get("direct_bgm_paths", []) if os.path.exists(p)] or flatten_files(uploaded_bgm)
    ref_obj = LocalDiskFileProxy(st.session_state["direct_ref_path"]) if (st.session_state.get("direct_ref_path") and os.path.exists(st.session_state["direct_ref_path"])) else uploaded_ref_video

    save_settings(current_full_config)
    st.session_state["sys_logs"], st.session_state["last_eta"], st.session_state["processed_bytes_total"], st.session_state["last_system_error"] = [], 0.0, 0, ""
    start_wall_time = time.time()
    st.session_state["process_start_time"], st.session_state["last_ui_refresh"] = start_wall_time, 0.0

    if strict_sequential_video:
        v_files = sorted(v_files, key=natural_sort_key)
    if a_files:
        a_files = sorted(a_files, key=natural_sort_key)

    work_dir = tempfile.mkdtemp(prefix="studio_pro_")
    percent_banner, progress_bar, log_console, status_box = st.empty(), st.progress(0.0), st.empty(), st.empty()

    def set_live_progress(pct: float, stage_title: str, detail: str = "", force_render: bool = False):
        now_t = time.time()
        if not force_render and pct < 0.999 and (now_t - st.session_state.get("last_ui_refresh", 0.0)) < 0.09:
            return
        st.session_state["last_ui_refresh"] = now_t
        pct_c = max(0.001, min(1.0, pct))
        el_s = now_t - start_wall_time
        if pct_c > 0.015 and el_s > 0.4:
            raw_rem = max(0.0, (el_s / pct_c) - el_s)
            prev_e = st.session_state.get("last_eta", 0.0)
            sm_e = raw_rem if prev_e <= 0.0 else (prev_e * 0.65 + raw_rem * 0.35)
            st.session_state["last_eta"] = sm_e
            eta_s = int(round(sm_e))
            eta_str = f"⏳ Còn: {eta_s // 60}p {eta_s % 60:02d}s" if eta_s >= 60 else f"⏳ Còn: {eta_s}s ({int(sm_e * 1000)} ms)"
        else:
            eta_str = "⏳ Đang đo vận tốc..."

        progress_bar.progress(pct_c)
        q_now = load_quota_tracker()
        percent_banner.markdown(
            f'<div class="cyber-card" style="border-left:6px solid #38bdf8;padding:16px 22px;"><div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div style="max-width:78%;"><div style="font-size:19px;font-weight:900;color:#38bdf8;margin-bottom:5px;">{stage_title}</div>'
            f'<div style="font-size:14px;color:#fff;font-weight:600;">{detail}</div>'
            f'<div style="font-size:12.5px;margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;">'
            f'<span style="background:#064e3b;color:#6ee7b7;padding:3px 9px;border-radius:6px;">⏱️ Đã chạy: {el_s:.2f}s</span>'
            f'<span style="background:#713f12;color:#fde047;padding:3px 9px;border-radius:6px;">{eta_str}</span>'
            f'<span style="background:#1e3a8a;color:#93c5fd;padding:3px 9px;border-radius:6px;">🌐 I/O: {get_realtime_io_speed()}</span>'
            f'<span style="background:#581c87;color:#e9d5ff;padding:3px 9px;border-radius:6px;">🔑 Quota: {q_now.get("requests_used",0)}</span>'
            f'</div></div><div style="font-size:36px;font-weight:900;color:#facc15;font-family:Consolas,monospace;">{pct_c*100:.2f}%</div></div></div>',
            unsafe_allow_html=True
        )
        ts_ms = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"{stage_title} → {detail}"
        if not st.session_state["sys_logs"] or log_entry not in st.session_state["sys_logs"][-1]:
            st.session_state["sys_logs"].append(f"<span class='log-time'>[{ts_ms}]</span> {log_entry}")
            if len(st.session_state["sys_logs"]) > 7:
                st.session_state["sys_logs"].pop(0)
        log_html = "".join([f"<div class='log-line {'log-new' if i == len(st.session_state['sys_logs'])-1 else 'log-old'}'>>_ {m}</div>" for i, m in enumerate(st.session_state["sys_logs"])])
        log_console.markdown(f"<div class='terminal-box'><div class='terminal-header'>⚙️ LIVE AI CONSOLE</div>{log_html}</div>", unsafe_allow_html=True)

    try:
        set_live_progress(0.02, "⚡ LIÊN KẾT KHO VIDEO SIÊU TỐC", f"Đang nạp {len(v_files)} video...", force_render=True)
        video_paths, ep_to_work_path = [], {}
        for idx, vf in enumerate(v_files):
            vp, wb = resolve_or_cache_media_file(vf, prefix_idx=idx + 1)
            st.session_state["processed_bytes_total"] += wb
            video_paths.append(vp)
            ep_to_work_path[vf.name] = vp

        raw_bgms = [resolve_or_cache_media_file(bf)[0] for bf in b_files]
        bgm_paths = build_spaced_bgm_files(raw_bgms, float(total_bgm_spacing_sec), work_dir) if raw_bgms else []

        res_lookup = {"1920x1080": (1920, 1080), "1080x1920": (1080, 1920), "1080x1080": (1080, 1080), "1080x1350": (1080, 1350), "1280x720": (1280, 720), "2560x1440": (2560, 1440), "3840x2160": (3840, 2160)}
        target_res = next((val for k, val in res_lookup.items() if k in aspect_ratio), (1920, 1080))

        editor = AutoVideoEditor(
            target_resolution=target_res, target_fps=int(target_fps), zoom_factor=zoom_factor,
            default_speed=default_speed, min_speed=min_speed, max_speed=max_speed, flip_probability=flip_prob,
            color_segments_count=color_segments, color_style=color_style, lut_opacity=lut_opacity,
            brightness=brightness, contrast=contrast, saturation=saturation, sharpness_boost=sharpness_boost,
            enable_vignette=enable_vignette, enable_letterbox=enable_letterbox, letterbox_ratio=letterbox_ratio,
            enable_crossfade=enable_crossfade, crossfade_duration=crossfade_duration, enable_fade_inout=enable_fade_inout,
            fit_mode=fit_mode, watermark_text=watermark_text, watermark_pos=watermark_pos, render_preset=render_preset,
            video_codec=video_codec, crf_quality=crf_quality, cpu_threads=cpu_threads, enable_audio_ducking=enable_audio_ducking
        )

        active_mode = "dub_only" if force_dub_only else edit_mode

        # ======================================================================
        # LUỒNG CHẾ ĐỘ 5 (DUB_ONLY): GHÉP VOICE & NHẠC VÀO VIDEO ĐÃ CẮT SẴN (1 PHÚT -> 3 TIẾNG)
        # ======================================================================
        if active_mode == "dub_only":
            timed_items: List[Dict[str, Any]] = []
            master_aud_p: Optional[str] = None
            lead_sec = float(eleven_cfg.get("voice_lead_ms", 0)) / 1000.0
            align_mode = str(eleven_cfg.get("slot_align_mode", "start"))
            auto_fit_slot = bool(eleven_cfg.get("auto_fit_voice_slot", True))

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
                    if (not r.get("audio_path") or not os.path.exists(str(r.get("audio_path", "")))) and clean_eleven_api_key(elevenlabs_api_key) and not is_preview_only:
                        set_live_progress(
                            0.08 + (ir / max(1, len(el_rows))) * 0.25,
                            "🎙️ ELEVENLABS TỰ ĐỘNG TẠO CÁC CÂU VOICE CÒN THIẾU",
                            f"Đang tạo câu #{r['STT']}/{len(el_rows)}...",
                            force_render=True
                        )
                        rt = generate_elevenlabs_tts_segment(
                            elevenlabs_api_key,
                            r["Kịch Bản Đạo Diễn (Nhấp Đúp Để Sửa)"],
                            eleven_cfg["eleven_voice_id"],
                            eleven_cfg["eleven_model_id"],
                            eleven_cfg["eleven_speed"],
                            eleven_cfg["eleven_stability"],
                            eleven_cfg["eleven_similarity"],
                            eleven_cfg["eleven_style"],
                            False
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
                ap = AudioProcessor(model_size=whisper_model_size)
                if s_fs and v_fs:
                    segs = ap.parse_script_file(s_fs[0]).get("segments", [])
                    if len(v_fs) > 1 and len(segs) >= len(v_fs):
                        for i_v, vp_a in enumerate(v_fs):
                            s_al, e_al = _compute_aligned_window(float(segs[i_v]["start"]), float(segs[i_v]["end"]), 0.0)
                            timed_items.append({"start": s_al, "end": e_al, "audio_path": vp_a, "text": segs[i_v].get("text", "")})
                    else:
                        master_aud_p = v_fs[0]
                elif len(v_fs) == 1:
                    master_aud_p = v_fs[0]
                elif len(v_fs) > 1:
                    master_aud_p = os.path.join(work_dir, "merged_voice_dub.wav")
                    ap.process_multiple_audios(v_fs, master_aud_p)

            if is_preview_only:
                set_live_progress(1.0, "✅ SẴN SÀNG GHÉP VOICE VÀO VIDEO ĐÃ CẮT SẴN!", "Kiểm tra bảng kịch bản ở Mục 3 hoặc bấm Bước 2 để ghép ngay.", force_render=True)
                status_box.info("🎙️ **Chế độ 5 đã sẵn sàng!** Hãy kiểm tra bảng thời gian ở Mục 3 hoặc bấm **🚀 BƯỚC 2: XÁC NHẬN & XUẤT VIDEO CHÍNH THỨC**.")
                return

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
                progress_callback=lambda r, msg: set_live_progress(0.35 + r * 0.65, "🎙️ ĐANG GHÉP VOICE SẠCH & ĐỒ HỌA VÀO TOÀN BỘ VIDEO", msg)
            )
            tot_el = round(time.time() - start_wall_time, 2)
            set_live_progress(1.0, f"🎉 HOÀN TẤT GHÉP VOICE 100% ({tot_el}s)!", f"Đã lưu: {output_filename}", force_render=True)
            status_box.success(f"✅ Đã ghép {rep.get('total_clips_assembled', 1)} câu Voice (Sạch 100% không lặp từ) vào toàn bộ **{rep.get('video_duration_sec', 0)} giây** Video tại: `{output_filename}`")
            st.session_state["last_rendered_output"], st.session_state["last_audit_report"], st.session_state["last_elapsed_sec"] = output_filename, rep, tot_el
            return

        # ======================================================================
        # LUỒNG CHẾ ĐỘ 1, 2, 3, 4: CẮT GHÉP VIDEO TỰ ĐỘNG TỪ KHO VIDEO THÔ
        # ======================================================================
        voice_data, master_audio_path = None, None
        if a_files:
            a_paths = [resolve_or_cache_media_file(af)[0] for af in a_files]
            s_fs = [p for p in a_paths if p.lower().endswith(('.md', '.srt', '.txt'))]
            v_fs = [p for p in a_paths if not p.lower().endswith(('.md', '.srt', '.txt'))]
            ap = AudioProcessor(model_size=whisper_model_size)
            if s_fs:
                voice_data = ap.parse_script_file(s_fs[0])
                if len(v_fs) == 1:
                    master_audio_path = v_fs[0]
                elif len(v_fs) > 1:
                    master_audio_path = os.path.join(work_dir, "master_voice_merged.wav")
                    ap.process_multiple_audios(v_fs, master_audio_path)
            elif len(v_fs) == 1:
                master_audio_path = v_fs[0]
                voice_data = ap.transcribe_and_analyze(master_audio_path)
            elif len(v_fs) > 1:
                master_audio_path = os.path.join(work_dir, "master_voice_merged.wav")
                voice_data = ap.process_multiple_audios(v_fs, master_audio_path)

        active_api = None if is_preview_only else (gemini_api_key if (use_ai_vision and gemini_api_key) else None)
        va = VisionAnalyzer(gemini_api_key=active_api, temp_dir=THUMB_CACHE_DIR, use_cache=bool(use_smart_cache), max_workers=min(8, int(cpu_threads)))

        ref_scenes_dna = []
        if ref_obj is not None:
            ref_p, _ = resolve_or_cache_media_file(ref_obj)
            ref_va = VisionAnalyzer(gemini_api_key=None, temp_dir=THUMB_CACHE_DIR, use_cache=bool(use_smart_cache), max_workers=min(8, int(cpu_threads)))
            ref_scenes_dna = [s for s in ref_va.analyze_videos([ref_p], is_reference=True) if float(s.get("duration", 3.0)) > 0.4]

        valid_scenes = []
        total_vids = len(video_paths)
        for v_idx, v_path in enumerate(video_paths):
            orig_ep = v_files[v_idx].name
            base_p = 0.20 + (v_idx / max(1, total_vids)) * 0.28
            span_p = 0.28 / max(1, total_vids)
            scs = va.analyze_videos(
                [v_path], is_reference=False,
                progress_callback=lambda r, msg, bp=base_p, sp=span_p, vi=v_idx: set_live_progress(bp + r * sp, f"👁️ QUÉT VIDEO ({vi + 1}/{total_vids})", msg)
            )
            for s in scs:
                s["video_path"], s["episode_index"], s["episode_name"] = v_path, v_idx + 1, orig_ep
            valid_scenes.extend(scs)
            if active_api and not getattr(va, "last_was_cached", False):
                update_quota_tracker(add_requests=max(1, len(scs)), add_tokens=max(1, len(scs)) * 320)

        if strict_sequential_video:
            valid_scenes.sort(key=lambda x: (x.get("episode_index", 0), float(x.get("start", x.get("start_time", 0.0)))))
        st.session_state["cached_valid_scenes"] = valid_scenes

        if use_manual_edited_edl and st.session_state.get("edl_objects") and st.session_state.get("preview_edl"):
            edl = apply_user_table_edits_to_edl(st.session_state["preview_edl"], st.session_state["edl_objects"], valid_scenes)
            for it in edl:
                ep_n = it.get("scene", {}).get("episode_name", "")
                if ep_n in ep_to_work_path:
                    it["scene"]["video_path"] = ep_to_work_path[ep_n]
        else:
            mode_arg = "voice_sync" if (edit_mode == "voice_sync" and voice_data and voice_data.get("segments")) else ("reference_clone" if (ref_obj is not None and ref_scenes_dna) else "auto")
            edl = smart_director_build_edl(
                mode=mode_arg, valid_scenes=valid_scenes,
                voice_segments=voice_data["segments"] if (voice_data and voice_data.get("segments")) else None,
                ref_scenes_dna=ref_scenes_dna, strict_sequential=(edit_mode != "random" and strict_sequential_video),
                target_total_duration=float(auto_total_duration), default_clip_duration=float(auto_clip_duration), default_spd=float(default_speed)
            )

        st.session_state["edl_objects"] = edl
        preview_rows = []
        for i, item in enumerate(edl):
            v_seg, sc = item.get("voice_segment", {}), item.get("scene", {})
            s_t, e_t = float(sc.get("start", sc.get("start_time", 0.0))), float(sc.get("end", sc.get("end_time", 0.0)))
            preview_rows.append({
                "STT": i + 1, "Mốc Bắt Đầu": format_timestamp_srt(float(v_seg.get("start", 0.0))),
                "Mốc Kết Thúc": format_timestamp_srt(float(v_seg.get("end", 0.0))), "Thời Lượng (s)": round(float(v_seg.get("duration", 0.0)), 2),
                "Phụ Đề Tiếng Việt (Chỉ Đối Soát - Không Nhúng Video)": v_seg.get("text", ""),
                "Tập Video Nguồn (Chuẩn Thứ Tự)": sc.get("episode_name", os.path.basename(sc.get("video_path", ""))),
                "Khung Giờ Cảnh Gốc": f"{round(s_t, 1)}s - {round(e_t, 1)}s", "Góc Máy": sc.get("shot_type", "Wide shot"), "Giai Đoạn": item.get("assigned_stage", 2)
            })

        st.session_state["preview_edl"] = preview_rows
        st.session_state["preview_srt_content"] = export_edl_to_srt_string(edl)
        st.session_state["preview_xml_content"] = export_edl_to_fcpxml_string(edl, fps=int(target_fps))
        st.session_state["preview_json_content"] = json.dumps(preview_rows, indent=2, ensure_ascii=False)
        st.session_state["preview_ready"] = True
        st.session_state["preview_summary"] = {
            "total_clips": len(edl), "total_duration_sec": round(float(edl[-1]["voice_segment"]["end"]), 2) if edl else 0,
            "estimated_gemini_calls": len(valid_scenes) if (use_ai_vision and gemini_api_key) else 0,
            "bgm_spacing_sec": total_bgm_spacing_sec,
            "unique_episodes_covered": len(set(r["Tập Video Nguồn (Chuẩn Thứ Tự)"] for r in preview_rows)),
            "total_episodes_uploaded": total_vids
        }

        if is_preview_only:
            el_p = round(time.time() - start_wall_time, 2)
            set_live_progress(1.0, f"✅ ĐÃ TẠO BẢN XEM TRƯỚC TRONG {el_p} GIÂY!", "Kiểm tra bảng bên dưới (Quá 3 phút không bấm sẽ tự động xuất video).", force_render=True)
            return

        render_sig = inspect.signature(editor.render_final_video)
        render_kwargs: Dict[str, Any] = {
            "edl": edl, "audio_path": master_audio_path, "output_path": output_filename,
            "bgm_volume": bgm_volume, "voice_volume": voice_volume, "keep_original_audio": keep_original_audio,
            "original_audio_volume": original_audio_volume,
            "progress_callback": lambda r, msg: set_live_progress(0.52 + r * 0.48, "🎞️ ĐANG KẾT XUẤT VIDEO MP4", msg)
        }
        if "bgm_paths" in render_sig.parameters:
            render_kwargs["bgm_paths"] = bgm_paths
        if "all_scenes_pool" in render_sig.parameters:
            render_kwargs["all_scenes_pool"] = valid_scenes

        audit_rep = editor.render_final_video(**render_kwargs)
        tot_el = round(time.time() - start_wall_time, 2)
        set_live_progress(1.0, f"🎉 HOÀN TẤT XUẤT VIDEO 100% ({tot_el}s)!", f"Đã lưu: {output_filename}", force_render=True)
        st.session_state["last_rendered_output"], st.session_state["last_audit_report"], st.session_state["last_elapsed_sec"] = output_filename, audit_rep, tot_el

    except Exception as e:
        err_t = str(e)
        st.session_state["last_system_error"] = err_t
        update_quota_tracker(error_msg=err_t)
        st.error(f"❌ Đã xảy ra lỗi: {err_t}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if preview_clicked:
    execute_studio_pipeline(is_preview_only=True, use_manual_edited_edl=False)

if st.session_state.get("trigger_dub_only_render", False):
    st.session_state["trigger_dub_only_render"] = False
    execute_studio_pipeline(is_preview_only=False, use_manual_edited_edl=False, force_dub_only=True)

if start_clicked or st.session_state.get("trigger_confirm_render", False):
    if st.session_state.get("trigger_confirm_render", False):
        st.session_state["trigger_confirm_render"] = False
        execute_studio_pipeline(is_preview_only=False, use_manual_edited_edl=True)
    elif edit_mode == "dub_only":
        execute_studio_pipeline(is_preview_only=False, use_manual_edited_edl=False, force_dub_only=True)
    elif require_preview_confirm and not st.session_state.get("preview_ready", False):
        execute_studio_pipeline(is_preview_only=True, use_manual_edited_edl=False)
    else:
        execute_studio_pipeline(is_preview_only=False, use_manual_edited_edl=True)

if st.session_state.get("last_rendered_output") and os.path.exists(st.session_state["last_rendered_output"]):
    with st.expander("🎬 KẾT QUẢ VIDEO HOÀN CHỈNH ĐÃ XUẤT (THU GỌN / MỞ RỘNG)", expanded=True):
        out_done = st.session_state["last_rendered_output"]
        rep_d = st.session_state.get("last_audit_report") or {}
        if st.button("📂 Mở Thư Mục Chứa Video Ngay", key="open_out_btn"):
            if hasattr(os, "startfile"):
                os.startfile(os.path.dirname(os.path.abspath(out_done)))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Thời gian xử lý", f"{st.session_state.get('last_elapsed_sec', 0.0)}s")
        m2.metric("Thời lượng Video", f"{rep_d.get('video_duration_sec', 0)}s")
        m3.metric("Tỷ lệ khớp nhịp", f"{rep_d.get('shot_match_rate_percent', 100)}%")
        m4.metric("Tổng số phân đoạn", rep_d.get("total_clips_assembled", 1))
        st.video(out_done)

if st.session_state.get("last_system_error"):
    st.error(f"🚨 Lỗi hệ thống: {st.session_state['last_system_error']}")
    if st.button("🛠️ BẤM ĐỂ AI TỰ SỬA LỖI (1-CLICK FIX)", type="primary", use_container_width=True):
        st.session_state["fix_report"] = run_one_click_ai_fix(current_full_config)
        st.session_state["last_system_error"] = ""
        st.rerun()

if st.session_state.get("preview_ready") and st.session_state.get("preview_edl"):
    with st.expander("🇻🇳 MỤC 6: BÀN DỰNG TƯƠNG TÁC TRỰC TIẾP & ĐỐI SOÁT PHỤ ĐỀ (THU GỌN / MỞ RỘNG)", expanded=True):
        render_auto_confirm_countdown_widget(
            target_btn_text="ĐỒNG Ý BẢNG TRÊN",
            step_label="Kiểm tra Bảng Timeline Xem Trước",
            timeout_sec=int(auto_confirm_seconds),
            enabled=bool(auto_confirm_3min)
        )

        if st.session_state.get("edl_objects"):
            st.markdown(build_visual_timeline_html(st.session_state["edl_objects"]), unsafe_allow_html=True)

        edited_tb = st.data_editor(st.session_state["preview_edl"], use_container_width=True, height=320, num_rows="fixed", key="edl_editor")
        st.session_state["preview_edl"] = edited_tb

        with st.expander("🖼️ XEM STORYBOARD HÌNH ẢNH THỰC TẾ TỪNG CẢNH", expanded=False):
            edl_objs = st.session_state.get("edl_objects", [])
            for r_s in range(0, len(edl_objs), 4):
                st_cols = st.columns(4)
                for ci, it_o in enumerate(edl_objs[r_s:r_s + 4]):
                    with st_cols[ci]:
                        sc_o, v_o = it_o.get("scene", {}), it_o.get("voice_segment", {})
                        th_p = sc_o.get("thumb_path") or sc_o.get("keyframe_path")
                        if th_p and os.path.exists(th_p):
                            st.image(th_p, use_container_width=True)
                        st.caption(f"**#{r_s + ci + 1}** | `{sc_o.get('episode_name','')}` | {v_o.get('text','')[:70]}")

        dc1, dc2, dc3, dc4 = st.columns([0.22, 0.22, 0.22, 0.34])
        with dc1:
            st.download_button("📥 Tải Phụ Đề (.SRT)", data=st.session_state.get("preview_srt_content", "").encode("utf-8"), file_name="Phu_De_Doc_Lap.srt", mime="text/plain", use_container_width=True)
        with dc2:
            st.download_button("📥 Tải Timeline (.XML)", data=st.session_state.get("preview_xml_content", "").encode("utf-8"), file_name="Timeline_Premiere.xml", mime="application/xml", use_container_width=True)
        with dc3:
            st.download_button("📥 Tải Dự Án (.JSON)", data=st.session_state.get("preview_json_content", "").encode("utf-8"), file_name="Project_EDL.json", mime="application/json", use_container_width=True)
        with dc4:
            if st.button("✅ ĐỒNG Ý BẢNG TRÊN → GỌI GEMINI & XUẤT VIDEO NGAY", type="primary", use_container_width=True, key="confirm_preview_render_btn"):
                st.session_state["trigger_confirm_render"] = True
                st.rerun()