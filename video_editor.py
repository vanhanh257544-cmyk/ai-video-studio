import os
import wave
import random
import tempfile
import subprocess
from typing import List, Dict, Any, Tuple, Optional
import cv2  # type: ignore
import numpy as np  # type: ignore
import PIL.Image  # type: ignore
import PIL.ImageDraw  # type: ignore
import PIL.ImageFont  # type: ignore

if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from moviepy.editor import (  # type: ignore
    VideoFileClip,
    AudioFileClip,
    CompositeAudioClip,
    concatenate_videoclips,
    concatenate_audioclips,
    vfx,
    afx
)
from moviepy.audio.AudioClip import AudioArrayClip  # type: ignore


def _safe_soundarray(clip, fps: int = 44100):
    """Thay clip.to_soundarray(): MoviePy 1.0.3 + NumPy >= 1.24 bị lỗi
    'arrays to stack must be passed as a sequence' vì đưa generator vào np.vstack."""
    chunks = list(clip.iter_chunks(fps=fps, quantize=False, nbytes=2, chunksize=50000))
    if not chunks:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


try:
    from proglog import ProgressBarLogger  # type: ignore

    class _RenderProgressLogger(ProgressBarLogger):
        def __init__(self, on_progress):
            super().__init__()
            self.on_progress = on_progress

        def bars_callback(self, bar, attr, value, old_value=None):
            if bar == "t" and attr == "index" and self.on_progress:
                total = self.bars[bar].get("total", 0)
                if total > 0:
                    self.on_progress(min(1.0, float(value) / float(total)))
except Exception:
    _RenderProgressLogger = None


class AutoVideoEditor:
    NATURAL_COLOR_OVERLAYS = [
        ("Warm Daylight", np.array([255, 244, 228], dtype=np.float32)),
        ("Crisp Architectural", np.array([235, 245, 255], dtype=np.float32)),
        ("Neutral Contrast", np.array([245, 245, 245], dtype=np.float32)),
        ("Earth Tone", np.array([250, 240, 230], dtype=np.float32)),
        ("Clean Cyan Tint", np.array([232, 248, 248], dtype=np.float32)),
        ("Golden Finish", np.array([255, 242, 220], dtype=np.float32)),
    ]

    COLOR_PRESETS = {
        "warm": np.array([255, 238, 210], dtype=np.float32),
        "cool": np.array([220, 240, 255], dtype=np.float32),
        "cinematic": np.array([230, 238, 248], dtype=np.float32),
        "vibrant": np.array([255, 245, 235], dtype=np.float32),
        "vintage": np.array([245, 232, 205], dtype=np.float32),
    }

    def __init__(
        self,
        target_resolution: Tuple[int, int] = (1920, 1080),
        target_fps: int = 60,
        zoom_factor: float = 1.10,
        default_speed: float = 2.0,
        min_speed: float = 0.5,
        max_speed: float = 20.0,
        flip_probability: float = 0.15,
        color_segments_count: int = 6,
        color_style: str = "none",
        lut_opacity: float = 0.06,
        brightness: float = 0.0,
        contrast: float = 1.0,
        saturation: float = 1.05,
        sharpness_boost: float = 0.0,
        enable_vignette: bool = False,
        enable_letterbox: bool = False,
        letterbox_ratio: float = 0.065,
        enable_crossfade: bool = False,
        crossfade_duration: float = 0.2,
        enable_fade_inout: bool = True,
        fit_mode: str = "crop",
        watermark_text: str = "",
        watermark_pos: str = "bottom_right",
        render_preset: str = "superfast",
        video_codec: str = "libx264",
        crf_quality: int = 18,
        cpu_threads: int = 8,
        enable_audio_ducking: bool = True
    ):
        self.target_w, self.target_h = target_resolution
        self.target_fps = target_fps
        self.zoom_factor = max(1.0, zoom_factor)
        self.default_speed = max(0.1, default_speed)
        self.min_speed = max(0.1, min_speed)
        self.max_speed = max(self.min_speed, max_speed)
        self.flip_probability = flip_probability
        self.color_segments_count = max(1, min(30, color_segments_count))
        self.color_style = color_style
        self.lut_opacity = float(np.clip(lut_opacity, 0.0, 0.5))
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.sharpness_boost = float(np.clip(sharpness_boost, 0.0, 2.5))
        self.enable_vignette = enable_vignette
        self.enable_letterbox = enable_letterbox
        self.letterbox_ratio = float(np.clip(letterbox_ratio, 0.0, 0.25))
        self.enable_crossfade = enable_crossfade
        self.crossfade_duration = max(0.05, crossfade_duration)
        self.enable_fade_inout = enable_fade_inout
        self.fit_mode = fit_mode
        self.watermark_text = watermark_text.strip()
        self.watermark_pos = watermark_pos
        self.render_preset = render_preset
        self.video_codec = video_codec
        self.crf_quality = int(np.clip(crf_quality, 14, 35))
        self.cpu_threads = max(1, cpu_threads)
        self.enable_audio_ducking = enable_audio_ducking

        self._vignette_mask = self._build_vignette_mask() if enable_vignette else None
        self._watermark_overlay = self._build_watermark_overlay() if self.watermark_text else None

    @staticmethod
    def _get_media_duration_fast(media_path: str) -> float:
        """Đo thời lượng chính xác của file Video/Audio mà không tốn RAM (Hỗ trợ video dài 2-3 tiếng)."""
        if not media_path or not os.path.exists(media_path):
            return 0.0
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                media_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
            val = float(res.stdout.strip())
            if val > 0:
                return val
        except Exception:
            pass
        try:
            cap = cv2.VideoCapture(media_path)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            f_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            if fps > 0 and f_cnt > 0:
                return f_cnt / fps
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _decode_audio_to_pcm_clean(audio_path: str, work_dir: str, tag: str, sample_rate: int = 44100) -> Optional[np.ndarray]:
        """
        Giải mã trực tiếp file MP3/WAV sang chuẩn WAV PCM 16-bit sạch bằng FFmpeg và đọc bằng thư viện wave.
        KHẮC PHỤC TRIỆT ĐỂ 100% LỖI NHẠI LẠI / LẶP LẠI TỪ CUỐI Ở CÁC VOICE DO MOVIEPY GÂY RA!
        """
        if not audio_path or not os.path.exists(audio_path):
            return None
        tmp_wav = os.path.join(work_dir, f"clean_pcm_{tag}.wav")
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(sample_rate),
                "-ac", "2",
                tmp_wav
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            if not os.path.exists(tmp_wav) or os.path.getsize(tmp_wav) <= 64:
                return None
            with wave.open(tmp_wav, "rb") as wf:
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)
            try:
                os.remove(tmp_wav)
            except Exception:
                pass
            pcm_i16 = np.frombuffer(raw_bytes, dtype=np.int16)
            if len(pcm_i16) == 0:
                return None
            arr = (pcm_i16.astype(np.float32) / 32768.0).reshape(-1, 2)
            # Cắt bỏ 12ms đệm cuối file MP3 để tuyệt đối không có tiếng vang hay lặp âm cuối
            trim_samples = int(0.012 * sample_rate)
            if len(arr) > trim_samples * 4:
                arr = arr[:-trim_samples]
            return arr
        except Exception as e:
            print(f"[AutoVideoEditor] Lỗi giải mã PCM sạch: {e}")
            return None

    @staticmethod
    def _time_stretch_audio_ffmpeg(input_path: str, output_path: str, ratio: float) -> str:
        """Co giãn thời gian file Voice bằng FFmpeg atempo và lưu ra WAV PCM sạch (không lặp từ cuối)."""
        safe_ratio = float(np.clip(ratio, 0.50, 2.00))
        if abs(safe_ratio - 1.0) <= 0.012:
            return input_path
        out_wav = os.path.splitext(output_path)[0] + "_clean.wav"
        try:
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-filter:a", f"atempo={safe_ratio:.4f}",
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                out_wav
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            if os.path.exists(out_wav) and os.path.getsize(out_wav) > 512:
                return out_wav
        except Exception as e:
            print(f"[AutoVideoEditor] Lỗi atempo FFmpeg: {e}")
        return input_path

    def render_dub_only_video(
        self,
        video_paths=None,
        output_path: str = "output.mp4",
        work_dir: Optional[str] = None,
        timed_voice_items: Optional[List[Dict[str, Any]]] = None,
        master_audio_path: Optional[str] = None,
        bgm_paths: Optional[List[str]] = None,
        bgm_volume: float = 0.15,
        voice_volume: float = 1.0,
        keep_original_audio: bool = True,
        original_audio_volume: float = 1.0,
        auto_fit_voice_slot: bool = True,
        fast_stream_copy: bool = True,
        enhance_video_sharpness: bool = False,
        sharpness_amount: float = 1.25,
        graphics_preset: str = "stream_copy",
        progress_callback=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Ghép Voice vào video đã cắt sẵn và giữ nguyên 100% tiếng ASMR gốc của video."""
        if isinstance(video_paths, str):
            video_list = [video_paths] if os.path.exists(video_paths) else []
        elif isinstance(video_paths, list):
            video_list = [p for p in video_paths if isinstance(p, str) and os.path.exists(p)]
        else:
            vp_kw = kwargs.get("video_path", "")
            video_list = [vp_kw] if (vp_kw and os.path.exists(vp_kw)) else []

        if not video_list:
            raise ValueError("Không tìm thấy file Video đầu vào hợp lệ!")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        local_work_dir = work_dir if (work_dir and os.path.exists(work_dir)) else tempfile.mkdtemp(prefix="dub_work_")
        sample_rate = 44100

        if progress_callback:
            progress_callback(0.05, f"Đang đo thời lượng của {len(video_list)} tập video...")

        if len(video_list) == 1:
            master_input_video = video_list[0]
        else:
            concat_txt = os.path.join(local_work_dir, "concat_videos.txt")
            with open(concat_txt, "w", encoding="utf-8") as f:
                for vp in video_list:
                    safe_vp = os.path.abspath(vp).replace("\\", "/").replace("'", "'\\''")
                    f.write(f"file '{safe_vp}'\n")
            master_input_video = os.path.join(local_work_dir, "merged_precut_video.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", master_input_video],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800
            )
            if not os.path.exists(master_input_video) or os.path.getsize(master_input_video) < 1024:
                master_input_video = video_list[0]

        total_vid_dur = self._get_media_duration_fast(master_input_video)
        if total_vid_dur <= 0:
            clips_tmp = [VideoFileClip(p) for p in video_list]
            total_vid_dur = max(1.0, sum(float(c.duration) for c in clips_tmp))
            for c in clips_tmp:
                c.close()

        total_samples = int(np.ceil(total_vid_dur * sample_rate))
        memmap_path = os.path.join(local_work_dir, "master_timeline_memmap.dat")
        if total_vid_dur > 900.0:
            final_audio_buffer = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(total_samples, 2))
        else:
            final_audio_buffer = np.zeros((total_samples, 2), dtype=np.float32)

        asmr_loaded_ok = False
        try:
            speech_intervals: List[Tuple[float, float]] = []
            valid_voice_count = 0

            # 1. ĐẶT CÁC CÂU VOICE VÀO ĐÚNG MỐC GIÂY TRÊN TIMELINE
            if timed_voice_items and isinstance(timed_voice_items, list):
                total_v = len(timed_voice_items)
                for idx, item in enumerate(timed_voice_items):
                    if not isinstance(item, dict):
                        continue
                    ap = item.get("audio_path", "")
                    if not ap or not os.path.exists(ap):
                        continue

                    st_s = max(0.0, float(item.get("start", 0.0)))
                    en_s = max(st_s + 0.3, float(item.get("end", st_s + 5.0)))
                    slot_d = max(0.3, en_s - st_s)

                    if progress_callback:
                        progress_callback(0.10 + (idx / max(1, total_v)) * 0.38, f"Đang gắn Voice #{idx + 1}/{total_v} ({st_s:.1f}s -> {en_s:.1f}s)...")

                    use_p = ap
                    raw_d = self._get_media_duration_fast(ap) or slot_d
                    if auto_fit_voice_slot and raw_d > 0.2:
                        ratio_diff = raw_d / slot_d
                        if ratio_diff > 1.015 or (0.84 <= ratio_diff < 0.985):
                            target_ratio = float(np.clip(raw_d / (slot_d * 0.99), 0.84, 1.45))
                            stretched_p = os.path.join(local_work_dir, f"fit_voice_{idx:04d}.wav")
                            use_p = self._time_stretch_audio_ffmpeg(ap, stretched_p, target_ratio)

                    arr = self._decode_audio_to_pcm_clean(use_p, local_work_dir, f"v_{idx:04d}", sample_rate=sample_rate)
                    if arr is not None and len(arr) > 0:
                        arr = arr * float(voice_volume)
                        fade_len = min(int(0.025 * sample_rate), len(arr) // 4)
                        if fade_len > 10:
                            arr[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)[:, np.newaxis]
                            arr[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)[:, np.newaxis]

                        s_idx = int(round(st_s * sample_rate))
                        if s_idx < total_samples:
                            cp_len = min(len(arr), total_samples - s_idx)
                            final_audio_buffer[s_idx:s_idx + cp_len] += arr[:cp_len]
                            speech_intervals.append((st_s, st_s + (cp_len / sample_rate)))
                            valid_voice_count += 1

            elif master_audio_path and os.path.exists(master_audio_path):
                arr = self._decode_audio_to_pcm_clean(master_audio_path, local_work_dir, "master_v", sample_rate=sample_rate)
                if arr is not None and len(arr) > 0:
                    cp_len = min(total_samples, len(arr))
                    final_audio_buffer[:cp_len] += arr[:cp_len] * float(voice_volume)
                    speech_intervals.append((0.0, cp_len / sample_rate))
                    valid_voice_count = 1

            # 2. GIỮ NGUYÊN 100% TIẾNG ASMR GỐC BẰNG LUỒNG RAW PCM S16LE (KHÔNG GIỚI HẠN DUNG LƯỢNG)
            if keep_original_audio:
                if progress_callback:
                    progress_callback(0.55, "Đang trích xuất & giữ nguyên gốc 100% tiếng ASMR hiện trường...")
                orig_raw_pcm = os.path.join(local_work_dir, "orig_asmr_stream.raw")
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", master_input_video,
                        "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
                        "-ar", str(sample_rate), "-ac", "2", orig_raw_pcm
                    ],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800
                )
                if os.path.exists(orig_raw_pcm) and os.path.getsize(orig_raw_pcm) > 1024:
                    # Mặc định giữ 100% âm lượng gốc (1.0) ở các khoảng nghỉ ASMR
                    base_asmr_vol = float(original_audio_volume) if float(original_audio_volume) >= 0.5 else 1.0
                    chunk_frames = sample_rate * 60
                    cur_pos = 0
                    with open(orig_raw_pcm, "rb") as rf_pcm:
                        while cur_pos < total_samples:
                            need_frames = min(chunk_frames, total_samples - cur_pos)
                            raw_b = rf_pcm.read(need_frames * 4)
                            if not raw_b:
                                break
                            c_arr = (np.frombuffer(raw_b, dtype=np.int16).astype(np.float32) / 32768.0).reshape(-1, 2)
                            c_len = min(len(c_arr), total_samples - cur_pos)
                            asmr_part = c_arr[:c_len] * base_asmr_vol
                            if self.enable_audio_ducking and speech_intervals:
                                chunk_start_s = cur_pos / sample_rate
                                chunk_end_s = (cur_pos + c_len) / sample_rate
                                for s_sec, e_sec in speech_intervals:
                                    if e_sec >= chunk_start_s and s_sec <= chunk_end_s:
                                        local_s = max(0, int((s_sec - 0.10 - chunk_start_s) * sample_rate))
                                        local_e = min(c_len, int((e_sec + 0.15 - chunk_start_s) * sample_rate))
                                        if local_e > local_s:
                                            asmr_part[local_s:local_e] *= 0.65
                            final_audio_buffer[cur_pos:cur_pos + c_len] += asmr_part
                            cur_pos += c_len
                            asmr_loaded_ok = True
                    try:
                        os.remove(orig_raw_pcm)
                    except Exception:
                        pass

            # 3. HÒA ÂM NHẠC NỀN BGM (NẾU CÓ)
            if bgm_paths and len(bgm_paths) > 0:
                if progress_callback:
                    progress_callback(0.70, "Đang hòa âm nhạc nền BGM...")
                bgm_arrays = [
                    self._decode_audio_to_pcm_clean(bp, local_work_dir, f"bgm_{bi}", sample_rate=sample_rate)
                    for bi, bp in enumerate(bgm_paths) if os.path.exists(bp)
                ]
                bgm_arrays = [ba for ba in bgm_arrays if ba is not None and len(ba) > 0]
                if bgm_arrays:
                    bgm_cycle = np.vstack(bgm_arrays)
                    cycle_len = len(bgm_cycle)
                    cur_p = 0
                    while cur_p < total_samples and cycle_len > 0:
                        take = min(cycle_len, total_samples - cur_p)
                        b_part = (bgm_cycle[:take].copy()) * float(bgm_volume)
                        if self.enable_audio_ducking and speech_intervals:
                            c_start_s, c_end_s = cur_p / sample_rate, (cur_p + take) / sample_rate
                            for s_sec, e_sec in speech_intervals:
                                if e_sec >= c_start_s and s_sec <= c_end_s:
                                    l_s = max(0, int((s_sec - 0.20 - c_start_s) * sample_rate))
                                    l_e = min(take, int((e_sec + 0.25 - c_start_s) * sample_rate))
                                    if l_e > l_s:
                                        b_part[l_s:l_e] *= 0.32
                        final_audio_buffer[cur_p:cur_p + take] += b_part
                        cur_p += take

            # 4. GHI FILE MASTER WAV
            if progress_callback:
                progress_callback(0.82, "Đang kết xuất Master Audio WAV (ASMR + Voice)...")
            master_wav = os.path.join(local_work_dir, "master_full_timeline_dub.wav")
            with wave.open(master_wav, "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                step_frames = sample_rate * 60
                for s_i in range(0, total_samples, step_frames):
                    e_i = min(total_samples, s_i + step_frames)
                    pcm_chunk = (np.clip(final_audio_buffer[s_i:e_i], -1.0, 1.0) * 32767.0).astype(np.int16)
                    wf.writeframes(pcm_chunk.tobytes())

            if isinstance(final_audio_buffer, np.memmap):
                del final_audio_buffer

            # 5. ĐÓNG GÓI VIDEO MP4 (CÓ BỘ TRỘN AMIX DỰ PHÒNG ĐẢM BẢO 100% CÓ TIẾNG ASMR GỐC)
            needs_visual_filter = (not fast_stream_copy) or bool(enhance_video_sharpness) or (graphics_preset not in ["stream_copy", ""])
            mix_filter = f"[0:a]volume={max(0.9, float(original_audio_volume)):.2f}[a0];[1:a]volume=1.0[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"

            if not needs_visual_filter:
                if progress_callback:
                    progress_callback(0.90, f"⚡ Đang đóng gói Video + Tiếng ASMR gốc ({total_vid_dur:.1f}s)...")
                if keep_original_audio and not asmr_loaded_ok:
                    cmd = ["ffmpeg", "-y", "-i", master_input_video, "-i", master_wav, "-filter_complex", mix_filter, "-map", "0:v:0", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", output_path]
                else:
                    cmd = ["ffmpeg", "-y", "-i", master_input_video, "-i", master_wav, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", output_path]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600)
                if res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                    if progress_callback:
                        progress_callback(1.0, "Hoàn tất ghép Voice & giữ nguyên 100% tiếng ASMR gốc!")
                    return {"sequence_valid": True, "duration_synced": True, "video_duration_sec": round(total_vid_dur, 2), "shot_match_rate_percent": 100.0, "total_clips_assembled": max(1, valid_voice_count)}

            preset_lookup = {"1080p_ultra_sharp": (1920, 1080, 16, 1.30), "2k_qhd_master": (2560, 1440, 15, 1.40), "4k_uhd_cinema": (3840, 2160, 14, 1.50)}
            out_w, out_h, use_crf, preset_sharp = preset_lookup.get(graphics_preset, (self.target_w, self.target_h, self.crf_quality, sharpness_amount))
            vf_filters = [f"scale={out_w}:{out_h}:flags=lanczos"]
            sharp_val = max(preset_sharp if needs_visual_filter else 0.0, self.sharpness_boost)
            if sharp_val > 0.05:
                vf_filters.append(f"unsharp=5:5:{sharp_val:.2f}:5:5:0.0")
            if abs(self.brightness) > 0.5 or abs(self.contrast - 1.0) > 0.02 or abs(self.saturation - 1.0) > 0.03:
                vf_filters.append(f"eq=brightness={float(np.clip(self.brightness / 100.0, -0.5, 0.5)):.3f}:contrast={self.contrast:.2f}:saturation={self.saturation:.2f}")

            cmd_hq = ["ffmpeg", "-y", "-i", master_input_video, "-i", master_wav, "-map", "0:v:0", "-map", "1:a:0", "-vf", ",".join(vf_filters), "-c:v", self.video_codec, "-preset", self.render_preset, "-crf", str(use_crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "320k", output_path]
            res_hq = subprocess.run(cmd_hq, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=14400)
            if res_hq.returncode != 0 and self.video_codec != "libx264":
                cmd_hq[cmd_hq.index("-c:v") + 1] = "libx264"
                subprocess.run(cmd_hq, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=14400)

            if progress_callback:
                progress_callback(1.0, "Hoàn tất xuất video Siêu Nét kèm tiếng ASMR gốc!")
            return {"sequence_valid": True, "duration_synced": True, "video_duration_sec": round(total_vid_dur, 2), "shot_match_rate_percent": 100.0, "total_clips_assembled": max(1, valid_voice_count)}
        finally:
            try:
                import gc
                gc.collect()
                if os.path.exists(memmap_path):
                    os.remove(memmap_path)
            except Exception:
                pass

    def _build_vignette_mask(self) -> np.ndarray:
        y, x = np.ogrid[:self.target_h, :self.target_w]
        cy, cx = self.target_h / 2.0, self.target_w / 2.0
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist
        mask = 1.0 - 0.38 * (dist ** 2)
        return np.clip(mask, 0.60, 1.0).astype(np.float32)[:, :, np.newaxis]

    def _build_watermark_overlay(self) -> Optional[Tuple[int, int, np.ndarray, np.ndarray]]:
        try:
            font_size = max(18, int(min(self.target_w, self.target_h) * 0.028))
            try:
                font = PIL.ImageFont.truetype("arialbd.ttf", font_size)
            except Exception:
                try:
                    font = PIL.ImageFont.truetype("arial.ttf", font_size)
                except Exception:
                    font = PIL.ImageFont.load_default()

            dummy = PIL.Image.new("RGBA", (10, 10))
            draw = PIL.ImageDraw.Draw(dummy)
            bbox = draw.textbbox((0, 0), self.watermark_text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            pad_x, pad_y = 18, 10
            box_w = min(self.target_w - 20, tw + pad_x * 2)
            box_h = min(self.target_h - 20, th + pad_y * 2)

            overlay_img = PIL.Image.new("RGBA", (box_w, box_h), (0, 0, 0, 115))
            draw = PIL.ImageDraw.Draw(overlay_img)
            draw.text((pad_x, pad_y - 2), self.watermark_text, font=font, fill=(255, 255, 255, 235))

            arr = np.array(overlay_img)
            rgb = arr[:, :, :3].astype(np.float32)
            alpha = (arr[:, :, 3:4].astype(np.float32)) / 255.0

            margin_x = int(self.target_w * 0.03)
            margin_y = int(self.target_h * 0.08) if self.enable_letterbox else int(self.target_h * 0.03)

            if self.watermark_pos == "top_left":
                x0, y0 = margin_x, margin_y
            elif self.watermark_pos == "top_right":
                x0, y0 = self.target_w - box_w - margin_x, margin_y
            elif self.watermark_pos == "bottom_left":
                x0, y0 = margin_x, self.target_h - box_h - margin_y
            else:
                x0, y0 = self.target_w - box_w - margin_x, self.target_h - box_h - margin_y

            return max(0, x0), max(0, y0), rgb, alpha
        except Exception:
            return None

    def _build_color_lut(self, tint_rgb: Optional[np.ndarray]) -> Optional[np.ndarray]:
        needs_lut = (
            tint_rgb is not None
            or abs(self.brightness) > 0.5
            or abs(self.contrast - 1.0) > 0.01
        )
        if not needs_lut:
            return None

        base = np.arange(256, dtype=np.float32)
        lut = np.stack([base, base, base], axis=-1)

        if abs(self.contrast - 1.0) > 0.01 or abs(self.brightness) > 0.5:
            lut = (lut - 128.0) * self.contrast + 128.0 + self.brightness

        if tint_rgb is not None and self.lut_opacity > 0:
            lut = lut * (1.0 - self.lut_opacity) + tint_rgb * self.lut_opacity

        return np.clip(lut, 0, 255).astype(np.uint8).reshape(1, 256, 3)

    def build_auto_edl(
        self,
        scenes: List[Dict[str, Any]],
        clip_duration: float = 3.0,
        target_total_duration: float = 0.0,
        edit_mode: str = "highlight"
    ) -> List[Dict[str, Any]]:
        if not scenes:
            raise ValueError("Danh sách cảnh đầu vào trống!")

        if edit_mode == "sequential":
            ordered_scenes = [s.copy() for s in scenes]
        elif edit_mode == "random":
            ordered_scenes = [s.copy() for s in scenes]
            random.shuffle(ordered_scenes)
        else:
            stage_pools: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: [], 4: []}
            for sc in scenes:
                st = sc.get("workflow_stage", 2)
                stage_pools.setdefault(st, []).append(sc.copy())
            ordered_scenes = []
            for st in [1, 2, 3, 4]:
                pool = stage_pools.get(st, [])
                if len(pool) > 1:
                    random.shuffle(pool)
                ordered_scenes.extend(pool)
            if not ordered_scenes:
                ordered_scenes = [s.copy() for s in scenes]

        edl: List[Dict[str, Any]] = []
        curr_time = 0.0
        idx = 0

        if target_total_duration > 0:
            max_time = target_total_duration
        else:
            max_time = sum(max(0.8, min(clip_duration, s["duration"] / self.default_speed)) for s in ordered_scenes)

        while curr_time < max_time - 0.15:
            sc = ordered_scenes[idx % len(ordered_scenes)].copy()
            natural_dur = max(0.5, sc["duration"] / max(0.1, self.default_speed))
            seg_dur = min(clip_duration, natural_dur) if clip_duration > 0 else natural_dur
            seg_dur = max(0.5, seg_dur)

            if target_total_duration > 0 and curr_time + seg_dur > target_total_duration:
                seg_dur = max(0.4, target_total_duration - curr_time)

            v_seg = {
                "index": idx,
                "start": round(curr_time, 2),
                "end": round(curr_time + seg_dur, 2),
                "duration": round(seg_dur, 2),
                "text": f"[Cảnh #{idx + 1}] {sc.get('content_type', 'Thi công')} ({sc.get('shot_type', 'Shot')})",
                "content_type": sc.get("content_type", "Thao tác"),
                "preferred_shot": sc.get("shot_type", "Close-up"),
                "workflow_stage": sc.get("workflow_stage", 2)
            }
            edl.append({
                "voice_segment": v_seg,
                "scene": sc,
                "assigned_stage": sc.get("workflow_stage", 2)
            })
            curr_time += seg_dur
            idx += 1

            if target_total_duration <= 0 and idx >= len(ordered_scenes):
                break

        return edl

    def match_scenes_to_voice(
        self,
        voice_segments: List[Dict[str, Any]],
        scenes: List[Dict[str, Any]],
        enable_smart_shuffle: bool = True
    ) -> List[Dict[str, Any]]:
        if not scenes:
            raise ValueError("Danh sách cảnh đầu vào trống!")

        stage_pools: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: [], 4: []}
        for sc in scenes:
            st = sc.get("workflow_stage", 2)
            stage_pools.setdefault(st, []).append(sc.copy())

        if enable_smart_shuffle:
            for st in stage_pools:
                if len(stage_pools[st]) > 2:
                    random.shuffle(stage_pools[st])

        edit_decision_list: List[Dict[str, Any]] = []
        current_min_stage = 1

        for v_seg in voice_segments:
            desired_stage = max(current_min_stage, v_seg.get("workflow_stage", 2))
            current_min_stage = desired_stage

            chosen_scene = self._select_best_scene_for_voice(
                v_seg,
                stage_pools,
                desired_stage,
                scenes
            )

            edit_decision_list.append({
                "voice_segment": v_seg,
                "scene": chosen_scene,
                "assigned_stage": desired_stage
            })

        return edit_decision_list

    def _select_best_scene_for_voice(
        self,
        v_seg: Dict[str, Any],
        stage_pools: Dict[int, List[Dict[str, Any]]],
        desired_stage: int,
        all_scenes_fallback: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        candidates = stage_pools.get(desired_stage, [])
        if not candidates:
            for next_st in range(desired_stage, 5):
                if stage_pools.get(next_st):
                    candidates = stage_pools[next_st]
                    break
        if not candidates:
            candidates = all_scenes_fallback

        best_scene = candidates[0]
        best_score = -999.0

        for sc in candidates:
            score = 0.0
            if not sc.get("used", False):
                score += 50.0
            if sc.get("content_type") == v_seg.get("content_type"):
                score += 30.0
            if sc.get("shot_type") == v_seg.get("preferred_shot"):
                score += 20.0

            ideal_raw_len = v_seg["duration"] * self.default_speed
            len_diff = abs(sc["duration"] - ideal_raw_len)
            score -= min(15.0, len_diff * 0.8)

            if score > best_score:
                best_score = score
                best_scene = sc

        best_scene["used"] = True
        return best_scene

    def _format_aspect_and_motion(self, clip: VideoFileClip, index: int) -> VideoFileClip:
        if self.fit_mode == "blur_bg":
            tw, th = self.target_w, self.target_h
            scale_fg = min(tw / max(1, clip.w), th / max(1, clip.h)) * self.zoom_factor
            fg_w = min(tw, max(2, int(clip.w * scale_fg)))
            fg_h = min(th, max(2, int(clip.h * scale_fg)))
            x_off = (tw - fg_w) // 2
            y_off = (th - fg_h) // 2

            def make_blur_frame(frame: np.ndarray) -> np.ndarray:
                small_bg = cv2.resize(frame, (tw // 6, th // 6), interpolation=cv2.INTER_LINEAR)
                blurred_bg = cv2.GaussianBlur(small_bg, (15, 15), 0)
                bg = cv2.resize(blurred_bg, (tw, th), interpolation=cv2.INTER_LINEAR)
                bg = (bg.astype(np.float32) * 0.55).astype(np.uint8)
                fg = cv2.resize(frame, (fg_w, fg_h), interpolation=cv2.INTER_LANCZOS4)
                bg[y_off:y_off + fg_h, x_off:x_off + fg_w] = fg
                return bg

            clip = clip.fl_image(make_blur_frame)
        else:
            scale_w = (self.target_w * self.zoom_factor) / max(1, clip.w)
            scale_h = (self.target_h * self.zoom_factor) / max(1, clip.h)
            scale = max(scale_w, scale_h)

            zoomed_w = max(self.target_w, int(clip.w * scale))
            zoomed_h = max(self.target_h, int(clip.h * scale))
            clip = clip.resize(newsize=(zoomed_w, zoomed_h))

            max_dx = (zoomed_w - self.target_w) // 2
            max_dy = (zoomed_h - self.target_h) // 2

            offsets = [
                (0, 0),
                (-int(max_dx * 0.65), 0),
                (int(max_dx * 0.65), 0),
                (0, -int(max_dy * 0.65)),
                (0, int(max_dy * 0.65)),
            ]
            dx, dy = offsets[index % len(offsets)]

            clip = clip.crop(
                x_center=(zoomed_w // 2) + dx,
                y_center=(zoomed_h // 2) + dy,
                width=self.target_w,
                height=self.target_h
            )

        if random.random() < self.flip_probability:
            clip = clip.fx(vfx.mirror_x)

        return clip

    def _apply_graphics_and_color(self, clip: VideoFileClip, band_index: int) -> VideoFileClip:
        tint_rgb = None
        if self.color_style == "auto_lut":
            _, tint_rgb = self.NATURAL_COLOR_OVERLAYS[band_index % len(self.NATURAL_COLOR_OVERLAYS)]
        elif self.color_style in self.COLOR_PRESETS:
            tint_rgb = self.COLOR_PRESETS[self.color_style]

        lut_table = self._build_color_lut(tint_rgb)
        bar_h = int(self.target_h * self.letterbox_ratio) if self.enable_letterbox else 0
        vig_mask = self._vignette_mask
        wm = self._watermark_overlay
        sat = self.saturation
        sharp = self.sharpness_boost

        if (
            lut_table is None
            and abs(sat - 1.0) <= 0.02
            and sharp <= 0.05
            and vig_mask is None
            and bar_h == 0
            and wm is None
        ):
            return clip

        def process_frame_fast(frame: np.ndarray) -> np.ndarray:
            res = frame if lut_table is None else cv2.LUT(frame, lut_table)

            if sharp > 0.05:
                blurred = cv2.GaussianBlur(res, (0, 0), 2.0)
                res = cv2.addWeighted(res, 1.0 + sharp * 0.65, blurred, -sharp * 0.65, 0)

            if abs(sat - 1.0) > 0.02:
                hsv = cv2.cvtColor(res, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat, 0, 255)
                res = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            if vig_mask is not None:
                res = np.clip(res.astype(np.float32) * vig_mask, 0, 255).astype(np.uint8)
            elif res is frame:
                res = frame.copy()

            if bar_h > 0:
                res[:bar_h, :, :] = 0
                res[-bar_h:, :, :] = 0

            if wm is not None:
                x0, y0, wm_rgb, wm_alpha = wm
                h_wm, w_wm = wm_rgb.shape[:2]
                roi = res[y0:y0 + h_wm, x0:x0 + w_wm].astype(np.float32)
                res[y0:y0 + h_wm, x0:x0 + w_wm] = (roi * (1.0 - wm_alpha) + wm_rgb * wm_alpha).astype(np.uint8)

            return res

        return clip.fl_image(process_frame_fast)

    def verify_sync_and_sequence(
        self,
        edl: List[Dict[str, Any]],
        final_video_duration: float,
        voice_duration: float
    ) -> Dict[str, Any]:
        stages = [
            int(item["assigned_stage"]) if isinstance(item.get("assigned_stage"), (int, float)) else 2
            for item in edl
        ]
        is_monotonic = all(stages[i] <= stages[i + 1] for i in range(len(stages) - 1)) if len(stages) > 1 else True

        duration_drift = abs(final_video_duration - voice_duration)
        is_duration_synced = duration_drift < 0.35

        shot_matches = sum(
            1 for item in edl
            if item.get("scene", {}).get("shot_type") == item.get("voice_segment", {}).get("preferred_shot", item.get("scene", {}).get("shot_type"))
        )
        shot_match_rate = (shot_matches / max(1, len(edl))) * 100.0

        return {
            "sequence_valid": is_monotonic,
            "duration_synced": is_duration_synced,
            "video_duration_sec": round(final_video_duration, 2),
            "voice_duration_sec": round(voice_duration, 2),
            "drift_sec": round(duration_drift, 3),
            "shot_match_rate_percent": round(shot_match_rate, 1),
            "total_clips_assembled": len(edl)
        }

    def _expand_edl_with_broll_gaps(
        self,
        edl: List[Dict[str, Any]],
        all_scenes_pool: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        if not edl:
            return []

        pool = all_scenes_pool if (all_scenes_pool and len(all_scenes_pool) > 0) else [item["scene"] for item in edl]
        expanded: List[Dict[str, Any]] = []
        curr_t = 0.0
        pool_cursor = 0

        for idx, item in enumerate(edl):
            v_seg = item["voice_segment"]
            seg_start = float(v_seg.get("start", curr_t))
            seg_end = float(v_seg.get("end", seg_start + float(v_seg.get("duration", 3.0))))
            seg_dur = max(0.3, seg_end - seg_start)

            gap = seg_start - curr_t
            if gap > 0.6 and pool:
                rem_gap = gap
                while rem_gap > 0.35:
                    step_dur = min(rem_gap, 5.0 if rem_gap > 6.5 else rem_gap)
                    broll_sc = pool[pool_cursor % len(pool)].copy()
                    pool_cursor += 1
                    expanded.append({
                        "voice_segment": {
                            "index": -1,
                            "start": round(curr_t, 3),
                            "end": round(curr_t + step_dur, 3),
                            "duration": round(step_dur, 3),
                            "text": "[B-Roll Thi Công]",
                            "preferred_shot": broll_sc.get("shot_type", "Wide shot"),
                            "content_type": broll_sc.get("content_type", "Thao tác"),
                            "workflow_stage": item.get("assigned_stage", 2)
                        },
                        "scene": broll_sc,
                        "assigned_stage": item.get("assigned_stage", 2),
                        "is_broll_gap": True
                    })
                    curr_t += step_dur
                    rem_gap = seg_start - curr_t

            main_copy = dict(item)
            main_copy["is_broll_gap"] = False
            expanded.append(main_copy)
            curr_t = seg_start + seg_dur

        return expanded

    def render_final_video(
        self,
        edl: List[Dict[str, Any]],
        audio_path: Optional[str],
        output_path: str,
        bgm_paths: Optional[List[str]] = None,
        bgm_volume: float = 0.15,
        voice_volume: float = 1.0,
        keep_original_audio: bool = False,
        original_audio_volume: float = 0.25,
        progress_callback=None,
        all_scenes_pool: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        voice_audio = None
        bgm_audio = None
        target_total_duration = float(edl[-1]["voice_segment"]["end"]) if edl else 10.0

        if audio_path and os.path.exists(audio_path):
            is_script_file = audio_path.lower().endswith(('.md', '.srt', '.txt'))
            if not is_script_file:
                voice_audio = AudioFileClip(audio_path)
                if abs(voice_volume - 1.0) > 0.02:
                    voice_audio = voice_audio.volumex(voice_volume)
                target_total_duration = max(target_total_duration, float(voice_audio.duration))

        render_edl = self._expand_edl_with_broll_gaps(edl, all_scenes_pool=all_scenes_pool)
        has_broll_gaps = any(item.get("is_broll_gap", False) for item in render_edl)

        source_clips_cache: Dict[str, VideoFileClip] = {}
        processed_subclips: List[VideoFileClip] = []
        total_items = len(render_edl)
        num_color_bands = self.color_segments_count

        try:
            for idx, item in enumerate(render_edl):
                sc = item["scene"]
                v_seg = item["voice_segment"]
                target_clip_dur = max(0.3, float(v_seg["duration"]))
                is_broll = bool(item.get("is_broll_gap", False))

                v_path = sc["video_path"]
                if v_path not in source_clips_cache:
                    raw_clip = VideoFileClip(v_path)
                    if not keep_original_audio:
                        raw_clip = raw_clip.without_audio()
                    source_clips_cache[v_path] = raw_clip

                base_video = source_clips_cache[v_path]
                raw_start = float(sc.get("start", sc.get("start_time", 0.0)))
                raw_end = float(sc.get("end", sc.get("end_time", raw_start + float(sc.get("duration", 5.0)))))
                raw_start_clamped = max(0.0, min(raw_start, max(0.0, base_video.duration - 0.5)))
                raw_end_clamped = max(raw_start_clamped + 0.3, min(raw_end, base_video.duration))
                raw_dur = max(0.2, raw_end_clamped - raw_start_clamped)

                desired_raw_span = target_clip_dur * self.default_speed
                if raw_dur > desired_raw_span * 1.25:
                    margin = (raw_dur - desired_raw_span) / 2.0
                    sub = base_video.subclip(raw_start_clamped + margin, raw_end_clamped - margin)
                else:
                    sub = base_video.subclip(raw_start_clamped, raw_end_clamped)

                speed_factor = sub.duration / target_clip_dur
                speed_factor = float(np.clip(speed_factor, self.min_speed, self.max_speed))
                if abs(speed_factor - 1.0) > 0.02:
                    sub = sub.fx(vfx.speedx, speed_factor)

                if sub.duration > target_clip_dur:
                    sub = sub.subclip(0, target_clip_dur)
                elif sub.duration < target_clip_dur - 0.05:
                    sub = afx.audio_loop(sub, duration=target_clip_dur) if hasattr(sub, "audio") and sub.audio else sub.fx(vfx.loop, duration=target_clip_dur)

                if keep_original_audio and sub.audio is not None:
                    asmr_vol = original_audio_volume * (1.5 if (self.enable_audio_ducking and is_broll) else 1.0)
                    sub = sub.volumex(asmr_vol)

                sub = self._format_aspect_and_motion(sub, idx)

                band_index = int((idx / max(1, total_items)) * num_color_bands)
                sub = self._apply_graphics_and_color(sub, band_index)

                if self.enable_crossfade and idx > 0 and target_clip_dur > self.crossfade_duration * 2.5:
                    sub = sub.fx(vfx.fadein, self.crossfade_duration)

                processed_subclips.append(sub)

                if progress_callback:
                    progress_callback((idx + 1) / total_items * 0.35, f"Đang dựng cảnh {idx + 1}/{total_items}")

            concat_method = "compose" if self.enable_crossfade else "chain"
            assembled_video = concatenate_videoclips(processed_subclips, method=concat_method)

            if 0.08 < abs(assembled_video.duration - target_total_duration) < (target_total_duration * 0.08):
                final_speed_ratio = assembled_video.duration / target_total_duration
                assembled_video = assembled_video.fx(vfx.speedx, final_speed_ratio)

            if self.enable_fade_inout and target_total_duration > 2.5:
                assembled_video = assembled_video.fx(vfx.fadein, 0.35).fx(vfx.fadeout, 0.35)

            audio_tracks = []
            if keep_original_audio and assembled_video.audio is not None:
                audio_tracks.append(assembled_video.audio)
            if voice_audio is not None:
                audio_tracks.append(voice_audio)

            if bgm_paths and len(bgm_paths) > 0:
                valid_bgm_clips = []
                for p in bgm_paths:
                    if os.path.exists(p):
                        valid_bgm_clips.append(AudioFileClip(p))

                if valid_bgm_clips:
                    merged_bgm = concatenate_audioclips(valid_bgm_clips)
                    if merged_bgm.duration < target_total_duration:
                        bgm_audio = afx.audio_loop(merged_bgm, duration=target_total_duration)
                    else:
                        bgm_audio = merged_bgm.subclip(0, target_total_duration)

                    if self.enable_audio_ducking and has_broll_gaps:
                        bgm_segments = []
                        for item in render_edl:
                            v_s = float(item["voice_segment"]["start"])
                            v_e = min(bgm_audio.duration, float(item["voice_segment"]["end"]))
                            if v_e > v_s:
                                seg_a = bgm_audio.subclip(v_s, v_e)
                                vol_mult = 1.65 if item.get("is_broll_gap", False) else 1.0
                                seg_a = seg_a.volumex(bgm_volume * vol_mult)
                                bgm_segments.append(seg_a)
                        if bgm_segments:
                            bgm_audio = concatenate_audioclips(bgm_segments)
                    else:
                        bgm_audio = bgm_audio.volumex(bgm_volume)

                    if target_total_duration > 3.0:
                        bgm_audio = bgm_audio.fx(afx.audio_fadeout, 1.0)
                    audio_tracks.append(bgm_audio)

            final_dur = min(assembled_video.duration, target_total_duration) if assembled_video.duration > target_total_duration else assembled_video.duration
            if audio_tracks:
                final_audio = CompositeAudioClip(audio_tracks).set_duration(final_dur)
                assembled_video = assembled_video.set_audio(final_audio)
            else:
                samples_count = int(max(1.0, final_dur) * 44100)
                silent_audio = AudioArrayClip(np.zeros((samples_count, 2), dtype=np.float32), fps=44100)
                assembled_video = assembled_video.set_audio(silent_audio)

            assembled_video = assembled_video.set_duration(final_dur)

            audit_report = self.verify_sync_and_sequence(
                edl,
                assembled_video.duration,
                final_dur
            )

            def on_encode_step(encode_ratio: float):
                if progress_callback:
                    total_ratio = 0.35 + encode_ratio * 0.65
                    progress_callback(total_ratio, f"Đang kết xuất MP4 siêu nét ({int(encode_ratio * 100)}%)")

            custom_logger = _RenderProgressLogger(on_encode_step) if _RenderProgressLogger else None
            ffmpeg_extra = ["-crf", str(self.crf_quality), "-pix_fmt", "yuv420p"]

            try:
                assembled_video.write_videofile(
                    output_path,
                    fps=self.target_fps,
                    codec=self.video_codec,
                    audio_codec="aac",
                    audio_bitrate="256k",
                    preset=self.render_preset,
                    threads=self.cpu_threads,
                    ffmpeg_params=ffmpeg_extra,
                    logger=custom_logger
                )
            except Exception as gpu_err:
                if self.video_codec != "libx264":
                    assembled_video.write_videofile(
                        output_path,
                        fps=self.target_fps,
                        codec="libx264",
                        audio_codec="aac",
                        audio_bitrate="256k",
                        preset="superfast",
                        threads=self.cpu_threads,
                        ffmpeg_params=ffmpeg_extra,
                        logger=custom_logger
                    )
                else:
                    raise gpu_err

            if progress_callback:
                progress_callback(1.0, "Hoàn tất xuất video 100%!")

            return audit_report

        finally:
            if voice_audio is not None:
                voice_audio.close()
            if bgm_audio is not None:
                bgm_audio.close()
            for clip in source_clips_cache.values():
                clip.close()