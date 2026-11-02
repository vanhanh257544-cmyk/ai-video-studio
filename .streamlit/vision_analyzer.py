import os
import hashlib
import subprocess
from typing import List, Dict, Any, Optional, Tuple
import cv2  # type: ignore
import numpy as np  # type: ignore

from video_editor import AutoVideoEditor  # type: ignore


class VisionAnalyzer:
    """
    Chuyên trách trích xuất lưới 6 khung hình (Filmstrip 6-Frame) tại từng mốc thời gian của kịch bản
    và dò tìm điểm chạm chuyển động mạnh nhất (Visual Action Anchor) để đồng bộ từ khóa hành động.
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        temp_dir: str = "studio_cache/thumbnails",
        use_cache: bool = True,
        max_workers: int = 4
    ):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        self.gemini_api_key = gemini_api_key
        self.use_cache = use_cache
        self.max_workers = max_workers

    @staticmethod
    def get_video_timeline_info(video_paths: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
        """Tính tổng thời lượng và mốc bắt đầu/kết thúc của từng tập video đã cắt sẵn."""
        segments_meta = []
        cum_t = 0.0
        for vp in video_paths:
            if not vp or not os.path.exists(vp):
                continue
            dur = AutoVideoEditor._get_media_duration_fast(vp)
            if dur > 0:
                segments_meta.append({
                    "video_path": vp,
                    "start_global": cum_t,
                    "end_global": cum_t + dur,
                    "duration": dur
                })
                cum_t += dur
        return (cum_t if cum_t > 0 else 120.0), segments_meta

    @staticmethod
    def compute_video_fingerprint(video_paths: List[str]) -> str:
        """Tạo chữ ký nhận diện bộ video để lưu cache ảnh 6 khung hình không bị trùng."""
        if not video_paths:
            return "no_video"
        parts = [f"{os.path.basename(p)}_{os.path.getsize(p)}" for p in video_paths if p and os.path.exists(p)]
        return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:16] if parts else "no_video"

    def extract_slot_filmstrip(
        self,
        video_path_or_list: Any,
        start_sec: float,
        end_sec: float,
        idx_tag: int
    ) -> Tuple[str, str]:
        """
        Cắt 6 khung hình liên tiếp trong khoảng [start_sec, end_sec] của câu kịch bản
        và ghép thành 1 tấm ảnh lưới 2x3 (Filmstrip) để bạn và AI đối soát dụng cụ/hành động.
        """
        if isinstance(video_path_or_list, list):
            v_list = [p for p in video_path_or_list if isinstance(p, str) and os.path.exists(p)]
        elif isinstance(video_path_or_list, str) and os.path.exists(video_path_or_list):
            v_list = [video_path_or_list]
        else:
            v_list = []

        if not v_list:
            return "", ""

        sig_v = self.compute_video_fingerprint(v_list)[:8]
        thumb_jpg = os.path.join(self.temp_dir, f"grid6_thumb_{sig_v}_{idx_tag:04d}_{int(start_sec * 10)}_{int(end_sec * 10)}.jpg")
        strip_jpg = os.path.join(self.temp_dir, f"grid6_strip_{sig_v}_{idx_tag:04d}_{int(start_sec * 10)}_{int(end_sec * 10)}.jpg")

        if self.use_cache and os.path.exists(thumb_jpg) and os.path.exists(strip_jpg):
            return thumb_jpg, strip_jpg

        _, seg_metas = self.get_video_timeline_info(v_list)
        if not seg_metas:
            return "", ""

        dur = max(1.0, end_sec - start_sec)
        sample_times = [start_sec + dur * r for r in (0.06, 0.23, 0.41, 0.59, 0.77, 0.94)]
        frames = []

        for i_t, gt in enumerate(sample_times):
            chosen_seg = next((sm for sm in seg_metas if sm["start_global"] <= gt <= sm["end_global"]), seg_metas[-1])
            local_t = max(0.0, min(chosen_seg["duration"] - 0.05, gt - chosen_seg["start_global"]))
            tmp_f = os.path.join(self.temp_dir, f"_tmp_g6_{sig_v}_{idx_tag}_{i_t}.jpg")
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-ss", f"{local_t:.3f}",
                        "-i", chosen_seg["video_path"],
                        "-frames:v", "1",
                        "-vf", "scale=480:270:flags=lanczos",
                        "-q:v", "2", tmp_f
                    ],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15
                )
                if os.path.exists(tmp_f):
                    fr = cv2.imread(tmp_f)
                    os.remove(tmp_f)
                    if fr is not None:
                        cv2.rectangle(fr, (6, 6), (245, 36), (0, 0, 0), -1)
                        cv2.putText(
                            fr, f"Hinh #{i_t+1} (+{gt - start_sec:.1f}s)",
                            (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA
                        )
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

    def detect_visual_anchor(
        self,
        video_paths: List[str],
        start_sec: float,
        end_sec: float,
        target_wps: float = 2.5
    ) -> Dict[str, Any]:
        """Dò tìm giây có chuyển động thao tác rõ nhất trong khung giờ để đặt Động Từ Hành Động."""
        slot_dur = max(4.0, end_sec - start_sec)
        min_x = 2
        max_x = max(2, min(8, int(round(slot_dur * 0.55))))
        default_x = max(2, min(max_x, int(round(slot_dur * 0.35))))

        _, seg_metas = self.get_video_timeline_info(video_paths)
        if not seg_metas:
            return {
                "anchor_sec_in_clip": default_x,
                "anchor_global_sec": round(start_sec + default_x, 3),
                "anchor_word_index": max(4, int(round(default_x * target_wps))),
                "motion_peak": 5.0
            }

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

        return {
            "anchor_sec_in_clip": int(best_x),
            "anchor_global_sec": round(start_sec + best_x, 3),
            "anchor_word_index": max(4, int(round(default_x * target_wps))),
            "motion_peak": round(max(1.0, max_diff), 2)
        }