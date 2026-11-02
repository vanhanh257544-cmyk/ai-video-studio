import os
import re
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore

try:
    import google.generativeai as genai  # type: ignore
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False


class VisionAnalyzer:
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        temp_dir: str = "temp",
        use_cache: bool = True,
        max_workers: int = 6
    ):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        self.use_gemini = False
        self.use_cache = use_cache
        self.max_workers = max(2, min(12, max_workers))
        self.last_was_cached = False
        self.cache_file = os.path.join(self.temp_dir, "scene_analysis_cache.json")

        if gemini_api_key and GEMINI_AVAILABLE and genai is not None:
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel("gemini-3.1-pro-preview")
            self.use_gemini = True

    def _load_cache_db(self) -> Dict[str, Any]:
        if self.use_cache and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache_db(self, db: Dict[str, Any]):
        if not self.use_cache:
            return
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(db, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _get_video_signature(self, video_path: str) -> str:
        base_name = re.sub(r'^\d{3,5}_\d+_', '', os.path.basename(video_path))
        base_name = re.sub(r'^\d{3,5}_', '', base_name)
        try:
            f_size = os.path.getsize(video_path)
        except Exception:
            f_size = 0
        mode_tag = "gemini31" if self.use_gemini else "opencv"
        return f"{base_name}_{f_size}_{mode_tag}"

    def analyze_videos(
        self,
        video_paths: List[str],
        max_scene_duration: float = 120.0,
        chunk_split_duration: float = 20.0,
        is_reference: bool = False,
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        all_valid_scenes: List[Dict[str, Any]] = []
        cache_db = self._load_cache_db()
        cache_updated = False
        self.last_was_cached = False

        for file_order, video_path in enumerate(video_paths):
            if not os.path.exists(video_path):
                continue

            sig = self._get_video_signature(video_path)
            if self.use_cache and sig in cache_db:
                cached_scenes = cache_db[sig]
                if cached_scenes and all(os.path.exists(s.get("thumb_path", "")) for s in cached_scenes):
                    if progress_callback:
                        progress_callback(1.0, f"⚡ Nạp tức thì từ Smart Cache: {os.path.basename(video_path)}")
                    for s in cached_scenes:
                        s_copy = dict(s)
                        s_copy["video_path"] = video_path
                        s_copy["file_order"] = file_order
                        s_copy["used"] = False
                        all_valid_scenes.append(s_copy)
                    self.last_was_cached = True
                    continue

            if progress_callback:
                progress_callback(0.10, f"Đang tách cảnh siêu tốc: {os.path.basename(video_path)}...")

            raw_scenes = self._detect_and_split_scenes(
                video_path,
                max_scene_duration=max_scene_duration,
                chunk_split_duration=chunk_split_duration
            )

            min_dur = 0.6 if is_reference else 1.2
            valid_intervals = [
                (idx, start_sec, end_sec)
                for idx, (start_sec, end_sec) in enumerate(raw_scenes)
                if (end_sec - start_sec) >= min_dur
            ]
            total_raw = max(1, len(raw_scenes))

            if progress_callback:
                progress_callback(0.35, f"Đang chấm điểm độ nét & trích xuất Filmstrip ({len(valid_intervals)} cảnh)...")

            def _eval_worker(item_tuple):
                c_idx, s_sec, e_sec = item_tuple
                m = self._evaluate_scene_quality(video_path, s_sec, e_sec, None)
                return (c_idx, s_sec, e_sec, m)

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                eval_results = list(pool.map(_eval_worker, valid_intervals))

            filtered_candidates = []
            last_hist = None
            for c_idx, s_sec, e_sec, metrics in eval_results:
                if not metrics["is_valid"]:
                    continue
                hist = metrics["histogram"]
                if not is_reference and last_hist is not None and hist is not None:
                    similarity = cv2.compareHist(last_hist, hist, cv2.HISTCMP_CORREL)
                    if similarity > 0.96 and metrics["motion_score"] < 3.0:
                        continue
                last_hist = hist
                filtered_candidates.append((c_idx, s_sec, e_sec, metrics))

            if progress_callback:
                progress_callback(0.65, f"Đang phân tích hành động & góc máy ({len(filtered_candidates)} cảnh chuẩn)...")

            def _classify_worker(cand_tuple):
                c_idx, s_sec, e_sec, metrics = cand_tuple
                rel_pos = c_idx / max(1, total_raw - 1) if total_raw > 1 else 0.5
                cls_res = self._classify_scene(
                    metrics["keyframe_path"],
                    metrics["edge_density"],
                    rel_pos
                )
                dur = e_sec - s_sec
                return {
                    "scene_id": f"v{file_order}_s{c_idx}",
                    "video_path": video_path,
                    "file_order": file_order,
                    "chronological_index": c_idx,
                    "start": round(float(s_sec), 3),
                    "end": round(float(e_sec), 3),
                    "start_time": round(float(s_sec), 3),
                    "end_time": round(float(e_sec), 3),
                    "duration": round(float(dur), 3),
                    "shot_type": cls_res["shot_type"],
                    "content_type": cls_res["content_type"],
                    "workflow_stage": cls_res["workflow_stage"],
                    "description": cls_res["description"],
                    "motion_score": round(float(metrics["motion_score"]), 3),
                    "sharpness_score": round(float(metrics["sharpness_score"]), 2),
                    "quality_score": round(min(10.0, (metrics["motion_score"] * 1.2) + (metrics["sharpness_score"] / 40.0)), 2),
                    "keyframe_path": metrics["keyframe_path"],
                    "thumb_path": metrics["thumb_path"],
                    "used": False
                }

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                file_valid_scenes = list(pool.map(_classify_worker, filtered_candidates))

            file_valid_scenes.sort(key=lambda x: x["chronological_index"])
            all_valid_scenes.extend(file_valid_scenes)

            if self.use_cache and file_valid_scenes:
                cache_db[sig] = file_valid_scenes
                cache_updated = True

            if progress_callback:
                progress_callback(1.0, f"Hoàn tất phân tích {len(file_valid_scenes)} cảnh!")

        if cache_updated:
            self._save_cache_db(cache_db)

        return all_valid_scenes

    def _detect_and_split_scenes(
        self,
        video_path: str,
        max_scene_duration: float = 120.0,
        chunk_split_duration: float = 20.0
    ) -> List[tuple]:
        """Tự động điều chỉnh bước nhảy quét cảnh thông minh cho cả video 1 phút lẫn video 2-3 tiếng."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_dur = frame_count / fps if fps > 0 else 0.0

        raw_intervals = []
        if total_dur > 0:
            step_sec = 2.0 if total_dur > 3600.0 else (1.0 if total_dur > 900.0 else 0.5)
            curr_t = 0.0
            scene_start = 0.0
            prev_gray = None

            while curr_t < total_dur:
                cap.set(cv2.CAP_PROP_POS_MSEC, curr_t * 1000.0)
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_NEAREST)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
                    if diff > 28.0 and (curr_t - scene_start) >= 1.8:
                        raw_intervals.append((scene_start, curr_t))
                        scene_start = curr_t
                prev_gray = gray
                curr_t += step_sec

            if total_dur - scene_start >= 1.2:
                raw_intervals.append((scene_start, total_dur))

        cap.release()

        if not raw_intervals and total_dur > 0:
            raw_intervals = [(0.0, total_dur)]

        effective_chunk = 45.0 if total_dur > 3600.0 else chunk_split_duration
        intervals = []
        for start_s, end_s in raw_intervals:
            dur = end_s - start_s
            if dur > max_scene_duration:
                curr = start_s
                while curr < end_s:
                    nxt = min(curr + effective_chunk, end_s)
                    if nxt - curr >= 1.5:
                        intervals.append((curr, nxt))
                    curr = nxt
            else:
                intervals.append((start_s, end_s))

        return intervals

    def _evaluate_scene_quality(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        prev_hist: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)

        sample_times = np.linspace(start_sec + 0.05, max(start_sec + 0.05, end_sec - 0.1), num=4)
        frames_gray = []
        sampled_bgr = []
        best_frame_bgr = None
        best_sharpness = -1.0

        for i, t in enumerate(sample_times):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            sampled_bgr.append(frame)
            small_bgr = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
            frames_gray.append(gray)

            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if sharpness > best_sharpness or i == 1:
                if sharpness > best_sharpness:
                    best_sharpness = sharpness
                if best_frame_bgr is None or (1 <= i <= 2):
                    best_frame_bgr = frame

        cap.release()

        if len(frames_gray) < 2 or best_frame_bgr is None:
            return {"is_valid": False, "reason": "Không đọc được frame"}

        diffs = [
            np.mean(cv2.absdiff(frames_gray[i], frames_gray[i + 1]))
            for i in range(len(frames_gray) - 1)
        ]
        motion_score = float(np.mean(diffs))

        if motion_score < 0.8:
            return {"is_valid": False, "reason": f"Cảnh đứng hình (Motion={motion_score:.2f})"}

        hsv = cv2.cvtColor(cv2.resize(best_frame_bgr, (320, 180)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        mid_gray = frames_gray[len(frames_gray) // 2]
        edges = cv2.Canny(mid_gray, 80, 160)
        edge_density = float(np.mean(edges > 0))

        clean_base = re.sub(r'^\d{3,5}_\d+_', '', os.path.basename(video_path))
        clean_base = re.sub(r'^\d{3,5}_', '', clean_base)
        thumb_path = os.path.join(self.temp_dir, f"thumb_{clean_base}_{int(start_sec * 10)}.jpg")
        keyframe_path = os.path.join(self.temp_dir, f"filmstrip_{clean_base}_{int(start_sec * 10)}.jpg")

        if not os.path.exists(thumb_path):
            thumb_img = cv2.resize(best_frame_bgr, (854, 480), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(thumb_path, thumb_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        if not os.path.exists(keyframe_path):
            if len(sampled_bgr) >= 3:
                f_start = cv2.resize(sampled_bgr[0], (640, 360), interpolation=cv2.INTER_LANCZOS4)
                f_mid = cv2.resize(sampled_bgr[len(sampled_bgr) // 2], (640, 360), interpolation=cv2.INTER_LANCZOS4)
                f_end = cv2.resize(sampled_bgr[-1], (640, 360), interpolation=cv2.INTER_LANCZOS4)
                filmstrip = cv2.hconcat([f_start, f_mid, f_end])
                cv2.imwrite(keyframe_path, filmstrip, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                thumb_img = cv2.resize(best_frame_bgr, (854, 480), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(keyframe_path, thumb_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        return {
            "is_valid": True,
            "motion_score": motion_score,
            "sharpness_score": max(1.0, best_sharpness),
            "histogram": hist,
            "edge_density": edge_density,
            "keyframe_path": keyframe_path,
            "thumb_path": thumb_path,
            "reason": "OK"
        }

    def _classify_scene(
        self,
        keyframe_path: str,
        edge_density: float,
        relative_pos: float
    ) -> Dict[str, Any]:
        if self.use_gemini and os.path.exists(keyframe_path):
            try:
                img = Image.open(keyframe_path)
                prompt = (
                    "Bạn là đạo diễn dựng phim tài liệu xây dựng/kỹ thuật chuyên nghiệp. "
                    "Bức ảnh này là dải phim 3 khung hình liên tiếp (Đầu - Giữa - Cuối) của một phân cảnh. "
                    "Hãy phân tích tiến trình hành động và trả về DUY NHẤT một chuỗi JSON hợp lệ có cấu trúc:\n"
                    "{\n"
                    '  "shot_type": "Close-up" hoặc "Wide shot",\n'
                    '  "content_type": một trong ["Thao tác", "Kỹ thuật", "Công cụ/Vật liệu", "Vấn đề", "Cách xử lý", "Kết quả"],\n'
                    '  "workflow_stage": số nguyên từ 1 đến 4,\n'
                    '  "description": "mô tả hành động và công cụ bằng cả tiếng Việt và từ khóa tiếng Anh"\n'
                    "}"
                )
                response = self.model.generate_content([prompt, img])
                raw_text = response.text.strip()
                if "```" in raw_text:
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
                parsed = json.loads(raw_text.strip())

                return {
                    "shot_type": parsed.get("shot_type", "Close-up"),
                    "content_type": parsed.get("content_type", "Thao tác"),
                    "workflow_stage": int(parsed.get("workflow_stage", 2)),
                    "description": parsed.get("description", "Cảnh thi công")
                }
            except Exception:
                pass

        shot_type = "Wide shot" if edge_density > 0.14 or relative_pos < 0.15 or relative_pos > 0.85 else "Close-up"
        if relative_pos <= 0.15:
            workflow_stage, content_type = 1, "Công cụ/Vật liệu"
        elif relative_pos >= 0.85:
            workflow_stage, content_type = 4, "Kết quả"
        elif relative_pos >= 0.55:
            workflow_stage, content_type = 3, "Kỹ thuật"
        else:
            workflow_stage, content_type = 2, "Thao tác"

        return {
            "shot_type": shot_type,
            "content_type": content_type,
            "workflow_stage": workflow_stage,
            "description": f"Cảnh {shot_type} - {content_type}"
        }