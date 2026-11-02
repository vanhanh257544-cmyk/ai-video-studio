import os
import re
from typing import List, Dict, Any, Optional
try:
    import whisper  # type: ignore
except ImportError:
    whisper = None
from moviepy.editor import AudioFileClip, concatenate_audioclips  # type: ignore


class AudioProcessor:
    CONTENT_KEYWORDS = {
        "Thao tác": [
            "cắt", "hàn", "đục", "khoan", "lắp", "trát", "sơn", "đổ", "buộc",
            "dán", "chà", "mài", "bắt vít", "đo", "căn chỉnh", "thi công", "làm",
            "cut", "weld", "drill", "install", "assemble", "paint", "pour", "plaster",
            "grind", "screw", "measure", "align", "apply", "build", "construct", "working",
            "drives", "carving", "hauling", "kneads", "sets", "pounds", "stripping",
            "chops", "nailing", "packs", "spreads", "places", "presses", "saws", "hammers",
            "hooks", "brushes", "rams", "crushes", "hangs", "strikes", "slices", "locks", "anchors", "seals"
        ],
        "Kỹ thuật": [
            "kỹ thuật", "tiêu chuẩn", "độ dày", "tỷ lệ", "góc", "mạch", "kết cấu",
            "chống thấm", "chịu lực", "cốt thép", "bê tông", "nguyên tắc", "cách thức",
            "technique", "standard", "thickness", "ratio", "angle", "structure",
            "waterproof", "reinforcement", "concrete", "rebar", "method", "principle",
            "plumb", "level", "lap joints", "trusses", "triangulating", "insulation",
            "equilibrium", "tension", "watertight"
        ],
        "Công cụ/Vật liệu": [
            "máy", "kìm", "búa", "bay", "vật liệu", "xi măng", "cát", "đá", "gạch",
            "sắt", "thép", "keo", "sơn", "ống", "dụng cụ", "thiết bị", "tấm",
            "machine", "tool", "hammer", "pliers", "material", "cement", "sand", "stone",
            "brick", "steel", "iron", "glue", "pipe", "equipment", "panel", "gear",
            "twine", "axe", "shovel", "flagstones", "clay", "mortar", "rammer", "drawknife",
            "tar", "spikes", "spruce", "bitumen", "shingles", "moss", "flue", "cordwood", "slate"
        ],
        "Vấn đề": [
            "lỗi", "nứt", "thấm", "lệch", "hỏng", "vấn đề", "nhược điểm", "rủi ro",
            "cong vênh", "rỗ", "bong tróc", "sai lầm", "cẩn thận", "tránh",
            "error", "crack", "leak", "damage", "issue", "problem", "risk",
            "defect", "mistake", "careful", "avoid", "warning", "flaw",
            "damp", "rot", "crushing", "gales", "erosion", "decayed", "drafts"
        ],
        "Cách xử lý": [
            "xử lý", "khắc phục", "sửa", "gia cố", "bù", "vá", "giải pháp",
            "cách giải quyết", "biện pháp", "làm lại",
            "fix", "solve", "repair", "reinforce", "solution", "patch", "handle",
            "redo", "remedy", "resolve", "sealing", "compress", "banish", "shields", "block"
        ],
        "Kết quả": [
            "hoàn thiện", "kết quả", "xong", "bàn giao", "đẹp", "chắc chắn",
            "bền", "hoàn thành", "nghiệm thu", "cuối cùng", "thành quả",
            "finish", "result", "done", "complete", "completed", "final",
            "outcome", "durable", "solid", "inspection", "handover",
            "sanctuary", "chandelier", "forever", "leaving behind", "sunlit"
        ]
    }

    WIDE_SHOT_KEYWORDS = [
        "toàn bộ", "tổng thể", "quy mô", "toàn cảnh", "kết cấu", "công trình",
        "mặt tiền", "không gian", "hoàn thiện", "kết quả", "nhìn chung", "hiện trạng",
        "overall", "wide", "overview", "entire", "structure", "site", "building",
        "facade", "project", "space", "landscape", "scale", "general", "view",
        "woods", "forest", "cabin", "roofline", "canopy", "hills", "terrace",
        "veranda", "porch", "walks into", "sanctuary"
    ]

    CLOSE_UP_KEYWORDS = [
        "chi tiết", "cận cảnh", "mối hàn", "vết nứt", "bề mặt", "mạch", "ốc vít",
        "góc cạnh", "kỹ thuật", "thao tác", "quan sát kỹ", "điểm nối",
        "detail", "close-up", "closeup", "joint", "weld", "screw", "surface",
        "edge", "crack", "inspect", "closely", "connection", "precision",
        "nail", "hammer", "chisel", "knot", "twine", "stake", "mortar", "clay",
        "seam", "plumb", "level", "hinge", "chain", "eyebolt", "hook", "candle", "brush"
    ]

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = whisper.load_model(self.model_size)
        return self._model

    def process_multiple_audios(self, audio_paths: List[str], output_merged_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        def natural_sort_key(s):
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

        sorted_paths = sorted(audio_paths, key=natural_sort_key)
        audio_clips = []
        raw_segments = []
        current_time = 0.0

        for a_path in sorted_paths:
            clip = AudioFileClip(a_path)
            dur = max(0.5, float(clip.duration))
            audio_clips.append(clip)

            try:
                res = self.model.transcribe(a_path, language=language, verbose=False)
                sub_segs = res.get("segments", [])
                text = res.get("text", "").strip()
            except Exception:
                sub_segs = []
                text = os.path.splitext(os.path.basename(a_path))[0]

            if sub_segs and len(sub_segs) > 1:
                for s in sub_segs:
                    st_text = s.get("text", "").strip()
                    if not st_text:
                        continue
                    s_start = current_time + float(s.get("start", 0.0))
                    s_end = current_time + float(s.get("end", dur))
                    raw_segments.append({
                        "start": s_start,
                        "end": min(current_time + dur, s_end),
                        "text": st_text,
                        "vi_text": ""
                    })
                if raw_segments and raw_segments[-1]["end"] < current_time + dur:
                    raw_segments[-1]["end"] = current_time + dur
            else:
                raw_segments.append({
                    "start": current_time,
                    "end": current_time + dur,
                    "text": text if text else os.path.splitext(os.path.basename(a_path))[0],
                    "vi_text": ""
                })

            current_time += dur

        merged_audio = concatenate_audioclips(audio_clips)
        merged_audio.write_audiofile(output_merged_path, fps=44100, codec="pcm_s16le", logger=None)

        for c in audio_clips:
            c.close()
        merged_audio.close()

        return self._build_analyzed_result(raw_segments, output_merged_path, language or "multi-audio")

    def parse_script_file(self, file_path: str) -> Dict[str, Any]:
        """
        Đọc chính xác 100% mốc thời gian, câu thoại chính và dòng phụ đề tiếng Việt (VI: ...) nếu có.
        """
        raw_segments: List[Dict[str, Any]] = []

        def time_to_sec(t_str: str) -> float:
            s = t_str.strip().replace(",", ".")
            parts = s.split(":")
            if len(parts) == 3:
                return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
            elif len(parts) == 2:
                return float(parts[0]) * 60.0 + float(parts[1])
            return float(parts[0])

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()

            blocks = re.split(r'\n\s*\n', content)
            time_range_pattern = re.compile(
                r'(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?)\s*(?:-->|->|-|–)\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?)'
            )

            for block in blocks:
                lines = [line.strip() for line in block.split('\n') if line.strip()]
                if not lines:
                    continue

                matched_time = None
                time_line_idx = -1
                for idx_l, line in enumerate(lines):
                    m = time_range_pattern.search(line)
                    if m:
                        matched_time = m
                        time_line_idx = idx_l
                        break

                if matched_time is not None:
                    start_sec = time_to_sec(matched_time.group(1))
                    end_sec = time_to_sec(matched_time.group(2))

                    en_lines = []
                    vi_lines = []
                    candidate_lines = []
                    for l_after in lines[time_line_idx + 1:]:
                        l_clean = re.sub(r'^[\*\-\•]+\s*', '', l_after).strip()
                        if l_clean.upper().startswith("EN:"):
                            en_text = l_clean[3:].strip()
                            en_text = re.sub(r'\(\s*(?:Số từ|Target|\d+\s*từ).*?\)\s*$', '', en_text, flags=re.IGNORECASE).strip()
                            en_lines.append(en_text)
                        elif l_clean.upper().startswith("VI:"):
                            vi_lines.append(l_clean[3:].strip())
                        elif not any(l_clean.lower().startswith(p) for p in [
                            "điểm chạm", "target words", "khoảng cách", "=== ", "---"
                        ]):
                            candidate_lines.append(l_clean)

                    final_text = " ".join(en_lines) if en_lines else " ".join(candidate_lines)
                    final_text = final_text.strip()
                    final_vi = " ".join(vi_lines).strip()
                    final_text = re.sub(r'\(\s*(?:Số từ|Target|\d+\s*từ).*?\)\s*$', '', final_text, flags=re.IGNORECASE).strip()
                    if final_text and end_sec > start_sec:
                        raw_segments.append({
                            "start": round(start_sec, 3),
                            "end": round(end_sec, 3),
                            "duration": round(max(0.5, end_sec - start_sec), 3),
                            "text": final_text,
                            "vi_text": final_vi
                        })
        except Exception as e:
            print(f"[AudioProcessor] Lỗi đọc file kịch bản: {e}")

        return self._build_analyzed_result(raw_segments, file_path, "script_text")

    def transcribe_and_analyze(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        result = self.model.transcribe(audio_path, language=language, verbose=False)
        raw_segments = [
            {
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 2.0)),
                "text": s.get("text", "").strip(),
                "vi_text": ""
            }
            for s in result.get("segments", [])
        ]
        return self._build_analyzed_result(raw_segments, audio_path, result.get("language", "unknown"))

    def _build_analyzed_result(self, raw_segments: List[Dict[str, Any]], path: str, lang: str) -> Dict[str, Any]:
        total_segments = len(raw_segments)
        analyzed: List[Dict[str, Any]] = []

        for idx, seg in enumerate(raw_segments):
            text = seg["text"]
            start = round(float(seg["start"]), 3)
            end = round(float(seg["end"]), 3)
            duration = round(max(0.5, end - start), 3)
            progress_ratio = idx / max(1, total_segments - 1) if total_segments > 1 else 0.0

            content_type = self._classify_content_type(text)
            preferred_shot = self._determine_shot_type(text, content_type, progress_ratio)
            workflow_stage = self._determine_workflow_stage(text, content_type, progress_ratio)

            analyzed.append({
                "index": idx,
                "start": start,
                "end": end,
                "duration": duration,
                "text": text,
                "vi_text": seg.get("vi_text", ""),
                "content_type": content_type,
                "preferred_shot": preferred_shot,
                "workflow_stage": workflow_stage
            })

        total_duration = analyzed[-1]["end"] if analyzed else 0.0
        return {
            "audio_path": path,
            "detected_language": lang,
            "total_duration": total_duration,
            "full_text": "\n".join(s["text"] for s in analyzed),
            "segments": analyzed
        }

    def _classify_content_type(self, text: str) -> str:
        text_lower = text.lower()
        priority_order = ["Thao tác", "Kỹ thuật", "Công cụ/Vật liệu", "Vấn đề", "Cách xử lý", "Kết quả"]
        best_category = "Thao tác"
        max_hits = 0

        for category in priority_order:
            keywords = self.CONTENT_KEYWORDS[category]
            hits = sum(1 for kw in keywords if kw in text_lower)
            if hits > max_hits:
                max_hits = hits
                best_category = category

        return best_category

    def _determine_shot_type(self, text: str, content_type: str, progress_ratio: float) -> str:
        text_lower = text.lower()
        wide_score = sum(1 for kw in self.WIDE_SHOT_KEYWORDS if kw in text_lower)
        close_score = sum(1 for kw in self.CLOSE_UP_KEYWORDS if kw in text_lower)

        if wide_score > close_score:
            return "Wide shot"
        if close_score > wide_score:
            return "Close-up"

        if content_type == "Kết quả" or progress_ratio < 0.12 or progress_ratio > 0.88:
            return "Wide shot"
        return "Close-up"

    def _determine_workflow_stage(self, text: str, content_type: str, progress_ratio: float) -> int:
        if progress_ratio <= 0.15:
            return 1
        elif progress_ratio >= 0.85 or content_type == "Kết quả":
            return 4
        elif content_type in ["Kỹ thuật", "Vấn đề", "Cách xử lý"] or (0.50 <= progress_ratio < 0.85):
            return 3
        else:
            return 2