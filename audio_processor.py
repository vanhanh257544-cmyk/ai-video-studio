import re
from typing import List, Dict, Any


class AudioProcessor:
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size

    def parse_script_file(self, file_path: str) -> Dict[str, Any]:
        """
        Đọc chính xác 100% mốc thời gian (00:00:05,000 --> 00:00:20,000 hoặc 00:05 - 00:20),
        câu thoại tiếng Anh (bao gồm cả thẻ cảm xúc Eleven v4 như [sighs]) và dòng phụ đề tiếng Việt.
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
                    final_text = re.sub(r'\(\s*(?:Số từ|Target|\d+\s*từ).*?\)\s*$', '', final_text.strip(), flags=re.IGNORECASE).strip()
                    final_vi = " ".join(vi_lines).strip()

                    if final_text and end_sec > start_sec:
                        raw_segments.append({
                            "index": len(raw_segments),
                            "start": round(start_sec, 3),
                            "end": round(end_sec, 3),
                            "duration": round(max(0.5, end_sec - start_sec), 3),
                            "text": final_text,
                            "vi_text": final_vi
                        })
        except Exception as e:
            print(f"[AudioProcessor] Lỗi đọc file kịch bản: {e}")

        total_duration = raw_segments[-1]["end"] if raw_segments else 0.0
        return {
            "audio_path": file_path,
            "detected_language": "script_text",
            "total_duration": total_duration,
            "full_text": "\n".join(s["text"] for s in raw_segments),
            "segments": raw_segments
        }