import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from datasets import load_dataset
from PIL import Image

from .base_loader import BaseDataLoader


class ScreenProDataLoader(BaseDataLoader):
    """
    Data loader for the ScreenSpot-Pro dataset (Voxel51/ScreenSpot-Pro).

    It normalizes each sample into the common:
        {'image': PIL.Image, 'question': str, 'bbox': (x, y, w, h), 'answer': Optional[str]}
    format that the rest of the pipeline expects.
    """

    _QUESTION_KEYS = ("instruction", "question", "prompt", "query", "caption")
    _ANSWER_KEYS = ("answer", "response", "label", "target_description", "description")
    _IMAGE_PATH_KEYS = ("image_path", "screenshot_path")
    _BBOX_KEYS = ("bbox", "target_bbox", "bbox_xywh")

    def _load_and_split_data(self):
        print(f"Loading dataset: {self.name}")
        try:
            dataset = load_dataset(self.name, split=self.split)
        except Exception as exc:
            print(f"Failed to load dataset {self.name}. Error: {exc}")
            raise

        normalized_samples = []
        for raw_sample in dataset:
            normalized = self._normalize_sample(raw_sample)
            if normalized is not None:
                normalized_samples.append(normalized)

        if not normalized_samples:
            raise RuntimeError(
                "ScreenSpot-Pro dataset did not yield any usable samples after normalization."
            )

        random.shuffle(normalized_samples)
        split_idx = int(self.split_ratio * len(normalized_samples))
        self.train_samples = normalized_samples[:split_idx]
        self.test_samples = normalized_samples[split_idx:]

        print(
            f"Dataset loaded and split: {len(self.train_samples)} for training, "
            f"{len(self.test_samples)} for testing."
        )

    def _normalize_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        image = self._extract_image(sample)
        bbox = self._extract_bbox(sample)
        if image is None or bbox is None:
            # 丢弃缺少图像或坐标答案的样本
            return None

        question = self._extract_text(sample, self._QUESTION_KEYS)
        if question is None:
            # 最起码要有一条自然语言指令
            return None

        answer = self._extract_text(sample, self._ANSWER_KEYS)

        normalized: Dict[str, Any] = {
            "image": image,
            "question": question,
            "bbox": bbox,
        }
        if answer:
            normalized["answer"] = answer

        return normalized

    def _extract_text(
        self, sample: Dict[str, Any], candidate_keys: Sequence[str]
    ) -> Optional[str]:
        for key in candidate_keys:
            value = sample.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _extract_image(self, sample: Dict[str, Any]) -> Optional[Image.Image]:
        image = sample.get("image") or sample.get("screenshot")
        if isinstance(image, Image.Image):
            return image

        # Some dataset variants only store relative paths.
        for key in self._IMAGE_PATH_KEYS:
            path_value = sample.get(key)
            if not path_value:
                continue
            try:
                path = Path(path_value)
                if path.exists():
                    return Image.open(path).convert("RGB")
            except Exception:
                continue

        return None

    def _extract_bbox(self, sample: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
        # 常见形式：字段本身是长度为 4 的 list/tuple，或 dict 包含 x,y,w,h / width,height
        for key in self._BBOX_KEYS:
            bbox_value = sample.get(key)
            parsed = self._parse_bbox_value(bbox_value)
            if parsed is not None:
                return parsed

        # 退路：单独的标量字段
        candidates = (
            ("x", "y", "w", "h"),
            ("left", "top", "width", "height"),
        )
        for keys in candidates:
            try:
                vals = [float(sample[k]) for k in keys]
                if len(vals) == 4:
                    return tuple(vals)  # type: ignore[return-value]
            except (KeyError, TypeError, ValueError):
                continue

        return None

    def _parse_bbox_value(
        self, value: Optional[Any]
    ) -> Optional[Tuple[float, float, float, float]]:
        if value is None:
            return None

        if isinstance(value, dict):
            x = value.get("x")
            y = value.get("y")
            w = value.get("w", value.get("width"))
            h = value.get("h", value.get("height"))
            try:
                return float(x), float(y), float(w), float(h)
            except (TypeError, ValueError):
                return None

        if isinstance(value, (list, tuple)):
            seq: Sequence[Any] = value
            if len(seq) != 4:
                return None
            try:
                return tuple(float(v) for v in seq)  # type: ignore[return-value]
            except (TypeError, ValueError):
                return None

        return None


