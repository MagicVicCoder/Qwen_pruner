import random
import os
import shutil
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
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # 第一次尝试：使用默认模式（重用缓存）
                download_mode = "reuse_cache_if_exists" if attempt == 0 else "force_redownload"
                dataset = load_dataset(self.name, split=self.split, download_mode=download_mode)
                break  # 成功加载，退出循环
            except (FileNotFoundError, OSError) as exc:
                error_msg = str(exc)
                # 检查是否是缓存文件损坏的错误
                if "arrow" in error_msg.lower() or "cache" in error_msg.lower() or "no such file" in error_msg.lower():
                    print(f"Attempt {attempt + 1}/{max_retries}: Detected corrupted cache. Cleaning up...")
                    self._cleanup_cache()
                    if attempt < max_retries - 1:
                        print("Retrying with force_redownload...")
                        continue
                    else:
                        print("Failed after retries. Trying force_redownload one more time...")
                        try:
                            dataset = load_dataset(self.name, split=self.split, download_mode="force_redownload")
                            break
                        except Exception as final_exc:
                            print(f"Final attempt failed: {final_exc}")
                            raise RuntimeError(
                                f"Failed to load dataset {self.name} after {max_retries} attempts. "
                                f"Please manually delete the cache directory and try again. "
                                f"Last error: {final_exc}"
                            ) from final_exc
                else:
                    # 其他类型的错误，直接抛出
                    raise
            except Exception as exc:
                print(f"Failed to load dataset {self.name}. Error: {exc}")
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}/{max_retries} failed. Retrying...")
                    self._cleanup_cache()
                    continue
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

    def _cleanup_cache(self):
        """清理可能损坏的数据集缓存"""
        try:
            # 获取 datasets 库的缓存目录
            cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                cache_dir = os.path.join(hf_home, "datasets")
            
            # 构建数据集特定的缓存路径
            dataset_name_safe = self.name.replace("/", "___")
            dataset_cache_path = os.path.join(cache_dir, dataset_name_safe)
            
            if os.path.exists(dataset_cache_path):
                print(f"Removing cache directory: {dataset_cache_path}")
                try:
                    shutil.rmtree(dataset_cache_path)
                    print("Cache directory removed successfully.")
                except Exception as e:
                    print(f"Warning: Could not fully remove cache directory: {e}")
                    # 尝试删除特定版本目录
                    for item in os.listdir(dataset_cache_path):
                        item_path = os.path.join(dataset_cache_path, item)
                        try:
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                        except Exception:
                            pass
        except Exception as e:
            print(f"Warning: Error during cache cleanup: {e}")
            # 不阻止继续执行


