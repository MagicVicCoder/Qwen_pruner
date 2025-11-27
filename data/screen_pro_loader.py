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
        
        # 先检查缓存状态
        cache_info = self._check_cache_status()
        print(f"Cache directory: {cache_info['cache_path']}")
        if cache_info['exists']:
            print(f"Cache exists: Yes (but datasets library will verify integrity)")
        else:
            print(f"Cache exists: No - will download from scratch")
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # 第一次尝试：使用默认模式（重用缓存）
                download_mode = "reuse_cache_if_exists" if attempt == 0 else "force_redownload"
                if attempt == 0:
                    print(f"Attempt {attempt + 1}: Using download_mode='{download_mode}'")
                else:
                    print(f"Attempt {attempt + 1}: Using download_mode='{download_mode}' (force redownload)")
                dataset = load_dataset(self.name, split=self.split, download_mode=download_mode)
                print("Dataset loaded successfully!")
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

        # 转换为列表，避免迭代器只能遍历一次的问题
        print("Converting dataset to list...")
        all_samples = list(dataset)
        total_count = len(all_samples)
        print(f"Total samples in dataset: {total_count}")
        
        normalized_samples = []
        skipped_count = 0
        skip_reasons = {"no_image": 0, "no_bbox": 0, "no_question": 0}
        
        # 先检查前几个样本的结构
        print("\n=== Inspecting first few samples ===")
        for i in range(min(3, total_count)):
            raw_sample = all_samples[i]
            print(f"\nSample {i} keys: {list(raw_sample.keys())}")
            # 检查关键字段
            for key in ["image", "instruction", "bbox", "target_bbox", "screenshot", "query", "prompt"]:
                if key in raw_sample:
                    value = raw_sample[key]
                    if isinstance(value, (list, tuple, dict)):
                        print(f"  {key}: {type(value).__name__} with length/shape {len(value) if hasattr(value, '__len__') else 'N/A'}")
                    else:
                        print(f"  {key}: {type(value).__name__} = {str(value)[:100]}")
        
        print("\n=== Processing all samples ===")
        for idx, raw_sample in enumerate(all_samples):
            normalized = self._normalize_sample(raw_sample, idx, skip_reasons)
            if normalized is not None:
                normalized_samples.append(normalized)
            else:
                skipped_count += 1
                
            # 每处理100个样本输出一次进度
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{total_count} samples: {len(normalized_samples)} valid, {skipped_count} skipped")

        print(f"\n=== Normalization Summary ===")
        print(f"Total samples processed: {total_count}")
        print(f"Valid samples: {len(normalized_samples)}")
        print(f"Skipped samples: {skipped_count}")
        print(f"Skip reasons: {skip_reasons}")

        if not normalized_samples:
            raise RuntimeError(
                f"ScreenSpot-Pro dataset did not yield any usable samples after normalization. "
                f"Skip reasons: {skip_reasons}. "
                f"Please check the dataset structure - expected fields: image, instruction/question, bbox"
            )

        random.shuffle(normalized_samples)
        split_idx = int(self.split_ratio * len(normalized_samples))
        self.train_samples = normalized_samples[:split_idx]
        self.test_samples = normalized_samples[split_idx:]

        print(
            f"Dataset loaded and split: {len(self.train_samples)} for training, "
            f"{len(self.test_samples)} for testing."
        )

    def _normalize_sample(self, sample: Dict[str, Any], idx: int = -1, skip_reasons: dict = None) -> Optional[Dict[str, Any]]:
        if skip_reasons is None:
            skip_reasons = {}
            
        image = self._extract_image(sample)
        if image is None:
            skip_reasons["no_image"] = skip_reasons.get("no_image", 0) + 1
            if idx < 3:  # 只对前几个样本输出详细调试信息
                print(f"  Sample {idx}: Missing image. Available keys: {list(sample.keys())}")
            return None
            
        bbox = self._extract_bbox(sample)
        if bbox is None:
            skip_reasons["no_bbox"] = skip_reasons.get("no_bbox", 0) + 1
            if idx < 3:
                print(f"  Sample {idx}: Missing bbox. Available keys: {list(sample.keys())}")
                # 尝试查找可能的bbox字段
                for key in sample.keys():
                    if "bbox" in key.lower() or "box" in key.lower() or "coord" in key.lower():
                        print(f"    Found potential bbox key: {key} = {sample[key]}")
            return None

        question = self._extract_text(sample, self._QUESTION_KEYS)
        if question is None:
            skip_reasons["no_question"] = skip_reasons.get("no_question", 0) + 1
            if idx < 3:
                print(f"  Sample {idx}: Missing question/instruction. Available keys: {list(sample.keys())}")
                # 尝试查找可能的question字段
                for key in sample.keys():
                    if any(term in key.lower() for term in ["text", "prompt", "query", "instruction", "desc"]):
                        print(f"    Found potential question key: {key} = {str(sample[key])[:100]}")
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

    def _get_cache_path(self):
        """获取数据集缓存路径"""
        # 获取 datasets 库的缓存目录
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_dir = os.path.join(os.path.expanduser(hf_home), "datasets")
        
        # 构建数据集特定的缓存路径
        dataset_name_safe = self.name.replace("/", "___")
        dataset_cache_path = os.path.join(cache_dir, dataset_name_safe)
        return cache_dir, dataset_cache_path

    def _check_cache_status(self):
        """检查缓存状态"""
        cache_dir, dataset_cache_path = self._get_cache_path()
        exists = os.path.exists(dataset_cache_path)
        
        # 检查是否有子目录（版本目录）
        version_dirs = []
        if exists:
            try:
                for item in os.listdir(dataset_cache_path):
                    item_path = os.path.join(dataset_cache_path, item)
                    if os.path.isdir(item_path):
                        version_dirs.append(item)
            except Exception:
                pass
        
        return {
            "cache_dir": cache_dir,
            "cache_path": dataset_cache_path,
            "exists": exists,
            "version_dirs": version_dirs
        }

    def _cleanup_cache(self):
        """清理可能损坏的数据集缓存"""
        try:
            cache_dir, dataset_cache_path = self._get_cache_path()
            
            if os.path.exists(dataset_cache_path):
                print(f"Removing cache directory: {dataset_cache_path}")
                try:
                    shutil.rmtree(dataset_cache_path)
                    print("Cache directory removed successfully.")
                except Exception as e:
                    print(f"Warning: Could not fully remove cache directory: {e}")
                    # 尝试删除特定版本目录
                    try:
                        for item in os.listdir(dataset_cache_path):
                            item_path = os.path.join(dataset_cache_path, item)
                            try:
                                if os.path.isdir(item_path):
                                    shutil.rmtree(item_path)
                            except Exception:
                                pass
                    except Exception:
                        pass
            else:
                print(f"Cache directory does not exist: {dataset_cache_path}")
        except Exception as e:
            print(f"Warning: Error during cache cleanup: {e}")
            # 不阻止继续执行


