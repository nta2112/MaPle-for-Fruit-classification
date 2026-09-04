import os
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

try:
    from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
    from dassl.utils import mkdir_if_missing, read_json, write_json
except ImportError:
    # Fallback khi chưa cài đặt Dassl hoặc dùng độc lập cho inference/eval
    class _MockRegistry:
        def register(self):
            def decorator(cls):
                return cls
            return decorator

    DATASET_REGISTRY = _MockRegistry()

    class Datum:
        def __init__(self, impath="", label=0, domain=0, classname=""):
            self.impath = impath
            self.label = label
            self.domain = domain
            self.classname = classname

    class DatasetBase:
        def __init__(self, train_x=None, train_u=None, val=None, test=None):
            self.train_x = train_x or []
            self.train_u = train_u or []
            self.val = val or []
            self.test = test or []

    def mkdir_if_missing(d):
        os.makedirs(d, exist_ok=True)
    def read_json(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    def write_json(obj, p):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4)


CLASS_NAME_MAP = {
    # Apple
    "america_apple":      "American apple",
    "japan_apple":        "Japanese apple",
    "newzealand_apple":   "New Zealand apple",
    "southafrica_apple":  "South African apple",
    "netherlands_apple":  "Dutch apple",
    "chinese_apple":      "Chinese apple",
    "vietnam_apple":      "Vietnamese apple",
    "china_apple":        "China apple",
    # Grape
    "america_grape":      "American grape",
    "japan_grape":        "Japanese grape",
    "korea_grape":        "Korean grape",
    "southafrica_grape":  "South African grape",
    "vietnamese_grape":   "Vietnamese grape",
    "vietnam_grape":      "Vietnamese grape",
    "china_grape":        "Chinese grape",
    "chinese_grape":      "Chinese grape",
    # Orange
    "america_orange":     "American orange",
    "japan_orange":       "Japanese orange",
    "korea_orange":       "Korean orange",
    "china_orange":       "Chinese orange",
    "australia_orange":   "Australian orange",
    "vietnamese_orange":  "Vietnamese orange",
    "vietnam_orange":     "Vietnamese orange",
    # Potato
    "other-potato":       "potato",
    "china-potato":       "Chinese potato",
    "dalat-potato":       "Da Lat potato",
    # Strawberry
    "china-strawberry":   "Chinese strawberry",
    "dalat-strawberry":   "Da Lat strawberry",
}

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}


@DATASET_REGISTRY.register()
class Fruit(DatasetBase):
    """Few-shot Fruit recognition dataset for MaPLe.
    
    Supports:
      - Local paths (e.g. D:/Fewshot-Fruit/archive/images/images and D:/Fewshot-Fruit/test_split.json)
      - Kaggle paths (e.g. /kaggle/input/.../archive/images/images and /kaggle/input/.../test_split.json)
      - Base-to-novel evaluation (base=14 train classes, new=5 test classes)
    """
    dataset_dir = "fruit"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT)) if cfg.DATASET.ROOT else ""
        split_path = getattr(cfg.DATASET, "SPLIT_PATH", "")

        # 1. Resolve split_path
        self.split_path = self._resolve_split_path(root, split_path)
        print(f"[Fruit Dataset] Using split file: {self.split_path}")

        # 2. Read split definition
        with open(self.split_path, "r", encoding="utf-8") as f:
            split_info = json.load(f)

        self.base_classes = split_info.get("train", [])
        self.val_classes = split_info.get("val", [])
        self.novel_classes = split_info.get("test", [])
        all_classes = self.base_classes + self.val_classes + self.novel_classes

        # 3. Resolve image directory
        self.image_dir = self._resolve_image_dir(root, all_classes)
        print(f"[Fruit Dataset] Using image directory: {self.image_dir}")

        # 4. Resolve writable cache directory (Kaggle /kaggle/input is read-only)
        self.cache_dir = self._resolve_cache_dir(root, cfg)
        self.split_fewshot_dir = os.path.join(self.cache_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # 5. Build raw train, val, test items
        train, val, test = self._build_data_splits(split_info)
        print(f"[Fruit Dataset] Raw counts -> train: {len(train)}, val: {len(val)}, test: {len(test)}")

        # 6. Subsample classes (base, new, val, all)
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)
        print(f"[Fruit Dataset] After subsample '{subsample}' -> train: {len(train)}, val: {len(val)}, test: {len(test)}")

        # 7. Generate few-shot dataset if num_shots is set
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1 and len(train) > 0:
            seed = cfg.SEED
            cache_file = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}-{subsample}.pkl")
            if os.path.exists(cache_file):
                print(f"[Fruit Dataset] Loading preprocessed few-shot data from {cache_file}")
                with open(cache_file, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                if len(val) > 0:
                    val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"[Fruit Dataset] Saving preprocessed few-shot data to {cache_file}")
                try:
                    with open(cache_file, "wb") as file:
                        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print(f"[Fruit Dataset] Warning: Could not write cache file {cache_file} ({e})")

        super().__init__(train_x=train, val=val, test=test)

    def _resolve_split_path(self, root: str, user_split_path: str) -> str:
        if user_split_path and os.path.isfile(user_split_path):
            return os.path.abspath(user_split_path)

        candidates = []
        if root:
            candidates.extend([
                os.path.join(root, "test_split.json"),
                os.path.join(root, "split.json"),
                os.path.join(root, "..", "test_split.json"),
                os.path.join(root, "..", "split.json"),
                os.path.join(root, "..", "..", "test_split.json"),
                os.path.join(root, "..", "..", "split.json"),
            ])
        candidates.extend([
            "D:/Fewshot-Fruit/test_split.json",
            "D:/Fewshot-Fruit/split.json",
            "/kaggle/input/test_split.json",
            "test_split.json",
            "split.json",
        ])

        for path in candidates:
            norm = os.path.normpath(path)
            if os.path.isfile(norm):
                return norm

        raise FileNotFoundError(
            f"Could not find test_split.json! Please specify via DATASET.SPLIT_PATH or place test_split.json in dataset folder. Candidates checked: {candidates[:5]}"
        )

    def _resolve_image_dir(self, root: str, sample_classes: list[str]) -> str:
        if root:
            # Check direct class folders inside root
            if any(os.path.isdir(os.path.join(root, c)) for c in sample_classes[:3]):
                return root

            # Check standard subdirectories
            for sub in [
                os.path.join("archive", "images", "images"),
                os.path.join("images", "images"),
                os.path.join("archive", "images"),
                "images",
            ]:
                candidate = os.path.join(root, sub)
                if os.path.isdir(candidate) and any(os.path.isdir(os.path.join(candidate, c)) for c in sample_classes[:3]):
                    return candidate

        # Fallback candidate list
        default_candidates = [
            "D:/Fewshot-Fruit/archive/images/images",
            "/kaggle/input/fruit-recognition/archive/images/images",
            "/kaggle/input/fruit-recognition/images/images",
        ]
        for candidate in default_candidates:
            if os.path.isdir(candidate):
                return candidate

        if root and os.path.isdir(root):
            return root

        raise FileNotFoundError(
            f"Could not locate image directory for Fruit dataset (checked root='{root}'). Please pass --root pointing to images."
        )

    def _resolve_cache_dir(self, root: str, cfg) -> str:
        output_dir = getattr(cfg, "OUTPUT_DIR", "")
        if output_dir and os.path.isdir(output_dir):
            try:
                test_file = os.path.join(output_dir, ".test_write")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                return os.path.join(output_dir, "fruit_cache")
            except OSError:
                pass

        # Check if root is writable
        if root and os.path.isdir(root):
            try:
                test_file = os.path.join(root, ".test_write")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                return os.path.join(root, "fruit_cache")
            except OSError:
                pass

        # Fallback to local or Kaggle working directory
        if os.path.isdir("/kaggle/working"):
            return "/kaggle/working/fruit_cache"
        return os.path.abspath("./fruit_cache")

    def _collect_images(self, class_name: str) -> list[str]:
        cls_dir = os.path.join(self.image_dir, class_name)
        if not os.path.isdir(cls_dir):
            return []
        imgs = []
        for fname in sorted(os.listdir(cls_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMG_EXTENSIONS:
                imgs.append(os.path.join(cls_dir, fname))
        return imgs

    def _build_data_splits(self, split_info: dict, p_trn: float = 0.8):
        """Construct Datum items for base, val, and novel classes.
        
        - Base classes (train): split into 80% train, 20% test
        - Val classes (val): 100% val
        - Novel classes (test): 100% test
        """
        train, val, test = [], [], []

        # Map all classes to integer index
        class_to_label = {}
        all_classes_ordered = self.base_classes + self.val_classes + self.novel_classes
        for idx, c in enumerate(all_classes_ordered):
            class_to_label[c] = idx

        # 1. Base classes (split into train and test for Base-to-Novel evaluation)
        for c in self.base_classes:
            imgs = self._collect_images(c)
            if not imgs:
                continue
            random.Random(42).shuffle(imgs)
            n_total = len(imgs)
            n_train = max(1, int(n_total * p_trn))
            label = class_to_label[c]
            cname = CLASS_NAME_MAP.get(c, c.replace("_", " ").replace("-", " "))

            for p in imgs[:n_train]:
                train.append(Datum(impath=p, label=label, classname=cname))
            for p in imgs[n_train:]:
                test.append(Datum(impath=p, label=label, classname=cname))

        # 2. Val classes
        for c in self.val_classes:
            imgs = self._collect_images(c)
            label = class_to_label[c]
            cname = CLASS_NAME_MAP.get(c, c.replace("_", " ").replace("-", " "))
            for p in imgs:
                val.append(Datum(impath=p, label=label, classname=cname))

        # 3. Novel classes (unseen test classes)
        # Tương tự như benchmark CoOp/MaPLe chuẩn: chia train/test cho novel classes
        # Khi đánh giá zero-shot eval-only trên novel classes, Dassl vẫn cần train_x có nhãn để xác định _num_classes
        for c in self.novel_classes:
            imgs = self._collect_images(c)
            if not imgs:
                continue
            random.Random(42).shuffle(imgs)
            n_total = len(imgs)
            n_train = max(1, int(n_total * p_trn))
            label = class_to_label[c]
            cname = CLASS_NAME_MAP.get(c, c.replace("_", " ").replace("-", " "))

            for p in imgs[:n_train]:
                train.append(Datum(impath=p, label=label, classname=cname))
            for p in imgs[n_train:]:
                test.append(Datum(impath=p, label=label, classname=cname))

        return train, val, test

    def subsample_classes(self, *args, subsample="all"):
        """Subsample classes based on test_split.json:
          - "base": only the 14 base classes, relabeled to 0..13
          - "new": only the 5 novel classes, relabeled to 0..4
          - "val": only the 5 val classes, relabeled to 0..4
          - "all": keep all classes as-is
        """
        assert subsample in ["all", "base", "new", "val"]

        if subsample == "all":
            return args

        if subsample == "base":
            target_class_names = set(
                CLASS_NAME_MAP.get(c, c.replace("_", " ").replace("-", " "))
                for c in self.base_classes
            )
        elif subsample == "new":
            target_class_names = set(
                CLASS_NAME_MAP.get(c, c.replace("_", " ").replace("-", " "))
                for c in self.novel_classes
            )
        else:  # val
            target_class_names = set(
                CLASS_NAME_MAP.get(c, c.replace("_", " ").replace("-", " "))
                for c in self.val_classes
            )

        # Sorted target class names for stable relabeling
        sorted_targets = sorted(list(target_class_names))
        relabeler = {name: i for i, name in enumerate(sorted_targets)}

        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.classname not in target_class_names:
                    continue
                new_item = Datum(
                    impath=item.impath,
                    label=relabeler[item.classname],
                    classname=item.classname
                )
                dataset_new.append(new_item)
            output.append(dataset_new)

        return output
