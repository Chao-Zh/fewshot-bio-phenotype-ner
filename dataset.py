"""Dataset / DataLoader。

优先使用 PyTorch；若环境未安装 torch，自动回退到纯 numpy DataLoader，
两者对外 API 一致（均可通过 iter_batches 迭代）。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import Config
from .schema import Episode, Feature, Sample

try:
    import torch
    from torch.utils.data import DataLoader as _TorchDataLoader
    from torch.utils.data import Dataset as _TorchDataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TorchDataLoader = object
    _TorchDataset = object
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# 词表
# ---------------------------------------------------------------------------
class LabelVocab:
    """BIO 标签词表。"""

    PAD = "O"  # O 作为 padding label（配合 -100 忽略）

    def __init__(self, label_set: List[str]):
        self.label_set = sorted(set(label_set))
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}

        # 0=pad/O, 1=[CLS], 2=[SEP]
        specials = ["O", "[CLS]", "[SEP]"]
        idx = 0
        for tok in specials:
            self.label2id[tok] = idx
            self.id2label[idx] = tok
            idx += 1
        for label in self.label_set:
            for prefix in ("B", "I"):
                tok = f"{prefix}-{label}"
                self.label2id[tok] = idx
                self.id2label[idx] = tok
                idx += 1

    def __len__(self) -> int:
        return len(self.label2id)

    def encode(self, tag: str) -> int:
        return self.label2id.get(tag, self.label2id["O"])

    def decode(self, idx: int) -> str:
        return self.id2label.get(idx, "O")


class TokenVocab:
    """字符级词表（中文按字）。可替换为 BioBERT tokenizer 的 vocab。"""

    PAD_TOKEN = "[PAD]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    UNK_TOKEN = "[UNK]"

    def __init__(self, chars: Optional[Iterable[str]] = None):
        specials = [self.PAD_TOKEN, self.CLS_TOKEN, self.SEP_TOKEN, self.UNK_TOKEN]
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        for tok in specials:
            self.token2id[tok] = len(self.token2id)
        if chars:
            for ch in chars:
                if ch not in self.token2id:
                    self.token2id[ch] = len(self.token2id)
        self.id2token = {v: k for k, v in self.token2id.items()}

    def __len__(self) -> int:
        return len(self.token2id)

    @property
    def pad_id(self) -> int:
        return self.token2id[self.PAD_TOKEN]

    @property
    def cls_id(self) -> int:
        return self.token2id[self.CLS_TOKEN]

    @property
    def sep_id(self) -> int:
        return self.token2id[self.SEP_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token2id[self.UNK_TOKEN]

    def encode(self, ch: str) -> int:
        return self.token2id.get(ch, self.unk_id)


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------
def build_vocabs(samples: Sequence[Sample], label_set: List[str]):
    chars = set()
    for s in samples:
        chars.update(s.text)
    return TokenVocab(chars), LabelVocab(label_set)


def sample_to_feature(
    sample: Sample, token_vocab: TokenVocab, label_vocab: LabelVocab, cfg: Optional[Config] = None
) -> Feature:
    """将一条（可能是窗口）Sample 转为模型特征。"""
    cfg = cfg or Config()
    from .cleaner import entities_to_bio

    tags = entities_to_bio(len(sample.text), sample.entities)

    # [CLS] + tokens + [SEP]
    input_ids = [token_vocab.cls_id] + [token_vocab.encode(c) for c in sample.text] + [token_vocab.sep_id]
    attention_mask = [1] * len(input_ids)
    label_ids = [label_vocab.encode("[CLS]")]  # 特殊 token 标签占位
    label_ids += [label_vocab.encode(t) for t in tags]
    label_ids += [label_vocab.encode("[SEP]")]

    # 特殊 token 与 padding 位置不参与 loss
    label_ids[0] = -100
    label_ids[-1] = -100

    offset_mapping = [(0, 0)] + [(i, i + 1) for i in range(len(sample.text))] + [(0, 0)]

    return Feature(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=label_ids,
        offset_mapping=offset_mapping,
        sample_id=sample.id,
        window_id=int(sample.metadata.get("window_id", 0)),
        is_truncated=bool(sample.metadata.get("is_truncated", False)),
    )


def samples_to_features(
    samples: Sequence[Sample], token_vocab: TokenVocab, label_vocab: LabelVocab, cfg: Optional[Config] = None
) -> List[Feature]:
    return [sample_to_feature(s, token_vocab, label_vocab, cfg) for s in samples]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class NerDataset(_TorchDataset):
    """NER 数据集。torch 可用时为 torch.utils.data.Dataset 子类。"""

    def __init__(self, features: List[Feature], as_tensors: Optional[bool] = None):
        self.features = features
        # None 表示跟随环境是否安装了 torch；DataLoader 可显式覆盖为 False（numpy 回退）
        self.as_tensors = TORCH_AVAILABLE if as_tensors is None else (as_tensors and TORCH_AVAILABLE)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        f = self.features[idx]
        if self.as_tensors:
            return {
                "input_ids": torch.tensor(f.input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(f.attention_mask, dtype=torch.long),
                "labels": torch.tensor(f.labels, dtype=torch.long),
                "sample_id": f.sample_id,
            }
        return f.to_dict()

    def to_dicts(self) -> List[dict]:
        return [f.to_dict() for f in self.features]


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def collate_features(batch: List[dict], pad_id: int = 0, pad_label: int = -100, pad_to_multiple_of: int = 8) -> dict:
    """动态 padding 并组装 batch。"""
    max_len = max(len(b["input_ids"]) for b in batch)
    if pad_to_multiple_of:
        max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    input_ids, attn, labels = [], [], []
    for b in batch:
        l = len(b["input_ids"])
        pad = max_len - l
        input_ids.append(list(b["input_ids"]) + [pad_id] * pad)
        attn.append(list(b.get("attention_mask", [1] * l)) + [0] * pad)
        labels.append(list(b["labels"]) + [pad_label] * pad)

    # 用已 padding 好的 list 构造输出，保证各条等长（不能对未 padding 的原始 tensor 直接 stack）
    if TORCH_AVAILABLE and isinstance(batch[0]["input_ids"], torch.Tensor):
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_ids": [b["sample_id"] for b in batch],
        }
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "sample_ids": [b["sample_id"] for b in batch],
    }


# ---------------------------------------------------------------------------
# DataLoader (torch 或 numpy 回退)
# ---------------------------------------------------------------------------
class NerDataLoader:
    """统一封装的 DataLoader。"""

    def __init__(
        self,
        dataset: NerDataset,
        batch_size: int = 16,
        shuffle: bool = True,
        pad_id: int = 0,
        pad_label: int = -100,
        pad_to_multiple_of: int = 8,
        seed: int = 42,
        use_torch: Optional[bool] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pad_id = pad_id
        self.pad_label = pad_label
        self.pad_to_multiple_of = pad_to_multiple_of
        self.seed = seed
        self.use_torch = TORCH_AVAILABLE if use_torch is None else (use_torch and TORCH_AVAILABLE)
        # 让 dataset 的返回类型与本 loader 后端一致，避免 torch 已安装但走 numpy 回退时类型错配
        self.dataset.as_tensors = self.use_torch

        if self.use_torch:
            import random
            generator = torch.Generator()
            generator.manual_seed(seed)
            self._loader = _TorchDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=lambda b: collate_features(
                    b, pad_id, pad_label, pad_to_multiple_of
                ),
                generator=generator,
            )
        else:
            self._loader = None
            self._rng_seed = seed

    def __iter__(self):
        if self.use_torch:
            yield from self._loader
        else:
            import random
            rng = random.Random(self._rng_seed)
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                idxs = indices[start:start + self.batch_size]
                # __getitem__ 在无 torch 时返回 dict
                batch = [self.dataset[i] for i in idxs]
                yield collate_features(
                    batch, self.pad_id, self.pad_label, self.pad_to_multiple_of
                )

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def build_dataloaders(
    bundle_features: Dict[str, List[Feature]],
    token_vocab: TokenVocab,
    label_vocab: LabelVocab,
    cfg: Config,
) -> Dict[str, NerDataLoader]:
    """为 train/val/test 构建 DataLoader。"""
    loaders: Dict[str, NerDataLoader] = {}
    for split, features in bundle_features.items():
        ds = NerDataset(features)
        loaders[split] = NerDataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=(split == "train") and cfg.shuffle,
            pad_id=token_vocab.pad_id,
            pad_label=-100,
            pad_to_multiple_of=cfg.pad_to_multiple_of,
            seed=cfg.seed,
            use_torch=cfg.use_torch,
        )
    return loaders


# ---------------------------------------------------------------------------
# Episode 批处理（供 few-shot 训练）
# ---------------------------------------------------------------------------
def episode_to_batch(episode: Episode, token_vocab, label_vocab, cfg: Optional[Config] = None) -> Dict[str, Any]:
    """将一个 episode 的 support/query 转为可直接 feed 的 batch。"""
    cfg = cfg or Config()
    support_feats = samples_to_features(episode.support, token_vocab, label_vocab, cfg)
    query_feats = samples_to_features(episode.query, token_vocab, label_vocab, cfg)

    s_ds = NerDataset(support_feats)
    q_ds = NerDataset(query_feats)

    s_loader = NerDataLoader(
        s_ds, batch_size=len(support_feats), shuffle=False,
        pad_id=token_vocab.pad_id, use_torch=cfg.use_torch
    )
    q_loader = NerDataLoader(
        q_ds, batch_size=len(query_feats), shuffle=False,
        pad_id=token_vocab.pad_id, use_torch=cfg.use_torch
    )
    return {
        "ways": episode.ways,
        "support": next(iter(s_loader)),
        "query": next(iter(q_loader)),
        "support_by_way": episode.support_by_way,
        "query_by_way": episode.query_by_way,
        "episode_id": episode.episode_id,
    }
