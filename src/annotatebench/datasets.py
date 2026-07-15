from __future__ import annotations

from dataclasses import dataclass
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import train_test_split

from .pilot import TextClassificationDataset


FINANCIAL_PHRASEBANK_LABELS = ["negative", "neutral", "positive"]
FINANCIAL_PHRASEBANK_URL = (
    "https://huggingface.co/datasets/financial_phrasebank/resolve/main/data/"
    "sentences_allagree/train-00000-of-00001.parquet"
)
TREC_LABELS = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]
TREC_TRAIN_URL = "https://huggingface.co/datasets/trec/resolve/main/data/train-00000-of-00001.parquet"
TREC_TEST_URL = "https://huggingface.co/datasets/trec/resolve/main/data/test-00000-of-00001.parquet"
BANKING77_TRAIN_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
)
BANKING77_TEST_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
)
DEFAULT_MAX_TRAIN_EXAMPLES = 1200
DEFAULT_MAX_TEST_EXAMPLES = 1000
BENCHMARK_DATASETS = [
    "financial_phrasebank",
    "trec",
    "banking77",
    "ag_news",
    "sst2",
    "twenty_newsgroups",
    "rotten_tomatoes",
    "yelp_polarity",
    "tweet_eval_sentiment",
    "emotion",
]


@dataclass(frozen=True)
class HuggingFaceTextDatasetConfig:
    name: str
    dataset_id: str
    config_name: str | None = None
    train_split: str = "train"
    test_split: str = "test"
    text_columns: tuple[str, ...] = ("text",)
    label_column: str = "label"
    label_names: tuple[str, ...] | None = None


def load_financial_phrasebank(
    path: str | Path | None = None,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    download: bool = False,
    url: str = FINANCIAL_PHRASEBANK_URL,
) -> TextClassificationDataset:
    """Load Financial PhraseBank as a deterministic text-classification split.

    Supported local formats are:
    - CSV with ``text``/``label`` or ``sentence``/``label`` columns.
    - Original PhraseBank text files with ``sentence@label`` rows.
    - ZIP archives containing one of the original ``Sentences_*.txt`` files.

    The optional download path is intentionally explicit so tests and normal
    package imports do not require network access.
    """
    source = Path(path) if path is not None else _default_financial_phrasebank_path(url, download)
    if not source.exists():
        if not download:
            raise FileNotFoundError(
                f"Financial PhraseBank data not found at {source}. "
                "Provide --financial-phrasebank-path or use --download-data."
            )
        source.parent.mkdir(parents=True, exist_ok=True)
        _download_file(url, source)

    texts, labels = _read_phrasebank_source(source)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return TextClassificationDataset(
        name="financial_phrasebank",
        train_texts=list(train_texts),
        train_labels=list(train_labels),
        test_texts=list(test_texts),
        test_labels=list(test_labels),
        label_names=FINANCIAL_PHRASEBANK_LABELS,
    )


def load_trec(*args, **kwargs) -> TextClassificationDataset:
    return _load_public_split_dataset(
        "trec",
        *args,
        label_names=TREC_LABELS,
        label_column="coarse_label",
        label_mapping={idx: label for idx, label in enumerate(TREC_LABELS)},
        train_url=kwargs.pop("train_url", TREC_TRAIN_URL),
        test_url=kwargs.pop("test_url", TREC_TEST_URL),
        **kwargs,
    )


def load_banking77(*args, **kwargs) -> TextClassificationDataset:
    return _load_public_split_dataset(
        "banking77",
        *args,
        label_column=kwargs.pop("label_column", "category"),
        train_url=kwargs.pop("train_url", BANKING77_TRAIN_URL),
        test_url=kwargs.pop("test_url", BANKING77_TEST_URL),
        **kwargs,
    )


def load_ag_news(**kwargs) -> TextClassificationDataset:
    return _load_huggingface_text_classification(
        HuggingFaceTextDatasetConfig("ag_news", "fancyzhx/ag_news"),
        **kwargs,
    )


def load_dbpedia_14(**kwargs) -> TextClassificationDataset:
    return _load_huggingface_text_classification(
        HuggingFaceTextDatasetConfig(
            "dbpedia_14",
            "fancyzhx/dbpedia_14",
            text_columns=("title", "content"),
        ),
        **kwargs,
    )


def load_sst2(**kwargs) -> TextClassificationDataset:
    return _load_huggingface_text_classification(
        HuggingFaceTextDatasetConfig("sst2", "SetFit/sst2", label_names=("negative", "positive")),
        **kwargs,
    )


def load_imdb(**kwargs) -> TextClassificationDataset:
    return _load_huggingface_text_classification(
        HuggingFaceTextDatasetConfig("imdb", "stanfordnlp/imdb"),
        **kwargs,
    )


def load_rotten_tomatoes(**kwargs) -> TextClassificationDataset:
    return _load_huggingface_text_classification(
        HuggingFaceTextDatasetConfig(
            "rotten_tomatoes",
            "cornell-movie-review-data/rotten_tomatoes",
            label_names=("negative", "positive"),
        ),
        **kwargs,
    )


def load_yelp_polarity(**kwargs) -> TextClassificationDataset:
    return _load_huggingface_text_classification(
        HuggingFaceTextDatasetConfig(
            "yelp_polarity",
            "fancyzhx/yelp_polarity",
            label_names=("negative", "positive"),
        ),
        **kwargs,
    )


def load_tweet_eval_sentiment(**kwargs) -> TextClassificationDataset:
    return _load_huggingface_text_classification(
        HuggingFaceTextDatasetConfig("tweet_eval_sentiment", "cardiffnlp/tweet_eval", config_name="sentiment"),
        **kwargs,
    )


def load_emotion(**kwargs) -> TextClassificationDataset:
    return _load_huggingface_text_classification(
        HuggingFaceTextDatasetConfig("emotion", "dair-ai/emotion"),
        **kwargs,
    )


def load_twenty_newsgroups(
    *,
    seed: int = 42,
    max_train_examples: int | None = DEFAULT_MAX_TRAIN_EXAMPLES,
    max_test_examples: int | None = DEFAULT_MAX_TEST_EXAMPLES,
) -> TextClassificationDataset:
    return _load_huggingface_text_classification(
        HuggingFaceTextDatasetConfig("twenty_newsgroups", "SetFit/20_newsgroups"),
        seed=seed,
        max_train_examples=max_train_examples,
        max_test_examples=max_test_examples,
    )


def load_benchmark_dataset(
    name: str,
    *,
    seed: int = 42,
    max_train_examples: int | None = DEFAULT_MAX_TRAIN_EXAMPLES,
    max_test_examples: int | None = DEFAULT_MAX_TEST_EXAMPLES,
    financial_phrasebank_path: str | Path | None = None,
    trec_train_path: str | Path | None = None,
    trec_test_path: str | Path | None = None,
    banking77_train_path: str | Path | None = None,
    banking77_test_path: str | Path | None = None,
    download: bool = False,
) -> TextClassificationDataset:
    """Load one of the ten text-classification benchmark datasets."""
    if name == "financial_phrasebank":
        dataset = load_financial_phrasebank(financial_phrasebank_path, seed=seed, download=download)
    elif name == "trec":
        dataset = load_trec(trec_train_path, trec_test_path, download=download)
    elif name == "banking77":
        dataset = load_banking77(banking77_train_path, banking77_test_path, download=download)
    elif name == "ag_news":
        return load_ag_news(seed=seed, max_train_examples=max_train_examples, max_test_examples=max_test_examples)
    elif name == "dbpedia_14":
        return load_dbpedia_14(seed=seed, max_train_examples=max_train_examples, max_test_examples=max_test_examples)
    elif name == "sst2":
        return load_sst2(seed=seed, max_train_examples=max_train_examples, max_test_examples=max_test_examples)
    elif name == "twenty_newsgroups":
        return load_twenty_newsgroups(
            seed=seed,
            max_train_examples=max_train_examples,
            max_test_examples=max_test_examples,
        )
    elif name == "imdb":
        return load_imdb(seed=seed, max_train_examples=max_train_examples, max_test_examples=max_test_examples)
    elif name == "rotten_tomatoes":
        return load_rotten_tomatoes(
            seed=seed,
            max_train_examples=max_train_examples,
            max_test_examples=max_test_examples,
        )
    elif name == "yelp_polarity":
        return load_yelp_polarity(seed=seed, max_train_examples=max_train_examples, max_test_examples=max_test_examples)
    elif name == "tweet_eval_sentiment":
        return load_tweet_eval_sentiment(
            seed=seed,
            max_train_examples=max_train_examples,
            max_test_examples=max_test_examples,
        )
    elif name == "emotion":
        return load_emotion(seed=seed, max_train_examples=max_train_examples, max_test_examples=max_test_examples)
    else:
        raise ValueError(f"Unknown benchmark dataset: {name}")

    train_texts, train_labels = _cap_examples(
        dataset.train_texts,
        dataset.train_labels,
        max_train_examples,
        seed,
    )
    test_texts, test_labels = _cap_examples(
        dataset.test_texts,
        dataset.test_labels,
        max_test_examples,
        seed,
    )
    return TextClassificationDataset(
        name=dataset.name,
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        label_names=dataset.label_names,
    )


def _default_financial_phrasebank_path(url: str, download: bool) -> Path:
    suffix = Path(url).suffix if download else ".csv"
    return Path(__file__).resolve().parents[2] / "data" / "raw" / f"financial_phrasebank{suffix}"


def _default_split_path(dataset_name: str, split: str, url: str, download: bool) -> Path:
    suffix = Path(url).suffix if download else ".csv"
    return Path(__file__).resolve().parents[2] / "data" / "raw" / f"{dataset_name}_{split}{suffix}"


def _download_file(url: str, output_path: Path) -> None:
    with urlopen(url, timeout=60) as response:
        output_path.write_bytes(response.read())


def _load_public_split_dataset(
    dataset_name: str,
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
    *,
    download: bool = False,
    train_url: str,
    test_url: str,
    text_column: str = "text",
    label_column: str = "label",
    label_names: list[str] | None = None,
    label_mapping: dict[int, str] | None = None,
) -> TextClassificationDataset:
    train_source = Path(train_path) if train_path is not None else _default_split_path(
        dataset_name, "train", train_url, download
    )
    test_source = Path(test_path) if test_path is not None else _default_split_path(
        dataset_name, "test", test_url, download
    )
    _ensure_public_split_file(train_source, train_url, download, dataset_name, "train")
    _ensure_public_split_file(test_source, test_url, download, dataset_name, "test")

    train_texts, train_labels = _read_classification_split(
        train_source,
        text_column=text_column,
        label_column=label_column,
        label_mapping=label_mapping,
    )
    test_texts, test_labels = _read_classification_split(
        test_source,
        text_column=text_column,
        label_column=label_column,
        label_mapping=label_mapping,
    )
    return TextClassificationDataset(
        name=dataset_name,
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        label_names=label_names,
    )


def _ensure_public_split_file(
    path: Path,
    url: str,
    download: bool,
    dataset_name: str,
    split: str,
) -> None:
    if path.exists():
        return
    if not download:
        raise FileNotFoundError(
            f"{dataset_name} {split} data not found at {path}. "
            f"Provide --{dataset_name}-train-path/--{dataset_name}-test-path or use --download-data."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    _download_file(url, path)


def _read_classification_split(
    path: Path,
    *,
    text_column: str,
    label_column: str,
    label_mapping: dict[int, str] | None,
) -> tuple[list[str], list[str]]:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, encoding="utf-8")
    if text_column not in df.columns:
        raise ValueError(f"Missing text column in {path}: {text_column}")
    if label_column not in df.columns:
        raise ValueError(f"Missing label column in {path}: {label_column}")
    texts = df[text_column].astype(str).tolist()
    labels = [_normalize_public_label(value, label_mapping) for value in df[label_column].tolist()]
    return texts, labels


def _normalize_public_label(value: object, label_mapping: dict[int, str] | None) -> str:
    if label_mapping is not None:
        try:
            numeric_value = int(value)
        except (TypeError, ValueError):
            return str(value).strip()
        return label_mapping.get(numeric_value, str(value).strip())
    return str(value).strip()


def _load_huggingface_text_classification(
    config: HuggingFaceTextDatasetConfig,
    *,
    seed: int = 42,
    max_train_examples: int | None = DEFAULT_MAX_TRAIN_EXAMPLES,
    max_test_examples: int | None = DEFAULT_MAX_TEST_EXAMPLES,
) -> TextClassificationDataset:
    try:
        from datasets import ClassLabel, load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package to load HuggingFace benchmarks.") from exc

    dataset_args = (config.dataset_id,) if config.config_name is None else (config.dataset_id, config.config_name)
    train_split = load_dataset(*dataset_args, split=config.train_split, streaming=True)
    try:
        test_split = load_dataset(*dataset_args, split=config.test_split, streaming=True)
    except ValueError:
        test_split = load_dataset(*dataset_args, split="validation", streaming=True)

    features = getattr(train_split, "features", None)
    label_feature = features.get(config.label_column) if features is not None else None
    label_names = (
        list(label_feature.names)
        if isinstance(label_feature, ClassLabel)
        else list(config.label_names or [])
    )
    label_names = label_names or None

    train_split = _shuffle_and_take_hf_split(train_split, max_train_examples, seed)
    test_split = _shuffle_and_take_hf_split(test_split, max_test_examples, seed)
    train_texts, train_labels = _hf_texts_and_labels(
        train_split,
        config.text_columns,
        config.label_column,
        label_names,
    )
    test_texts, test_labels = _hf_texts_and_labels(
        test_split,
        config.text_columns,
        config.label_column,
        label_names,
    )
    return TextClassificationDataset(
        name=config.name,
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        label_names=label_names,
    )


def _shuffle_and_take_hf_split(split: object, max_examples: int | None, seed: int) -> object:
    if max_examples is None:
        return split
    if hasattr(split, "shuffle"):
        split = split.shuffle(buffer_size=10_000, seed=seed)
    if hasattr(split, "take"):
        split = split.take(max_examples)
    return split


def _hf_texts_and_labels(
    split: object,
    text_columns: tuple[str, ...],
    label_column: str,
    label_names: list[str] | None,
) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for row in split:
        parts = [str(row[column]).strip() for column in text_columns if str(row[column]).strip()]
        if not parts:
            continue
        texts.append(" ".join(parts))
        labels.append(_normalize_hf_label(row[label_column], label_names))
    if not texts:
        raise ValueError("HuggingFace split did not contain any usable text rows.")
    return texts, labels


def _normalize_hf_label(value: object, label_names: list[str] | None) -> str:
    if label_names is not None:
        try:
            return label_names[int(value)]
        except (TypeError, ValueError, IndexError):
            pass
    return str(value).strip()


def _cap_examples(
    texts: list[str],
    labels: list[str],
    max_examples: int | None,
    seed: int,
) -> tuple[list[str], list[str]]:
    if max_examples is None or len(texts) <= max_examples:
        return list(texts), list(labels)
    stratify = labels if _can_stratify(labels, max_examples) else None
    sampled_texts, _, sampled_labels, _ = train_test_split(
        texts,
        labels,
        train_size=max_examples,
        random_state=seed,
        stratify=stratify,
    )
    return list(sampled_texts), list(sampled_labels)


def _can_stratify(labels: list[str], max_examples: int) -> bool:
    counts = pd.Series(labels).value_counts()
    return bool(not counts.empty and counts.min() >= 2 and len(counts) <= max_examples)


def _read_phrasebank_source(path: Path) -> tuple[list[str], list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return _read_phrasebank_zip(path)
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        return _texts_and_labels_from_dataframe(df)
    if suffix == ".csv":
        df = pd.read_csv(path, encoding="utf-8")
        return _texts_and_labels_from_dataframe(df)
    return _read_phrasebank_text(path.read_text(encoding="latin-1"))


def _read_phrasebank_zip(path: Path) -> tuple[list[str], list[str]]:
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            name for name in archive.namelist()
            if name.lower().endswith(".txt") and "sentences" in name.lower()
        )
        if not names:
            raise ValueError(f"No PhraseBank sentence text file found in {path}.")
        preferred = [name for name in names if "allagree" in name.lower()]
        name = preferred[0] if preferred else names[0]
        content = archive.read(name).decode("latin-1")
    return _read_phrasebank_text(content)


def _read_phrasebank_text(content: str) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or "@" not in line:
            continue
        text, label = line.rsplit("@", 1)
        label = label.strip().lower()
        if label not in FINANCIAL_PHRASEBANK_LABELS:
            continue
        texts.append(text.strip())
        labels.append(label)
    if not texts:
        raise ValueError("No Financial PhraseBank rows found.")
    return texts, labels


def _texts_and_labels_from_dataframe(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    lower_columns = {column.lower(): column for column in df.columns}
    text_column = lower_columns.get("text") or lower_columns.get("sentence")
    label_column = lower_columns.get("label") or lower_columns.get("sentiment")
    if text_column is None or label_column is None:
        if len(df.columns) >= 2:
            label_column, text_column = df.columns[:2]
        else:
            raise ValueError("Expected text/sentence and label/sentiment columns.")

    texts = df[text_column].astype(str).tolist()
    labels = [_normalize_label(value) for value in df[label_column].tolist()]
    keep = [idx for idx, label in enumerate(labels) if label in FINANCIAL_PHRASEBANK_LABELS]
    if not keep:
        raise ValueError("No supported Financial PhraseBank labels found.")
    return [texts[idx] for idx in keep], [labels[idx] for idx in keep]


def _normalize_label(value: object) -> str:
    if isinstance(value, int):
        return FINANCIAL_PHRASEBANK_LABELS[value]
    text = str(value).strip().lower()
    if text in {"0", "1", "2"}:
        return FINANCIAL_PHRASEBANK_LABELS[int(text)]
    return text
