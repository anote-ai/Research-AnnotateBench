from __future__ import annotations

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
