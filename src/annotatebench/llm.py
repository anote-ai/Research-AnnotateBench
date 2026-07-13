from __future__ import annotations

import json
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError

from .costs import LlmTokenPricing, estimate_llm_annotation_cost
from .pilot import TextClassificationDataset


DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"
GENERIC_PROMPT_VERSION = "text_classification_v1_zero_shot_json"
DATASET_PROMPT_VERSIONS = {
    "financial_phrasebank": "financial_phrasebank_v1_zero_shot_json",
}
DATASET_LABEL_DESCRIPTIONS = {
    "ag_news": {
        "World": "world news and international affairs",
        "Sports": "sports news",
        "Business": "business, markets, companies, or the economy",
        "Sci/Tech": "science or technology news",
    },
    "sst2": {
        "0": "negative movie-review sentiment",
        "1": "positive movie-review sentiment",
    },
    "rotten_tomatoes": {
        "0": "negative movie-review sentiment",
        "1": "positive movie-review sentiment",
    },
    "yelp_polarity": {
        "1": "negative review sentiment",
        "2": "positive review sentiment",
    },
    "tweet_eval_sentiment": {
        "negative": "negative tweet sentiment",
        "neutral": "neutral tweet sentiment",
        "positive": "positive tweet sentiment",
    },
    "emotion": {
        "sadness": "sadness",
        "joy": "joy",
        "love": "love",
        "anger": "anger",
        "fear": "fear",
        "surprise": "surprise",
    },
    "trec": {
        "ABBR": "abbreviation question",
        "DESC": "description or definition question",
        "ENTY": "entity question",
        "HUM": "human/person question",
        "LOC": "location question",
        "NUM": "numeric answer question",
    },
    "twenty_newsgroups": {
        "0": "alt.atheism, discussion about atheism",
        "1": "comp.graphics, computer graphics",
        "2": "comp.os.ms-windows.misc, Microsoft Windows operating system",
        "3": "comp.sys.ibm.pc.hardware, IBM PC compatible hardware",
        "4": "comp.sys.mac.hardware, Apple Macintosh hardware",
        "5": "comp.windows.x, X Window System",
        "6": "misc.forsale, items for sale",
        "7": "rec.autos, automobiles",
        "8": "rec.motorcycles, motorcycles",
        "9": "rec.sport.baseball, baseball",
        "10": "rec.sport.hockey, hockey",
        "11": "sci.crypt, cryptography",
        "12": "sci.electronics, electronics",
        "13": "sci.med, medicine and health",
        "14": "sci.space, space and astronomy",
        "15": "soc.religion.christian, Christianity",
        "16": "talk.politics.guns, gun politics",
        "17": "talk.politics.mideast, Middle East politics",
        "18": "talk.politics.misc, miscellaneous politics",
        "19": "talk.religion.misc, miscellaneous religion",
    },
}


@dataclass(frozen=True)
class LlmAnnotation:
    predicted_label: str
    confidence: float
    rationale: str
    raw_response: dict
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None


def dataset_label_names(dataset: TextClassificationDataset) -> list[str]:
    if dataset.label_names:
        return list(dataset.label_names)
    return sorted(set(dataset.train_labels) | set(dataset.test_labels))


def default_prompt_version(dataset_name: str) -> str:
    return DATASET_PROMPT_VERSIONS.get(dataset_name, GENERIC_PROMPT_VERSION)


def load_prompt(prompt_dir: str | Path, prompt_version: str) -> tuple[str, str]:
    path = Path(prompt_dir) / f"{prompt_version}.md"
    content = path.read_text(encoding="utf-8")
    system_marker = "## System Prompt"
    user_marker = "## User Prompt Template"
    if system_marker not in content or user_marker not in content:
        raise ValueError(f"Prompt file is missing required sections: {path}")
    _, after_system = content.split(system_marker, 1)
    system_prompt, user_prompt = after_system.split(user_marker, 1)
    return system_prompt.strip(), user_prompt.strip()


def format_label_list(label_names: list[str]) -> str:
    return "\n".join(f"- {label}" for label in label_names)


def format_label_descriptions(dataset_name: str, label_names: list[str]) -> str:
    descriptions = DATASET_LABEL_DESCRIPTIONS.get(dataset_name, {})
    lines = []
    for label in label_names:
        description = descriptions.get(str(label))
        lines.append(f"- {label}: {description}" if description else f"- {label}")
    return "\n".join(lines)


def build_user_prompt(
    template: str,
    *,
    text: str,
    dataset_name: str,
    label_names: list[str],
) -> str:
    return (
        template.replace("{text}", text)
        .replace("{dataset_name}", dataset_name)
        .replace("{labels}", format_label_list(label_names))
        .replace("{label_descriptions}", format_label_descriptions(dataset_name, label_names))
    )


def call_chat_completion(
    *,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
    retry_sleep_seconds: float = 2.0,
) -> dict:
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    api_request = request.Request(
        api_url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    ssl_context = default_ssl_context()
    attempts = max(1, max_retries + 1)
    for attempt in range(attempts):
        try:
            with request.urlopen(api_request, timeout=60, context=ssl_context) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt == attempts - 1:
                raise RuntimeError(f"API request failed with HTTP {exc.code}: {detail}") from exc
            time.sleep(retry_sleep_seconds * (2**attempt))
        except URLError as exc:
            if attempt == attempts - 1:
                raise RuntimeError(f"API request failed: {exc}") from exc
            time.sleep(retry_sleep_seconds * (2**attempt))

    content = response_payload["choices"][0]["message"]["content"]
    annotation = json.loads(content)
    annotation["_raw_response"] = response_payload
    return annotation


def default_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def dry_run_annotation(label_names: list[str]) -> dict:
    return {
        "predicted_label": label_names[0],
        "confidence": 0.34,
        "rationale": "Dry run placeholder annotation.",
        "_raw_response": {},
    }


def normalize_label(label: object, label_names: list[str]) -> str:
    raw_label = str(label).strip()
    by_casefold = {candidate.casefold(): candidate for candidate in label_names}
    normalized = by_casefold.get(raw_label.casefold())
    if normalized is None:
        raise ValueError(f"Unsupported predicted_label: {raw_label!r}")
    return normalized


def normalize_annotation(
    annotation: dict,
    label_names: list[str],
    pricing: LlmTokenPricing | None = None,
) -> LlmAnnotation:
    label = normalize_label(annotation.get("predicted_label", ""), label_names)
    confidence = float(annotation.get("confidence"))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1]: {confidence}")
    raw_response = annotation.get("_raw_response", annotation)
    return LlmAnnotation(
        predicted_label=label,
        confidence=confidence,
        rationale=str(annotation.get("rationale", "")).strip(),
        raw_response=raw_response,
        input_tokens=response_input_tokens(raw_response),
        output_tokens=response_output_tokens(raw_response),
        total_tokens=response_total_tokens(raw_response),
        cost_usd=estimate_response_cost(raw_response, pricing),
    )


def response_usage(raw_response: dict) -> dict:
    return raw_response.get("usage", {})


def response_input_tokens(raw_response: dict) -> int:
    return int(response_usage(raw_response).get("prompt_tokens", 0))


def response_output_tokens(raw_response: dict) -> int:
    return int(response_usage(raw_response).get("completion_tokens", 0))


def response_total_tokens(raw_response: dict) -> int:
    return int(response_usage(raw_response).get("total_tokens", 0))


def estimate_response_cost(raw_response: dict, pricing: LlmTokenPricing | None) -> float | None:
    if pricing is None:
        return None
    return estimate_llm_annotation_cost(
        response_input_tokens(raw_response),
        response_output_tokens(raw_response),
        pricing,
    )


def make_pricing(
    *,
    model: str,
    input_usd_per_million_tokens: float,
    output_usd_per_million_tokens: float,
) -> LlmTokenPricing | None:
    if input_usd_per_million_tokens <= 0 and output_usd_per_million_tokens <= 0:
        return None
    return LlmTokenPricing(
        source_name="configured API token pricing",
        source_url="",
        checked_at="",
        model=model,
        input_usd_per_million_tokens=input_usd_per_million_tokens,
        output_usd_per_million_tokens=output_usd_per_million_tokens,
    )
