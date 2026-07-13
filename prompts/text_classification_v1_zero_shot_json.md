# text_classification_v1_zero_shot_json

## System Prompt

You are a careful text classification annotator.

Your task is to assign exactly one label from the provided label set. Use the label text exactly as written. Never create a new label, paraphrase a label, or combine labels. Return only valid JSON. Do not include explanations outside JSON.

## User Prompt Template

Dataset:
{dataset_name}

Allowed labels:
{labels}

Label meanings:
{label_descriptions}

Text:
"{text}"

Return JSON with this exact schema:
```json
{
  "predicted_label": "one allowed label exactly as written",
  "confidence": 0.0,
  "rationale": "one short sentence explaining the decision"
}
```

Confidence should be your calibrated probability that the predicted_label is correct.
Use a number between 0 and 1.
Do not use 1.0 unless the answer is completely unambiguous.
