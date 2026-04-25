# ExTES Preprocessing Schema

This repo supports two ExTES ingestion paths for the research pipeline:

- `--examples-source extes_hf`
- `--examples-source extes_jsonl`

## Supported Hugging Face format

For `extes_hf`, the loader currently supports the public instruction-style
schema exposed by `ailover/ExTES`, where each row contains:

- `example`: a serialized training string of the form
  `<s>[INST] ... Please respond the following request: {user} [/INST] {assistant}</s>`

Source used for schema grounding:

- https://huggingface.co/datasets/ailover/ExTES

The loader parses that into the normalized internal format below.

## Normalized internal schema

Every dataset source is converted into this normalized `SeedExample` shape:

```json
{
  "example_id": "string",
  "source": "extes_hf | extes_jsonl | esconv_hf | jsonl | tasks",
  "scenario_brief": "string",
  "desired_outcome": "string",
  "context_turns": [
    {"role": "user", "content": "string"},
    {"role": "assistant", "content": "string"}
  ],
  "ground_truth_response": "string or null",
  "emotion_type": "string or null",
  "problem_type": "string or null"
}
```

## Supported JSONL format

For `extes_jsonl`, the recommended schema is:

```json
{
  "example_id": "extes-0001",
  "scenario_brief": "User is struggling after an argument with a close friend.",
  "desired_outcome": "Help the user feel heard and emotionally supported.",
  "context_turns": [
    {"role": "user", "content": "I'm really upset after a fight with my friend."}
  ],
  "ground_truth_response": "I'm sorry that happened. That kind of conflict can hurt a lot."
}
```

An alternative serialized form is also supported:

```json
{
  "example": "<s>[INST] ... Please respond the following request: I'm feeling lost. [/INST] I'm sorry you're going through that...</s>"
}
```

## Notes

- The public ExTES HF variant above is instruction-tuned data, not a richly
  structured multi-turn schema like ESConv.
- If you obtain a more canonical ExTES release with explicit dialogue turns and
  metadata, the preferred path is to add a dedicated loader that preserves that
  structure instead of flattening it through the instruction parser.
