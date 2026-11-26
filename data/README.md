# Dataset notes

Place your own UTF-8 JSONL conversation dataset in this folder or mount it in Kaggle.

Each line must be either:
- `{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`
- `{"prompt":"...","response":"..."}`

Keep conversations in English for the initial Juvao training objective. Include varied, high-quality examples; avoid repeated boilerplate and contradictory answers. Do not include personal data, credentials, private conversations, or content you do not have rights to use.

The small `raw/example.jsonl` file is a format example only and is not a meaningful training corpus. A tiny dataset is suitable only for testing that the pipeline runs.
