# prompt-cache-key

[![Crates.io](https://img.shields.io/crates/v/prompt-cache-key.svg)](https://crates.io/crates/prompt-cache-key)
[![Documentation](https://docs.rs/prompt-cache-key/badge.svg)](https://docs.rs/prompt-cache-key)
[![License](https://img.shields.io/crates/l/prompt-cache-key.svg)](https://crates.io/crates/prompt-cache-key)

Stable Anthropic prompt-cache scope hashes.

Anthropic's prompt cache hits when the prefix (system plus tools, up to a
`cache_control` breakpoint) is byte-identical to a previously seen request.
This crate produces a deterministic key from `(model, system, tools)` that
survives benign reordering of JSON keys.

## Install

```toml
[dependencies]
prompt-cache-key = "0.1"
serde_json = "1"
```

## Use

```rust
use prompt_cache_key::{compute_cache_key, System};
use serde_json::json;

let key = compute_cache_key(
    "claude-opus-4-7",
    System::Text("You are helpful."),
    Some(&json!([
        {"name": "search", "description": "", "input_schema": {}},
    ])),
);
assert!(key.starts_with("anthropic-cache:claude-opus-4-7:sha256:"));
```

When `system` contains a `cache_control` marker, anything after the last
marker is excluded from the key, matching Anthropic's cached prefix.

```rust
use prompt_cache_key::{compute_cache_key, System};
use serde_json::json;

let blocks = json!([
    {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": "this changes per request"},
]);
let key = compute_cache_key("claude-opus-4-7", System::Blocks(&blocks), None);
```

Pairs with [`llm-message-hash`](https://crates.io/crates/llm-message-hash)
(hashes the full request, not just the cache scope).

## License

MIT
