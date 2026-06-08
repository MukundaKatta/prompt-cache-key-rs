# prompt-cache-key

[![Crates.io](https://img.shields.io/crates/v/prompt-cache-key.svg)](https://crates.io/crates/prompt-cache-key)
[![Documentation](https://docs.rs/prompt-cache-key/badge.svg)](https://docs.rs/prompt-cache-key)
[![License](https://img.shields.io/crates/l/prompt-cache-key.svg)](https://crates.io/crates/prompt-cache-key)

Stable Anthropic prompt-cache scope hashes.

Anthropic's prompt cache hits when the prefix (system plus tools, up to a
`cache_control` breakpoint) is byte-identical to a previously seen request.
This crate produces a deterministic key from `(model, system, tools)` that
survives benign reordering of JSON keys.

## Why

Anthropic's [prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
keys off the *exact bytes* of the cached prefix. Two requests that are
semantically identical but differ in JSON key order, or that vary only in the
mutable tail after the last `cache_control` breakpoint, still hit the same
cache on Anthropic's side — yet a naive `serde_json::to_string` of the request
would produce different strings for them. This crate computes a key that
matches Anthropic's caching behavior:

- **Key order insensitive** — object keys are sorted recursively before
  hashing, so `{"a":1,"b":2}` and `{"b":2,"a":1}` collapse to one key.
- **Breakpoint aware** — only the content up to and including the last
  `cache_control` marker (the part Anthropic actually caches) is hashed.
- **Deterministic** — the same `(model, system, tools)` always yields the
  same key, across processes and machines.

Use it to deduplicate requests, pre-warm a cache, or build a local hit/miss
metric without round-tripping to the API.

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
marker is excluded from the key, matching Anthropic's cached prefix. The two
requests below share a cache key because they differ only in the mutable tail:

```rust
use prompt_cache_key::{compute_cache_key, System};
use serde_json::json;

let req_a = json!([
    {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": "today is Monday"},
]);
let req_b = json!([
    {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": "today is Tuesday"},
]);

let key_a = compute_cache_key("claude-opus-4-7", System::Blocks(&req_a), None);
let key_b = compute_cache_key("claude-opus-4-7", System::Blocks(&req_b), None);
assert_eq!(key_a, key_b); // same cached prefix -> same key
```

## API

| Item | Description |
| --- | --- |
| `compute_cache_key(model, system, tools) -> String` | The main entry point. Returns `anthropic-cache:<model>:sha256:<hex>` for the cache-scoped prefix of the request. `tools` is `Option<&Value>` and defaults to an empty array when `None`. |
| `enum System<'a>` | The system prompt: `None`, `Text(&str)` (a single text block), or `Blocks(&Value)` (a content-block array that may carry `cache_control` markers). |
| `find_breakpoints(blocks: &Value) -> Vec<usize>` | Indices of blocks carrying a non-null `cache_control` marker. Non-array input yields an empty list. |
| `scope_blocks(blocks: &Value) -> Vec<Value>` | The cache-scoped prefix: every block up to and including the last breakpoint (or all blocks when there is none). |
| `canonical_json(value: &Value) -> String` | Compact JSON with all object keys sorted recursively. Array order is preserved. |
| `KEY_PREFIX: &str` | The `"anthropic-cache"` constant every key starts with. |

### Behavior notes

- `compute_cache_key(.., None)` and `compute_cache_key(.., Some(&json!([])))`
  produce the same key — an absent tools list is treated as empty.
- A `System::Blocks` holding a non-array (or `null`) value contributes no
  scoped content and hashes the same as `System::None`.
- Tool/array order *is* significant; only object key order is normalized.

Pairs with [`llm-message-hash`](https://crates.io/crates/llm-message-hash)
(hashes the full request, not just the cache scope).

## License

MIT
