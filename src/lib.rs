/*!
prompt-cache-key: stable Anthropic prompt cache-scope hashes.

Anthropic's prompt cache hits when the prefix (system plus tools, up to a
`cache_control` breakpoint) is byte-identical to a previously seen request.
This crate produces a deterministic key from `(model, system, tools)` that
survives benign reordering of JSON keys. Distinct from `llm-message-hash`
which hashes the full request including user messages.

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
assert!(key.starts_with("anthropic-cache:claude-opus-4-7:sha256:"));
```
*/

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Prefix that every computed cache key starts with.
pub const KEY_PREFIX: &str = "anthropic-cache";

/// The system prompt for a request, in any of the forms the Anthropic API accepts.
#[derive(Debug, Clone, Copy)]
pub enum System<'a> {
    /// No system prompt.
    None,
    /// A plain-string system prompt. Treated as a single text block.
    Text(&'a str),
    /// A JSON array of content blocks, possibly carrying `cache_control` markers.
    Blocks(&'a Value),
}

// ---- canonical JSON ---------------------------------------------------

/// Recursively sort object keys so semantically identical JSON produces an
/// identical `Value`.
fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Object(m) => {
            let sorted: BTreeMap<&String, &Value> = m.iter().collect();
            let out: Map<String, Value> = sorted
                .into_iter()
                .map(|(k, v)| (k.clone(), canonicalize(v)))
                .collect();
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(canonicalize).collect()),
        other => other.clone(),
    }
}

/// Serialize a JSON value to a canonical, compact string with all object keys
/// sorted recursively.
///
/// This makes hashing insensitive to benign key reordering. Control characters
/// are escaped exactly as in standard JSON, and non-ASCII characters pass
/// through unescaped.
pub fn canonical_json(value: &Value) -> String {
    serde_json::to_string(&canonicalize(value)).unwrap_or_default()
}

// ---- breakpoints / scope ----------------------------------------------

/// Return the indices of blocks that carry a non-null `cache_control` marker.
///
/// A non-array (or `null`) input yields an empty list.
pub fn find_breakpoints(blocks: &Value) -> Vec<usize> {
    match blocks {
        Value::Array(arr) => arr
            .iter()
            .enumerate()
            .filter_map(|(i, block)| match block.get("cache_control") {
                Some(cc) if !cc.is_null() => Some(i),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Return the cache-scoped prefix of `blocks`: everything up to and including
/// the last `cache_control` breakpoint.
///
/// If there is no breakpoint, every block is returned (the whole system is part
/// of the cache scope). A non-array (or `null`) input yields an empty vector.
pub fn scope_blocks(blocks: &Value) -> Vec<Value> {
    match blocks {
        Value::Array(arr) => {
            let breakpoints = find_breakpoints(blocks);
            let end = match breakpoints.last() {
                Some(&last) => last + 1,
                None => arr.len(),
            };
            arr[..end].to_vec()
        }
        _ => Vec::new(),
    }
}

// ---- key computation --------------------------------------------------

/// Normalize a [`System`] into the list of content blocks that fall inside the
/// cache scope.
fn system_scope_blocks(system: System<'_>) -> Vec<Value> {
    match system {
        System::None => Vec::new(),
        System::Text(text) => vec![serde_json::json!({"type": "text", "text": text})],
        System::Blocks(blocks) => scope_blocks(blocks),
    }
}

/// Compute a stable cache-scope key for a request.
///
/// The key is `anthropic-cache:<model>:sha256:<hex>`, where the hex is the
/// SHA-256 of the canonical JSON of `{model, system, tools}`. The `system`
/// payload is the cache-scoped prefix of the system blocks (a string system is
/// treated as a single text block), and `tools` defaults to an empty array when
/// `None`.
pub fn compute_cache_key(model: &str, system: System<'_>, tools: Option<&Value>) -> String {
    let scoped = system_scope_blocks(system);
    let tools_value = tools.cloned().unwrap_or_else(|| Value::Array(Vec::new()));

    let payload = serde_json::json!({
        "model": model,
        "system": scoped,
        "tools": tools_value,
    });

    let canon = canonical_json(&payload);
    let mut hasher = Sha256::new();
    hasher.update(canon.as_bytes());
    let hex = format!("{:x}", hasher.finalize());

    format!("{KEY_PREFIX}:{model}:sha256:{hex}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn key_prefix_is_anthropic_cache() {
        assert_eq!(KEY_PREFIX, "anthropic-cache");
    }

    #[test]
    fn none_system_produces_valid_key() {
        let k = compute_cache_key("m", System::None, None);
        assert!(k.starts_with("anthropic-cache:m:sha256:"));
        assert_eq!(k.rsplit(':').next().unwrap().len(), 64);
    }

    #[test]
    fn scope_blocks_handles_non_array() {
        assert!(scope_blocks(&json!("not an array")).is_empty());
        assert!(scope_blocks(&json!(42)).is_empty());
    }
}
