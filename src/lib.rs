//! # prompt-cache-key
//!
//! Stable Anthropic prompt-cache scope hashes.
//!
//! Anthropic's prompt cache hits when the prefix (system + tools, up to a
//! `cache_control` breakpoint) is byte-identical to a previously seen
//! request. Coordinating that across workers needs a deterministic scope
//! key everyone can compute locally.
//!
//! [`compute_cache_key`] walks `(model, system, tools)` into a canonical
//! byte stream and returns a SHA-256 hex digest prefixed with the model:
//!
//! ```
//! use prompt_cache_key::{compute_cache_key, System};
//!
//! let key = compute_cache_key("claude-opus-4-7", System::Text("You are helpful."), None);
//! assert!(key.starts_with("anthropic-cache:claude-opus-4-7:sha256:"));
//! ```
//!
//! Anything AFTER the last `cache_control` breakpoint in `system` is
//! excluded from the key because it isn't part of the cached scope.
//! [`find_breakpoints`] returns the indices of those markers if you need
//! to inspect them yourself.
//!
//! ```
//! use prompt_cache_key::find_breakpoints;
//! use serde_json::json;
//!
//! let blocks = json!([
//!     {"type": "text", "text": "a"},
//!     {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
//!     {"type": "text", "text": "c"},
//! ]);
//! assert_eq!(find_breakpoints(&blocks), vec![1]);
//! ```
//!
//! Companion to [`llm-message-hash`](https://crates.io/crates/llm-message-hash),
//! which hashes the full request for idempotency rather than just the
//! cache scope.

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

pub const KEY_PREFIX: &str = "anthropic-cache";

/// System prompt input form. Anthropic accepts either a plain string or a
/// list of content blocks; this enum mirrors that surface.
pub enum System<'a> {
    /// Treated as a single `{"type": "text", "text": ...}` block.
    Text(&'a str),
    /// Pre-built content blocks. Must be a JSON array of objects.
    Blocks(&'a Value),
    /// Same as passing `None` to the Python API.
    None,
}

/// Return zero-based indices of blocks carrying a non-null `cache_control`.
///
/// Accepts any [`Value`]; non-arrays and non-object entries are ignored.
pub fn find_breakpoints(blocks: &Value) -> Vec<usize> {
    let Some(arr) = blocks.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .enumerate()
        .filter_map(|(i, b)| match b.as_object() {
            Some(o) => match o.get("cache_control") {
                Some(v) if !v.is_null() => Some(i),
                _ => None,
            },
            None => None,
        })
        .collect()
}

/// Return the prefix of `blocks` up to and including the LAST
/// `cache_control` marker. If no marker is present, the full list is
/// returned. Non-arrays produce an empty list.
pub fn scope_blocks(blocks: &Value) -> Vec<Value> {
    let Some(arr) = blocks.as_array() else {
        return Vec::new();
    };
    if arr.is_empty() {
        return Vec::new();
    }
    let bps = find_breakpoints(blocks);
    let end = match bps.last() {
        Some(&last) => last + 1,
        None => arr.len(),
    };
    arr[..end].to_vec()
}

/// Stable scope key for `(model, system, tools)`.
///
/// Returns `"{KEY_PREFIX}:{model}:sha256:{hex}"`. `tools` must be a JSON
/// array (or `None`); each element is canonicalized.
///
/// Everything after the last `cache_control` in `system` is dropped to
/// match Anthropic's cached prefix. Tools always participate because
/// Anthropic includes them in the cached prefix.
pub fn compute_cache_key(model: &str, system: System<'_>, tools: Option<&Value>) -> String {
    let scoped = scope_blocks(&normalize_system(system));
    let tools_vec: Vec<Value> = match tools.and_then(|v| v.as_array()) {
        Some(arr) => arr.to_vec(),
        None => Vec::new(),
    };

    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(model.to_string()));
    body.insert("system".to_string(), Value::Array(scoped));
    body.insert("tools".to_string(), Value::Array(tools_vec));
    let body = Value::Object(body);

    let blob = canonical_json(&body);
    let mut hasher = Sha256::new();
    hasher.update(blob.as_bytes());
    let digest = hex_lower(&hasher.finalize());
    format!("{KEY_PREFIX}:{model}:sha256:{digest}")
}

/// Serialize `value` as JSON with recursively sorted object keys and the
/// compact separator `","` / `":"` (no spaces). This matches Python's
/// `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
pub fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn normalize_system(system: System<'_>) -> Value {
    match system {
        System::None => Value::Array(Vec::new()),
        System::Text(s) => {
            let mut block = Map::new();
            block.insert("type".to_string(), Value::String("text".to_string()));
            block.insert("text".to_string(), Value::String(s.to_string()));
            Value::Array(vec![Value::Object(block)])
        }
        System::Blocks(v) => match v.as_array() {
            Some(arr) => Value::Array(arr.clone()),
            None => Value::Array(Vec::new()),
        },
    }
}

fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => write_json_string(s, out),
        Value::Array(arr) => {
            out.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json_string(k, out);
                out.push(':');
                write_canonical(&map[*k], out);
            }
            out.push('}');
        }
    }
}

fn write_json_string(s: &str, out: &mut String) {
    // Mirrors Python json.dumps with ensure_ascii=False: escapes only
    // control chars, quote, and backslash; passes non-ASCII through.
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let code = c as u32;
                out.push_str(&format!("\\u{code:04x}"));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}
