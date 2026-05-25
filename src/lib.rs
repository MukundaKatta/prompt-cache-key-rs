/*!
prompt-cache-key: stable Anthropic prompt cache-scope hashes.

Hash the scope-clipped system prompt, tool list, and model to produce a
stable key that identifies which Anthropic prompt cache bucket a request
will hit. Distinct from `llm-message-hash` which hashes the full request
including user messages.

```rust
use prompt_cache_key::{CacheKey, HashOpts};
use serde_json::json;

let system = json!("You are a helpful assistant.");
let tools: Vec<serde_json::Value> = vec![];
let key = CacheKey::compute("claude-sonnet-4-6", &system, &tools, HashOpts::default());
assert_eq!(key.hex.len(), 64);
```
*/

use sha2::{Digest, Sha256};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

// ---- canonical JSON ---------------------------------------------------

fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(m) => {
            let sorted: BTreeMap<&String, &Value> = m.iter().collect();
            let out: Map<String, Value> =
                sorted.into_iter().map(|(k, v)| (k.clone(), canonical(v))).collect();
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

fn sha256_hex(data: &str) -> String {
    let mut h = Sha256::new();
    h.update(data.as_bytes());
    format!("{:x}", h.finalize())
}

// ---- drop fields -------------------------------------------------------

/// Options for computing the cache key.
#[derive(Debug, Clone, Default)]
pub struct HashOpts {
    /// Extra field names to strip from every object before hashing.
    pub drop_fields: Vec<String>,
}

fn drop_object_fields(value: &Value, drop: &[String]) -> Value {
    match value {
        Value::Object(m) => {
            let out: Map<String, Value> = m
                .iter()
                .filter(|(k, _)| !drop.contains(k))
                .map(|(k, v)| (k.clone(), drop_object_fields(v, drop)))
                .collect();
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(|v| drop_object_fields(v, drop)).collect()),
        other => other.clone(),
    }
}

// ---- CacheKey ---------------------------------------------------------

/// A computed cache key for an Anthropic prompt cache scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKey {
    /// Lowercase hex SHA-256 of the canonical scope payload.
    pub hex: String,
    /// The model that was hashed.
    pub model: String,
}

impl CacheKey {
    /// Compute the cache key from model + system + tools.
    ///
    /// The system prompt and tool list are canonicalized (keys sorted
    /// recursively), optional fields stripped, then serialized to compact JSON
    /// and hashed together with the model name.
    pub fn compute(model: &str, system: &Value, tools: &[Value], opts: HashOpts) -> Self {
        let sys_clean = drop_object_fields(system, &opts.drop_fields);
        let sys_canon = canonical(&sys_clean);

        let tools_clean: Vec<Value> = tools
            .iter()
            .map(|t| drop_object_fields(t, &opts.drop_fields))
            .collect();
        let tools_canon = canonical(&Value::Array(tools_clean));

        let payload = format!(
            "model={}\nsystem={}\ntools={}",
            model,
            serde_json::to_string(&sys_canon).unwrap_or_default(),
            serde_json::to_string(&tools_canon).unwrap_or_default(),
        );

        CacheKey {
            hex: sha256_hex(&payload),
            model: model.to_owned(),
        }
    }

    /// Compute from a raw string system prompt (no JSON canonicalization).
    pub fn compute_str(model: &str, system: &str, tools: &[Value], opts: HashOpts) -> Self {
        Self::compute(model, &Value::String(system.to_owned()), tools, opts)
    }
}

impl std::fmt::Display for CacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.hex)
    }
}

/// Compute a cache key and return just the hex string.
pub fn cache_key(model: &str, system: &Value, tools: &[Value]) -> String {
    CacheKey::compute(model, system, tools, HashOpts::default()).hex
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hex_is_64_chars() {
        let k = CacheKey::compute("model", &json!("system"), &[], HashOpts::default());
        assert_eq!(k.hex.len(), 64);
    }

    #[test]
    fn same_inputs_same_key() {
        let a = CacheKey::compute("model", &json!("sys"), &[], HashOpts::default());
        let b = CacheKey::compute("model", &json!("sys"), &[], HashOpts::default());
        assert_eq!(a, b);
    }

    #[test]
    fn different_model_different_key() {
        let a = CacheKey::compute("model-a", &json!("sys"), &[], HashOpts::default());
        let b = CacheKey::compute("model-b", &json!("sys"), &[], HashOpts::default());
        assert_ne!(a.hex, b.hex);
    }

    #[test]
    fn different_system_different_key() {
        let a = CacheKey::compute("model", &json!("sys1"), &[], HashOpts::default());
        let b = CacheKey::compute("model", &json!("sys2"), &[], HashOpts::default());
        assert_ne!(a.hex, b.hex);
    }

    #[test]
    fn key_order_independent() {
        let sys_a = json!({"b": 2, "a": 1});
        let sys_b = json!({"a": 1, "b": 2});
        let a = CacheKey::compute("model", &sys_a, &[], HashOpts::default());
        let b = CacheKey::compute("model", &sys_b, &[], HashOpts::default());
        assert_eq!(a.hex, b.hex);
    }

    #[test]
    fn tools_order_matters() {
        let t1 = json!({"name": "tool_a"});
        let t2 = json!({"name": "tool_b"});
        let a = CacheKey::compute("model", &json!("sys"), &[t1.clone(), t2.clone()], HashOpts::default());
        let b = CacheKey::compute("model", &json!("sys"), &[t2.clone(), t1.clone()], HashOpts::default());
        assert_ne!(a.hex, b.hex);
    }

    #[test]
    fn drop_fields_excluded() {
        let sys_with = json!({"text": "hello", "cache_control": {"type": "ephemeral"}});
        let sys_without = json!({"text": "hello"});
        let opts = HashOpts { drop_fields: vec!["cache_control".to_string()] };
        let a = CacheKey::compute("model", &sys_with, &[], opts);
        let b = CacheKey::compute("model", &sys_without, &[], HashOpts::default());
        assert_eq!(a.hex, b.hex);
    }

    #[test]
    fn compute_str_matches_json_string() {
        let a = CacheKey::compute_str("model", "hello", &[], HashOpts::default());
        let b = CacheKey::compute("model", &json!("hello"), &[], HashOpts::default());
        assert_eq!(a.hex, b.hex);
    }

    #[test]
    fn model_stored() {
        let k = CacheKey::compute("my-model", &json!("s"), &[], HashOpts::default());
        assert_eq!(k.model, "my-model");
    }

    #[test]
    fn display_is_hex() {
        let k = CacheKey::compute("m", &json!("s"), &[], HashOpts::default());
        assert_eq!(k.to_string(), k.hex);
    }

    #[test]
    fn cache_key_fn_wrapper() {
        let hex = cache_key("m", &json!("s"), &[]);
        assert_eq!(hex.len(), 64);
    }

    #[test]
    fn tools_affect_key() {
        let a = CacheKey::compute("model", &json!("sys"), &[], HashOpts::default());
        let b = CacheKey::compute("model", &json!("sys"), &[json!({"name": "t"})], HashOpts::default());
        assert_ne!(a.hex, b.hex);
    }

    #[test]
    fn nested_key_order_independent() {
        let a = json!({"outer": {"b": 2, "a": 1}});
        let b = json!({"outer": {"a": 1, "b": 2}});
        let ka = CacheKey::compute("m", &a, &[], HashOpts::default());
        let kb = CacheKey::compute("m", &b, &[], HashOpts::default());
        assert_eq!(ka.hex, kb.hex);
    }

    #[test]
    fn empty_system_and_tools() {
        let k = CacheKey::compute("model", &json!(null), &[], HashOpts::default());
        assert_eq!(k.hex.len(), 64);
    }

    #[test]
    fn all_models_produce_different_keys() {
        let models = ["claude-opus-4-7", "claude-sonnet-4-6", "gpt-4o", "gemini-2.5-pro"];
        let keys: Vec<String> = models
            .iter()
            .map(|m| CacheKey::compute(m, &json!("sys"), &[], HashOpts::default()).hex)
            .collect();
        let unique: std::collections::HashSet<_> = keys.iter().collect();
        assert_eq!(unique.len(), models.len());
    }
}
