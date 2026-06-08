use prompt_cache_key::{
    canonical_json, compute_cache_key, find_breakpoints, scope_blocks, System, KEY_PREFIX,
};
use serde_json::{json, Value};

// ---- find_breakpoints --------------------------------------------------

#[test]
fn find_breakpoints_empty() {
    assert_eq!(find_breakpoints(&json!([])), Vec::<usize>::new());
    assert_eq!(find_breakpoints(&Value::Null), Vec::<usize>::new());
}

#[test]
fn find_breakpoints_no_markers() {
    let blocks = json!([
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]);
    assert_eq!(find_breakpoints(&blocks), Vec::<usize>::new());
}

#[test]
fn find_breakpoints_single_marker() {
    let blocks = json!([
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "c"},
    ]);
    assert_eq!(find_breakpoints(&blocks), vec![1]);
}

#[test]
fn find_breakpoints_multiple_markers() {
    let blocks = json!([
        {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "b"},
        {"type": "text", "text": "c", "cache_control": {"type": "ephemeral"}},
    ]);
    assert_eq!(find_breakpoints(&blocks), vec![0, 2]);
}

#[test]
fn find_breakpoints_null_cache_control_skipped() {
    let blocks = json!([
        {"type": "text", "text": "a", "cache_control": null},
        {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
    ]);
    assert_eq!(find_breakpoints(&blocks), vec![1]);
}

// ---- scope_blocks ------------------------------------------------------

#[test]
fn scope_blocks_includes_through_last_breakpoint() {
    let blocks = json!([
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "c"},
        {"type": "text", "text": "d", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "e"},
    ]);
    let out = scope_blocks(&blocks);
    let texts: Vec<&str> = out.iter().map(|b| b["text"].as_str().unwrap()).collect();
    assert_eq!(texts, vec!["a", "b", "c", "d"]);
}

#[test]
fn scope_blocks_no_breakpoint_returns_all() {
    let blocks = json!([
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]);
    let out = scope_blocks(&blocks);
    let texts: Vec<&str> = out.iter().map(|b| b["text"].as_str().unwrap()).collect();
    assert_eq!(texts, vec!["a", "b"]);
}

#[test]
fn scope_blocks_empty_returns_empty() {
    assert!(scope_blocks(&json!([])).is_empty());
    assert!(scope_blocks(&Value::Null).is_empty());
}

// ---- compute_cache_key -------------------------------------------------

#[test]
fn compute_key_returns_prefixed_hex() {
    let key = compute_cache_key("claude-opus-4-7", System::Text("You are helpful."), None);
    assert!(key.starts_with(&format!("{KEY_PREFIX}:claude-opus-4-7:sha256:")));
    let hex_part = key.rsplit(':').next().unwrap();
    assert_eq!(hex_part.len(), 64);
    assert!(hex_part.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn compute_key_stable() {
    let a = compute_cache_key("claude-opus-4-7", System::Text("x"), None);
    let b = compute_cache_key("claude-opus-4-7", System::Text("x"), None);
    assert_eq!(a, b);
}

#[test]
fn compute_key_changes_with_model() {
    let a = compute_cache_key("claude-opus-4-7", System::Text("x"), None);
    let b = compute_cache_key("claude-sonnet-4-6", System::Text("x"), None);
    assert_ne!(a, b);
}

#[test]
fn compute_key_changes_with_system() {
    let a = compute_cache_key("claude-opus-4-7", System::Text("x"), None);
    let b = compute_cache_key("claude-opus-4-7", System::Text("y"), None);
    assert_ne!(a, b);
}

#[test]
fn compute_key_changes_with_tools() {
    let tools = json!([{"name": "search", "description": "d", "input_schema": {}}]);
    let a = compute_cache_key("claude-opus-4-7", System::Text("x"), Some(&tools));
    let b = compute_cache_key("claude-opus-4-7", System::Text("x"), Some(&json!([])));
    assert_ne!(a, b);
}

#[test]
fn compute_key_string_system_equals_single_text_block() {
    let a = compute_cache_key("claude-opus-4-7", System::Text("hello"), None);
    let blocks = json!([{"type": "text", "text": "hello"}]);
    let b = compute_cache_key("claude-opus-4-7", System::Blocks(&blocks), None);
    assert_eq!(a, b);
}

#[test]
fn compute_key_ignores_content_after_last_breakpoint() {
    let base = json!([
        {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "EXTRA-1"},
    ]);
    let other = json!([
        {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "DIFFERENT"},
    ]);
    let a = compute_cache_key("claude-opus-4-7", System::Blocks(&base), None);
    let b = compute_cache_key("claude-opus-4-7", System::Blocks(&other), None);
    assert_eq!(a, b);
}

#[test]
fn compute_key_includes_content_before_and_at_breakpoint() {
    let a_blocks = json!([
        {"type": "text", "text": "stable-a", "cache_control": {"type": "ephemeral"}},
    ]);
    let b_blocks = json!([
        {"type": "text", "text": "stable-b", "cache_control": {"type": "ephemeral"}},
    ]);
    let a = compute_cache_key("claude-opus-4-7", System::Blocks(&a_blocks), None);
    let b = compute_cache_key("claude-opus-4-7", System::Blocks(&b_blocks), None);
    assert_ne!(a, b);
}

#[test]
fn compute_key_none_system_works() {
    let key = compute_cache_key("claude-opus-4-7", System::None, None);
    assert!(key.starts_with(&format!("{KEY_PREFIX}:claude-opus-4-7:sha256:")));
}

#[test]
fn compute_key_tools_order_matters() {
    let t1 = json!([{"name": "a", "description": "", "input_schema": {}}]);
    let t2 = json!([
        {"name": "a", "description": "", "input_schema": {}},
        {"name": "b", "description": "", "input_schema": {}},
    ]);
    let a = compute_cache_key("claude-opus-4-7", System::None, Some(&t1));
    let b = compute_cache_key("claude-opus-4-7", System::None, Some(&t2));
    assert_ne!(a, b);
}

#[test]
fn compute_key_no_breakpoint_includes_full_system() {
    let a = compute_cache_key("claude-opus-4-7", System::Text("full prompt A"), None);
    let b = compute_cache_key("claude-opus-4-7", System::Text("full prompt B"), None);
    assert_ne!(a, b);
}

#[test]
fn key_prefix_constant() {
    assert_eq!(KEY_PREFIX, "anthropic-cache");
}

// ---- canonical_json ----------------------------------------------------

#[test]
fn canonical_json_sorts_keys_recursively() {
    let a = canonical_json(&json!({"b": 1, "a": 2}));
    let b = canonical_json(&json!({"a": 2, "b": 1}));
    assert_eq!(a, b);
    assert_eq!(a, r#"{"a":2,"b":1}"#);
}

#[test]
fn canonical_json_nested_objects_sorted() {
    let v = json!({"outer": {"z": 1, "a": 2}, "first": [{"y": 1, "x": 2}]});
    let out = canonical_json(&v);
    assert_eq!(out, r#"{"first":[{"x":2,"y":1}],"outer":{"a":2,"z":1}}"#);
}

#[test]
fn canonical_json_unicode_passes_through() {
    let v = json!({"hello": "héllo"});
    let out = canonical_json(&v);
    assert!(out.contains("héllo"));
}

#[test]
fn canonical_json_escapes_control_chars() {
    let v = json!({"k": "a\nb\tc"});
    let out = canonical_json(&v);
    assert_eq!(out, r#"{"k":"a\nb\tc"}"#);
}

#[test]
fn compute_key_order_independent_in_objects() {
    // semantically identical tool definitions hash to the same key
    let t1 = json!([{"name": "a", "description": "x", "input_schema": {}}]);
    let t2 = json!([{"description": "x", "input_schema": {}, "name": "a"}]);
    let a = compute_cache_key("claude-opus-4-7", System::None, Some(&t1));
    let b = compute_cache_key("claude-opus-4-7", System::None, Some(&t2));
    assert_eq!(a, b);
}

#[test]
fn compute_key_known_digest() {
    // Lock down a known-good digest so future refactors do not silently
    // shift the canonicalization scheme.
    let key = compute_cache_key("claude-opus-4-7", System::Text("hi"), None);
    assert_eq!(
        key,
        "anthropic-cache:claude-opus-4-7:sha256:05061cc26d0f9b68927bd818199f650bba1c36710ca18adcc4faa8a4d6dc1646"
    );
}

// ---- invariants around the tools and system arguments ------------------

#[test]
fn compute_key_none_tools_equals_empty_array_tools() {
    // `tools: None` defaults to an empty array, so it must hash identically
    // to an explicitly-empty tools list.
    let a = compute_cache_key("claude-opus-4-7", System::Text("x"), None);
    let b = compute_cache_key("claude-opus-4-7", System::Text("x"), Some(&json!([])));
    assert_eq!(a, b);
}

#[test]
fn compute_key_blocks_non_array_behaves_like_none() {
    // A `System::Blocks` carrying a non-array value contributes no scoped
    // blocks, so it must match an empty `System::None`.
    let scalar = json!("not an array");
    let a = compute_cache_key("claude-opus-4-7", System::Blocks(&scalar), None);
    let b = compute_cache_key("claude-opus-4-7", System::None, None);
    assert_eq!(a, b);
}

#[test]
fn compute_key_blocks_null_behaves_like_none() {
    let null = Value::Null;
    let a = compute_cache_key("claude-opus-4-7", System::Blocks(&null), None);
    let b = compute_cache_key("claude-opus-4-7", System::None, None);
    assert_eq!(a, b);
}

#[test]
fn scope_blocks_non_array_inputs_are_empty() {
    assert!(scope_blocks(&json!({"a": 1})).is_empty());
    assert!(scope_blocks(&json!("string")).is_empty());
    assert!(scope_blocks(&json!(42)).is_empty());
    assert!(scope_blocks(&json!(true)).is_empty());
}

#[test]
fn find_breakpoints_non_array_inputs_are_empty() {
    assert!(find_breakpoints(&json!({"cache_control": {"type": "ephemeral"}})).is_empty());
    assert!(find_breakpoints(&json!("string")).is_empty());
    assert!(find_breakpoints(&json!(0)).is_empty());
}

#[test]
fn compute_key_tool_field_order_does_not_matter() {
    // Reordering the keys inside a single tool object must not change the key,
    // because the whole payload is canonicalized before hashing.
    let t1 = json!([{
        "name": "search",
        "description": "look things up",
        "input_schema": {"type": "object", "properties": {}}
    }]);
    let t2 = json!([{
        "input_schema": {"properties": {}, "type": "object"},
        "description": "look things up",
        "name": "search"
    }]);
    let a = compute_cache_key("claude-opus-4-7", System::None, Some(&t1));
    let b = compute_cache_key("claude-opus-4-7", System::None, Some(&t2));
    assert_eq!(a, b);
}

#[test]
fn compute_key_extra_cache_control_marker_does_not_affect_scoped_prefix() {
    // Adding a trailing block beyond the last breakpoint must not change the
    // key, even when that trailing block itself differs structurally.
    let a_blocks = json!([
        {"type": "text", "text": "shared", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "tail-a"},
    ]);
    let b_blocks = json!([
        {"type": "text", "text": "shared", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "tail-b", "extra_field": 123},
    ]);
    let a = compute_cache_key("claude-opus-4-7", System::Blocks(&a_blocks), None);
    let b = compute_cache_key("claude-opus-4-7", System::Blocks(&b_blocks), None);
    assert_eq!(a, b);
}

#[test]
fn canonical_json_top_level_scalar_round_trips() {
    assert_eq!(canonical_json(&json!(42)), "42");
    assert_eq!(canonical_json(&json!("s")), r#""s""#);
    assert_eq!(canonical_json(&Value::Null), "null");
    assert_eq!(canonical_json(&json!([3, 2, 1])), "[3,2,1]");
}

#[test]
fn canonical_json_preserves_array_order() {
    // Arrays are ordered data; canonicalization must not reorder them.
    let a = canonical_json(&json!([3, 1, 2]));
    let b = canonical_json(&json!([1, 2, 3]));
    assert_ne!(a, b);
    assert_eq!(a, "[3,1,2]");
}
