// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Core / Value Types
//!
//! OBJECTIVE: Explore the Value enum, its variants, accessors, From impls, and ValueMap.
//! LEVEL: Basic
//! PARITY: Full -- DIV-001: Rust distinguishes Integer(i64) and Float(f64);
//!         Python has only float for numeric types.
//!         DIV-003: Rust uses BTreeMap (deterministic sorted order);
//!         Python dict preserves insertion order.
//! VALIDATES: Value enum, ValueMap, value_map! macro, From impls, accessor methods
//!
//! Run: cargo run -p tutorial-core --bin 07_value_types

use std::{collections::BTreeMap, sync::Arc};

use bytes::Bytes;
use kailash_value::{value_map, Value, ValueMap};

fn main() {
    // ── The Value Enum ──
    // Value is the universal data type for all workflow data. It mirrors
    // JSON semantics but adds binary data and distinguishes integers from
    // floats.
    //
    //   Null              -- JSON null / Python None
    //   Bool(bool)        -- true/false
    //   Integer(i64)      -- signed 64-bit integer (DIV-001)
    //   Float(f64)        -- 64-bit IEEE 754 float (DIV-001)
    //   String(Arc<str>)  -- UTF-8 string, reference-counted for cheap cloning
    //   Bytes(Bytes)      -- raw binary data
    //   Array(Vec<Value>) -- ordered sequence
    //   Object(BTreeMap<Arc<str>, Value>) -- sorted string-keyed map (DIV-003)

    // ── Constructing Values ──

    let _null = Value::Null;
    let _boolean = Value::Bool(true);
    let _integer = Value::Integer(42);
    let _float = Value::Float(3.14);
    let _string = Value::String(Arc::from("hello"));
    let _binary = Value::Bytes(Bytes::from_static(b"raw data"));
    let _array = Value::Array(vec![Value::from(1), Value::from(2), Value::from(3)]);
    let _object = Value::Object({
        let mut m = BTreeMap::new();
        m.insert(Arc::from("key"), Value::from("value"));
        m
    });

    // ── From Impls ──
    // Convenient conversion from Rust primitives.

    assert_eq!(Value::from(true), Value::Bool(true));
    assert_eq!(Value::from(false), Value::Bool(false));

    // Integer conversions: i32, i64, u32, u64 (u64 > i64::MAX becomes Float).
    assert_eq!(Value::from(42_i32), Value::Integer(42));
    assert_eq!(Value::from(42_i64), Value::Integer(42));
    assert_eq!(Value::from(42_u32), Value::Integer(42));
    assert_eq!(Value::from(42_u64), Value::Integer(42));

    // Float conversions: f32, f64.
    assert_eq!(Value::from(3.14_f32), Value::Float(3.14_f32 as f64));
    assert_eq!(Value::from(3.14_f64), Value::Float(3.14));

    // String conversions: &str, String, Arc<str>.
    assert_eq!(Value::from("hello"), Value::String(Arc::from("hello")));
    assert_eq!(
        Value::from(String::from("world")),
        Value::String(Arc::from("world"))
    );

    // Vec<Value> -> Array.
    let arr = Value::from(vec![Value::from(1), Value::from(2)]);
    assert!(arr.is_array());

    // BTreeMap -> Object.
    let obj = Value::from({
        let mut m = BTreeMap::<Arc<str>, Value>::new();
        m.insert(Arc::from("x"), Value::from(10));
        m
    });
    assert!(obj.is_object());

    // Bytes conversions: Bytes, Vec<u8>.
    let b1 = Value::from(Bytes::from_static(b"abc"));
    assert!(matches!(b1, Value::Bytes(_)));
    let b2 = Value::from(vec![1_u8, 2, 3]);
    assert!(matches!(b2, Value::Bytes(_)));

    // Option<T> -> Value: Some(x) -> x.into(), None -> Null.
    assert_eq!(Value::from(Some(42_i64)), Value::Integer(42));
    assert_eq!(Value::from(None::<i64>), Value::Null);

    // () -> Null.
    assert_eq!(Value::from(()), Value::Null);

    // ── Accessor Methods ──
    // Each variant has a corresponding as_* method returning Option.

    assert_eq!(Value::Bool(true).as_bool(), Some(true));
    assert_eq!(Value::Integer(42).as_bool(), None); // type mismatch

    assert_eq!(Value::Integer(42).as_i64(), Some(42));
    assert_eq!(Value::Float(3.14).as_i64(), None); // no implicit conversion

    assert_eq!(Value::Float(2.718).as_f64(), Some(2.718));
    assert_eq!(Value::Integer(1).as_f64(), None); // strict: Integer != Float

    assert_eq!(Value::from("hello").as_str(), Some("hello"));
    assert_eq!(Value::Integer(0).as_str(), None);

    let bytes_val = Value::Bytes(Bytes::from_static(b"data"));
    assert!(bytes_val.as_bytes().is_some());
    assert_eq!(bytes_val.as_bytes().map(|b| b.len()), Some(4));

    let arr_val = Value::Array(vec![Value::from(1), Value::from(2)]);
    assert_eq!(arr_val.as_array().map(|a| a.len()), Some(2));

    let obj_val = Value::Object(BTreeMap::new());
    assert!(obj_val.as_object().is_some());
    assert!(obj_val.as_object().map(|o| o.is_empty()) == Some(true));

    // ── Type Checking Methods ──

    assert!(Value::Null.is_null());
    assert!(!Value::Bool(false).is_null());

    assert!(Value::Bool(true).is_bool());
    assert!(!Value::Integer(1).is_bool());

    assert!(Value::Integer(1).is_number());
    assert!(Value::Float(1.0).is_number());
    assert!(!Value::from("1").is_number());

    assert!(Value::from("x").is_string());
    assert!(Value::Array(vec![]).is_array());
    assert!(Value::Object(BTreeMap::new()).is_object());

    // ── Truthiness (is_truthy) ──
    // Used by ConditionalNode for if/else branching.

    // Falsy values:
    assert!(!Value::Null.is_truthy());
    assert!(!Value::Bool(false).is_truthy());
    assert!(!Value::Integer(0).is_truthy());
    assert!(!Value::Float(0.0).is_truthy());
    assert!(!Value::from("").is_truthy());
    assert!(!Value::Bytes(Bytes::new()).is_truthy());
    assert!(!Value::Array(vec![]).is_truthy());
    assert!(!Value::Object(BTreeMap::new()).is_truthy());

    // Truthy values:
    assert!(Value::Bool(true).is_truthy());
    assert!(Value::Integer(1).is_truthy());
    assert!(Value::Integer(-1).is_truthy());
    assert!(Value::Float(0.1).is_truthy());
    assert!(Value::from("x").is_truthy());
    assert!(Value::Bytes(Bytes::from_static(b"x")).is_truthy());
    assert!(Value::Array(vec![Value::Null]).is_truthy());
    assert!(Value::Object({
        let mut m = BTreeMap::new();
        m.insert(Arc::from("k"), Value::Null);
        m
    })
    .is_truthy());

    // ── DIV-001: Integer vs Float ──
    // Rust Value distinguishes Integer(i64) from Float(f64).
    // Python's Value only has a single numeric type (float).
    // This means Rust preserves integer precision for large numbers.

    let big_int = Value::Integer(i64::MAX);
    assert_eq!(big_int.as_i64(), Some(i64::MAX));
    assert_eq!(big_int.as_f64(), None); // Not a Float!

    let precise_float = Value::Float(1.0000000000000002);
    assert_eq!(precise_float.as_f64(), Some(1.0000000000000002));
    assert_eq!(precise_float.as_i64(), None); // Not an Integer!

    // Integer and Float are different variants, even for "whole" numbers.
    let int_one = Value::Integer(1);
    let float_one = Value::Float(1.0);
    assert_ne!(int_one, float_one); // Different variants!

    // ── PartialEq ──
    // Value implements PartialEq but NOT Eq (because of Float/NaN).

    assert_eq!(Value::Integer(42), Value::Integer(42));
    assert_ne!(Value::Integer(42), Value::Integer(43));
    assert_eq!(Value::from("abc"), Value::from("abc"));
    assert_ne!(Value::from("abc"), Value::from("def"));

    // NaN != NaN (IEEE 754 semantics).
    let nan = Value::Float(f64::NAN);
    assert_ne!(nan, nan);

    // ── PartialOrd ──
    // Cross-variant ordering: Null < Bool < Integer < Float < String < Bytes < Array < Object.

    assert!(Value::Null < Value::Bool(false));
    assert!(Value::Bool(true) < Value::Integer(0));
    assert!(Value::Integer(0) < Value::Float(0.0));
    assert!(Value::Float(0.0) < Value::from(""));
    assert!(Value::from("") < Value::Bytes(Bytes::new()));

    // Within the same variant, uses natural ordering.
    assert!(Value::Integer(1) < Value::Integer(2));
    assert!(Value::from("a") < Value::from("b"));

    // ── DIV-003: BTreeMap (Sorted Order) ──
    // ValueMap = BTreeMap<Arc<str>, Value>.
    // Keys are always sorted lexicographically.
    // Python dict preserves insertion order instead.

    let map: ValueMap = {
        let mut m = BTreeMap::new();
        m.insert(Arc::from("zebra"), Value::from(1));
        m.insert(Arc::from("alpha"), Value::from(2));
        m.insert(Arc::from("middle"), Value::from(3));
        m
    };

    let keys: Vec<&str> = map.keys().map(|k| k.as_ref()).collect();
    assert_eq!(keys, vec!["alpha", "middle", "zebra"]); // Sorted!

    // ── value_map! Macro ──
    // Convenient macro for creating ValueMap literals.

    let map = value_map! {
        "name" => "Kailash",
        "version" => 1_i64,
        "active" => true,
    };

    assert_eq!(map.len(), 3);
    assert_eq!(map.get("name" as &str), Some(&Value::from("Kailash")));
    assert_eq!(map.get("version" as &str), Some(&Value::Integer(1)));
    assert_eq!(map.get("active" as &str), Some(&Value::Bool(true)));

    // Empty map.
    let empty: ValueMap = value_map! {};
    assert!(empty.is_empty());

    // ── Display ──
    // Human-readable, JSON-like formatting.

    assert_eq!(format!("{}", Value::Null), "null");
    assert_eq!(format!("{}", Value::Bool(true)), "true");
    assert_eq!(format!("{}", Value::Integer(42)), "42");
    assert_eq!(format!("{}", Value::from("hello")), "\"hello\"");

    // Float display: whole floats get .0 suffix to distinguish from integers.
    assert_eq!(format!("{}", Value::Float(1.0)), "1.0");
    assert_eq!(format!("{}", Value::Float(3.14)), "3.14");

    // Special float values.
    assert_eq!(format!("{}", Value::Float(f64::NAN)), "NaN");
    assert_eq!(format!("{}", Value::Float(f64::INFINITY)), "Infinity");
    assert_eq!(format!("{}", Value::Float(f64::NEG_INFINITY)), "-Infinity");

    // Array display.
    let arr = Value::Array(vec![Value::from(1), Value::from("two")]);
    assert_eq!(format!("{arr}"), "[1, \"two\"]");

    // Bytes display (shows length).
    let b = Value::Bytes(Bytes::from_static(b"hello"));
    assert_eq!(format!("{b}"), "<5 bytes>");

    // ── JSON Interop ──
    // Value converts to/from serde_json::Value.

    let json = serde_json::json!({
        "name": "Kailash",
        "count": 42,
        "pi": 3.14,
        "tags": ["rust", "sdk"],
        "active": true,
        "empty": null
    });

    let val: Value = Value::from(json);
    assert!(val.is_object());
    let obj = val.as_object().expect("is object");
    assert_eq!(obj.get("name" as &str), Some(&Value::from("Kailash")));
    assert_eq!(obj.get("count" as &str), Some(&Value::Integer(42)));
    assert_eq!(obj.get("active" as &str), Some(&Value::Bool(true)));
    assert_eq!(obj.get("empty" as &str), Some(&Value::Null));

    // Round-trip back to serde_json::Value.
    let back: serde_json::Value = serde_json::Value::from(val);
    assert_eq!(back["name"], "Kailash");
    assert_eq!(back["count"], 42);

    // ── Index Support ──
    // Value::Object can be indexed by &str, returning Null for missing keys.

    let obj = Value::Object(value_map! {
        "name" => "Alice",
        "age" => 30_i64,
    });

    assert_eq!(obj["name"].as_str(), Some("Alice"));
    assert_eq!(obj["age"].as_i64(), Some(30));
    assert!(obj["missing"].is_null()); // Missing key -> Null, no panic.

    // ── Merge ──
    // Deep-merge two ValueMaps. Overlay values replace base values;
    // nested Objects are merged recursively.

    let base = value_map! {
        "a" => 1_i64,
        "b" => 2_i64,
        "nested" => Value::Object(value_map! {
            "x" => 10_i64,
            "y" => 20_i64,
        }),
    };

    let overlay = value_map! {
        "b" => 99_i64,
        "c" => 3_i64,
        "nested" => Value::Object(value_map! {
            "y" => 99_i64,
            "z" => 30_i64,
        }),
    };

    let merged = kailash_value::merge(&base, &overlay);
    assert_eq!(merged.get("a" as &str), Some(&Value::Integer(1))); // from base
    assert_eq!(merged.get("b" as &str), Some(&Value::Integer(99))); // overridden
    assert_eq!(merged.get("c" as &str), Some(&Value::Integer(3))); // from overlay

    // Nested merge: x preserved from base, y overridden, z added.
    let nested = merged
        .get("nested" as &str)
        .expect("nested")
        .as_object()
        .expect("nested is object");
    assert_eq!(nested.get("x" as &str), Some(&Value::Integer(10)));
    assert_eq!(nested.get("y" as &str), Some(&Value::Integer(99)));
    assert_eq!(nested.get("z" as &str), Some(&Value::Integer(30)));

    // ── Default ──
    // Value::default() is Value::Null.

    assert_eq!(Value::default(), Value::Null);

    // ── Clone is Cheap ──
    // String keys use Arc<str>, so cloning a Value::String or ValueMap
    // is just an Arc reference count bump -- no heap allocation.

    let original = Value::from("shared string");
    let cloned = original.clone();
    assert_eq!(original, cloned);

    // Object cloning shares the Arc<str> keys.
    let map = value_map! { "key" => "value" };
    let map_clone = map.clone();
    assert_eq!(map, map_clone);

    println!("PASS: 00-core/07_value_types");
}
