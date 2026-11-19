//! Values in the recorder's JSONL codec. Floats compare by round-trip value, and `-0.0` stays signed.

use serde_json::Value;

#[derive(Clone, Debug)]
pub enum PyVal {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    List(Vec<PyVal>),
    Map(Vec<(String, PyVal)>),
    Tuple(Vec<PyVal>),
    /// A field read that raised, stored as `Type: message`.
    Raised(String),
}

impl PartialEq for PyVal {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => float_eq(*a, *b),
            (Self::Str(a), Self::Str(b)) => a == b,
            (Self::List(a), Self::List(b)) | (Self::Tuple(a), Self::Tuple(b)) => a == b,
            (Self::Map(a), Self::Map(b)) => {
                let mut left = a.clone();
                let mut right = b.clone();
                left.sort_by(|x, y| x.0.cmp(&y.0));
                right.sort_by(|x, y| x.0.cmp(&y.0));
                left == right
            }
            (Self::Raised(a), Self::Raised(b)) => a == b,
            _ => false,
        }
    }
}

fn float_eq(a: f64, b: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a == 0.0 && b == 0.0 {
        return a.is_sign_negative() == b.is_sign_negative();
    }
    a == b
}

pub fn decode(value: &Value) -> PyVal {
    match value {
        Value::Null => PyVal::Null,
        Value::Array(items) => PyVal::List(items.iter().map(decode).collect()),
        Value::Object(map) if map.len() == 1 => {
            let (key, inner) = map.iter().next().expect("one entry");
            match key.as_str() {
                "bool" => PyVal::Bool(inner.as_bool().unwrap_or(false)),
                "int" => PyVal::Int(inner.as_str().unwrap_or("0").parse().unwrap_or(0)),
                "float" => PyVal::Float(parse_float(inner.as_str().unwrap_or("nan"))),
                "str" => PyVal::Str(inner.as_str().unwrap_or("").to_string()),
                "tuple" => PyVal::Tuple(
                    inner
                        .as_array()
                        .map(|xs| xs.iter().map(decode).collect())
                        .unwrap_or_default(),
                ),
                "error" => PyVal::Raised(inner.as_str().unwrap_or("").to_string()),
                _ => PyVal::Map(map.iter().map(|(k, v)| (k.clone(), decode(v))).collect()),
            }
        }
        Value::Object(map) => PyVal::Map(map.iter().map(|(k, v)| (k.clone(), decode(v))).collect()),
        // Plain JSON left by steps that were not passed through the codec.
        Value::Bool(b) => PyVal::Bool(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                PyVal::Int(i128::from(i))
            } else if let Some(u) = n.as_u64() {
                PyVal::Int(i128::from(u))
            } else {
                PyVal::Float(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        Value::String(s) => PyVal::Str(s.clone()),
    }
}

fn parse_float(text: &str) -> f64 {
    match text {
        "nan" => f64::NAN,
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        "-0.0" => -0.0,
        other => other.parse().unwrap_or(f64::NAN),
    }
}

pub fn int(value: i128) -> PyVal {
    PyVal::Int(value)
}

pub fn list_i64(values: &[i64]) -> PyVal {
    PyVal::List(
        values
            .iter()
            .copied()
            .map(|n| PyVal::Int(i128::from(n)))
            .collect(),
    )
}

impl PyVal {
    pub fn get(&self, key: &str) -> Option<&PyVal> {
        match self {
            Self::Map(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_i128(&self) -> Option<i128> {
        match self {
            Self::Int(n) => Some(*n),
            Self::Bool(true) => Some(1),
            Self::Bool(false) => Some(0),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }
}
