//! `SamplingParams`: temperature must be greater than `1e-10`.

use crate::pyval::PyVal;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    Bool(bool),
    Int(i128),
    Float(f64),
}

impl Scalar {
    fn greater_than_epsilon(self) -> bool {
        let value = match self {
            Self::Bool(true) => 1.0,
            Self::Bool(false) => 0.0,
            Self::Int(n) => n as f64,
            Self::Float(n) => n,
        };
        value > 1e-10
    }

    fn to_py(self) -> PyVal {
        match self {
            Self::Bool(v) => PyVal::Bool(v),
            Self::Int(v) => PyVal::Int(v),
            Self::Float(v) => PyVal::Float(v),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: Scalar,
    pub max_tokens: i128,
    pub ignore_eos: bool,
}

impl SamplingParams {
    pub fn try_new(args: &PyVal) -> Result<Self, String> {
        let temperature = scalar_arg(args, "temperature", Scalar::Float(1.0))?;
        let max_tokens = int_arg(args, "max_tokens", 64)?;
        let ignore_eos = bool_arg(args, "ignore_eos", false)?;
        if !temperature.greater_than_epsilon() {
            return Err("AssertionError: greedy sampling is not permitted".into());
        }
        Ok(Self {
            temperature,
            max_tokens,
            ignore_eos,
        })
    }

    pub fn snapshot(&self) -> PyVal {
        PyVal::Map(vec![
            ("temperature".into(), self.temperature.to_py()),
            ("max_tokens".into(), PyVal::Int(self.max_tokens)),
            ("ignore_eos".into(), PyVal::Bool(self.ignore_eos)),
        ])
    }
}

fn field<'a>(args: &'a PyVal, key: &str) -> Option<&'a PyVal> {
    args.get(key)
}

fn scalar_arg(args: &PyVal, key: &str, default: Scalar) -> Result<Scalar, String> {
    let Some(value) = field(args, key) else {
        return Ok(default);
    };
    match value {
        PyVal::Bool(v) => Ok(Scalar::Bool(*v)),
        PyVal::Int(v) => Ok(Scalar::Int(*v)),
        PyVal::Float(v) => Ok(Scalar::Float(*v)),
        PyVal::Str(_) => {
            Err("TypeError: '>' not supported between instances of 'str' and 'float'".into())
        }
        _ => Err("TypeError: unsupported temperature".into()),
    }
}

fn int_arg(args: &PyVal, key: &str, default: i128) -> Result<i128, String> {
    match field(args, key) {
        None => Ok(default),
        Some(PyVal::Int(v)) => Ok(*v),
        Some(PyVal::Bool(v)) => Ok(i128::from(*v)),
        Some(_) => Err(format!("TypeError: {key} is not an int")),
    }
}

fn bool_arg(args: &PyVal, key: &str, default: bool) -> Result<bool, String> {
    match field(args, key) {
        None => Ok(default),
        Some(PyVal::Bool(v)) => Ok(*v),
        Some(_) => Err(format!("TypeError: {key} is not a bool")),
    }
}
