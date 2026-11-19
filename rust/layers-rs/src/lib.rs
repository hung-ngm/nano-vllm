//! Python bindings for `nanovllm/layers/`: generated from the signatures in `port.rs` by rust-anything.
//! Regenerate with `rust-anything refresh` after changing them.

use pyo3::prelude::*;

pub mod js;
pub mod port;
pub mod pyconv;
pub mod pynum;

#[allow(unused_imports)]
use pynum::Num;

fn error(message: String) -> PyErr {
    use pyo3::exceptions as e;
    let (kind, text) = message.split_once(": ").unwrap_or(("", message.as_str()));
    let text = text.to_string();
    match kind {
        "TypeError" => e::PyTypeError::new_err(text),
        "KeyError" => e::PyKeyError::new_err(text),
        "IndexError" => e::PyIndexError::new_err(text),
        "OverflowError" => e::PyOverflowError::new_err(text),
        "ZeroDivisionError" => e::PyZeroDivisionError::new_err(text),
        "AttributeError" => e::PyAttributeError::new_err(text),
        "RuntimeError" => e::PyRuntimeError::new_err(text),
        "NotImplementedError" => e::PyNotImplementedError::new_err(text),
        "LookupError" => e::PyLookupError::new_err(text),
        "ArithmeticError" => e::PyArithmeticError::new_err(text),
        "UnicodeError" => e::PyUnicodeError::new_err(text),
        "ValueError" => e::PyValueError::new_err(text),
        // `divide` raises AssertionError. The generated match does not include it, so an
        // `Err("AssertionError: ...")` would be re-raised as ValueError and fail `diff`.
        "AssertionError" => e::PyAssertionError::new_err(text),
        _ => e::PyValueError::new_err(message),
    }
}

fn divide_inner(numerator: Bound<'_, PyAny>, denominator: Bound<'_, PyAny>) -> Result<Num, String> {
    let numerator: Num = crate::pyconv::FromPy::from_py(&numerator)?;
    let denominator: Num = crate::pyconv::FromPy::from_py(&denominator)?;
    let out = port::divide(numerator, denominator)?;
    Ok(out)
}

/// `divide`.
#[pyfunction]
#[pyo3(name = "divide", signature = (numerator, denominator))]
fn divide(
    py: Python<'_>,
    numerator: Bound<'_, PyAny>,
    denominator: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    divide_inner(numerator, denominator)
        .map_err(error)
        .and_then(|v| crate::pyconv::ToPy::to_py(&v, py))
}

#[pymodule]
#[pyo3(name = "_native")]
fn native_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(divide, m)?)?;
    Ok(())
}
