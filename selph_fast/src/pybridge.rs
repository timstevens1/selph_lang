//! PyO3 bridge — exposes the SELPH evaluator as a Python module.
//!
//! Python usage:
//!   import selph_fast
//!   env = selph_fast.Env()
//!   env.load_file("library.selph")
//!   result = env.eval_expr("(add 1 2)")   # returns 3
//!   result = env.eval_expr("(param 1.5)")  # returns Param(1.5, None, None)
//!
//! Value conversion:
//!   SELPH Int    <-> Python int
//!   SELPH Num    <-> Python float
//!   SELPH Str    <-> Python str
//!   SELPH Bool   <-> Python bool
//!   SELPH List   <-> Python list
//!   SELPH Nil    <-> Python None
//!   SELPH Param  <-> Python Param (custom class)
//!   SELPH Ns     <-> Python dict
//!   SELPH Function/Builtin/Node -> Python str (repr only)

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use std::rc::Rc;

use crate::types_v2::{Value, Env as SelphEnv};
use crate::eval_v2;
use crate::parser;
use crate::intern::{intern, resolve};

// ── Helper: parse and convert to v2 nodes ───────────────────────────────────

/// Parse a source string, convert to v2 nodes, return (Rc<[Node]>, root_index).
fn parse_expr(source: &str) -> PyResult<(Rc<[crate::types_v2::Node]>, usize)> {
    let (old_nodes, root) = parser::parse_source(source)
        .map_err(|e| PyRuntimeError::new_err(format!("parse error: {:?}", e)))?;
    let v2_nodes = eval_v2::convert_tree(&old_nodes);
    Ok((v2_nodes.into(), root))
}

/// Parse a file source, convert to v2 nodes, return (Rc<[Node]>, root_indices).
fn parse_multi(source: &str) -> PyResult<(Rc<[crate::types_v2::Node]>, Vec<usize>)> {
    let (old_nodes, roots) = parser::parse_file(source)
        .map_err(|e| PyRuntimeError::new_err(format!("parse error: {:?}", e)))?;
    let v2_nodes = eval_v2::convert_tree(&old_nodes);
    Ok((v2_nodes.into(), roots))
}

// ── Param Python class ──────────────────────────────────────────────────────

/// A SELPH optimization parameter exposed to Python.
#[pyclass]
#[derive(Clone, Debug)]
pub struct Param {
    #[pyo3(get)]
    pub value: f64,
    #[pyo3(get)]
    pub min: Option<f64>,
    #[pyo3(get)]
    pub max: Option<f64>,
}

#[pymethods]
impl Param {
    #[new]
    #[pyo3(signature = (value, min=None, max=None))]
    fn new(value: f64, min: Option<f64>, max: Option<f64>) -> Self {
        Param { value, min, max }
    }

    fn __repr__(&self) -> String {
        match (self.min, self.max) {
            (Some(lo), Some(hi)) => format!("Param({}, min={}, max={})", self.value, lo, hi),
            _ => format!("Param({})", self.value),
        }
    }
}

// ── Value conversion ────────────────────────────────────────────────────────

/// Convert a SELPH Value to a Python object.
fn value_to_python(py: Python<'_>, v: &Value) -> PyResult<PyObject> {
    match v {
        Value::Int(n) => Ok(n.into_pyobject(py)?.into_any().unbind()),
        Value::Num(n) => Ok(n.into_pyobject(py)?.into_any().unbind()),
        Value::Str(s) => Ok(s.as_ref().into_pyobject(py)?.into_any().unbind()),
        Value::Bool(b) => {
            let py_bool = PyBool::new(py, *b);
            Ok(py_bool.to_owned().into_any().unbind())
        }
        Value::Nil => Ok(py.None()),
        Value::Param(val, lo, hi) => {
            let p = Param {
                value: *val,
                min: *lo,
                max: *hi,
            };
            Ok(p.into_pyobject(py)?.into_any().unbind())
        }
        Value::List(items) => {
            let py_items: Vec<PyObject> = items
                .iter()
                .map(|item| value_to_python(py, item))
                .collect::<PyResult<_>>()?;
            Ok(PyList::new(py, &py_items)?.into_any().unbind())
        }
        Value::Ns(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map.iter() {
                let key = resolve(*k);
                let val = value_to_python(py, v)?;
                dict.set_item(key, val)?;
            }
            Ok(dict.into_any().unbind())
        }
        // Opaque types — return their string representation
        Value::Function(_) | Value::Builtin(_) | Value::Node(_) => {
            Ok(eval_v2::value_to_string(v).into_pyobject(py)?.into_any().unbind())
        }
    }
}

/// Convert a Python object to a SELPH Value.
fn python_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    // Check in order: bool before int (bool is subclass of int in Python)
    if obj.is_none() {
        return Ok(Value::Nil);
    }
    // Must check bool before int — Python bool is a subclass of int
    if obj.downcast::<PyBool>().is_ok() {
        let b: bool = obj.extract()?;
        return Ok(Value::Bool(b));
    }
    if obj.downcast::<PyInt>().is_ok() {
        let n: i64 = obj.extract()?;
        return Ok(Value::Int(n));
    }
    if obj.downcast::<PyFloat>().is_ok() {
        let n: f64 = obj.extract()?;
        return Ok(Value::Num(n));
    }
    if obj.downcast::<PyString>().is_ok() {
        let s: String = obj.extract()?;
        return Ok(Value::Str(Rc::from(s.as_str())));
    }
    if let Ok(param) = obj.extract::<Param>() {
        return Ok(Value::Param(param.value, param.min, param.max));
    }
    if let Ok(list) = obj.downcast::<PyList>() {
        let items: Vec<Value> = list
            .iter()
            .map(|item| python_to_value(&item))
            .collect::<PyResult<_>>()?;
        return Ok(Value::list(items));
    }
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = crate::types_v2::NsMap::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            let val = python_to_value(&v)?;
            map.insert(intern(&key), val);
        }
        return Ok(Value::Ns(Rc::new(map)));
    }
    Err(PyRuntimeError::new_err(format!(
        "cannot convert Python {} to SELPH Value",
        obj.get_type().name()?
    )))
}

// ── Env Python class ────────────────────────────────────────────────────────

/// A SELPH evaluation environment. Create one, optionally load files or
/// define bindings, then evaluate expressions.
///
/// unsendable because the SELPH Env uses Rc (not Arc) internally.
/// This means Env must be used from the thread that created it.
#[pyclass(unsendable)]
pub struct Env {
    inner: SelphEnv,
}

#[pymethods]
impl Env {
    /// Create a new environment with all default builtins.
    #[new]
    fn new() -> Self {
        Env {
            inner: eval_v2::make_default_env(),
        }
    }

    /// Load and evaluate a SELPH file, adding its definitions to the env.
    fn load_file(&mut self, path: &str) -> PyResult<()> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| PyRuntimeError::new_err(format!("cannot read {}: {}", path, e)))?;
        let (nodes_rc, roots) = parse_multi(&source)?;
        for root in roots {
            eval_v2::eval(&nodes_rc, root, &self.inner)
                .map_err(|e| PyRuntimeError::new_err(format!("eval error: {}", e)))?;
        }
        Ok(())
    }

    /// Evaluate a SELPH expression string and return the result as a Python object.
    fn eval_expr(&self, py: Python<'_>, source: &str) -> PyResult<PyObject> {
        let (nodes_rc, root) = parse_expr(source)?;
        let result = eval_v2::eval(&nodes_rc, root, &self.inner)
            .map_err(|e| PyRuntimeError::new_err(format!("eval error: {}", e)))?;
        value_to_python(py, &result)
    }

    /// Define a binding in the environment.
    fn define(&mut self, py: Python<'_>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = py;
        let val = python_to_value(value)?;
        let sym = intern(name);
        self.inner.define(sym, val);
        Ok(())
    }

    /// Look up a binding by name, returning None if not found.
    fn lookup(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        let sym = intern(name);
        match self.inner.lookup(sym) {
            Some(val) => value_to_python(py, &val),
            None => Ok(py.None()),
        }
    }

    /// Evaluate and return as a string (same as the CLI output).
    fn eval_to_string(&self, source: &str) -> PyResult<String> {
        let (nodes_rc, root) = parse_expr(source)?;
        let result = eval_v2::eval(&nodes_rc, root, &self.inner)
            .map_err(|e| PyRuntimeError::new_err(format!("eval error: {}", e)))?;
        Ok(eval_v2::value_to_string(&result))
    }
}

// ── Module definition ───────────────────────────────────────────────────────

/// SELPH — Symbolic Evaluation Language for Programmable Hierarchies.
/// Python bindings for the SELPH evaluator, parser, and type system.
#[pymodule]
fn selph_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Env>()?;
    m.add_class::<Param>()?;
    Ok(())
}
