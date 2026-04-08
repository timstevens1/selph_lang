//! ARC-AGI JSON task loader.
//!
//! Parses ARC-AGI task files (JSON format) into Selph Value::Grid pairs
//! for use in synthesis. Implements a minimal JSON parser to avoid external
//! dependencies.

use crate::types::Value;

/// A parsed ARC task with training and test examples.
#[derive(Debug)]
pub struct ArcTask {
    pub id: String,
    pub train: Vec<(Value, Value)>,  // (input_grid, output_grid)
    pub test: Vec<(Value, Option<Value>)>,  // (input_grid, optional output_grid)
}

/// Parse an ARC-AGI JSON task file into an ArcTask.
pub fn parse_arc_task(id: &str, json: &str) -> Result<ArcTask, String> {
    let val = parse_json(json)?;
    let obj = as_object(&val)?;

    let train_val = obj.iter().find(|(k, _)| k == "train")
        .ok_or("missing 'train' field")?.1.clone();
    let train_arr = as_array(&train_val)?;

    let mut train = Vec::new();
    for pair in train_arr {
        let pair_obj = as_object(&pair)?;
        let input = grid_from_json_array(
            pair_obj.iter().find(|(k, _)| k == "input").ok_or("missing 'input'")?.1.clone()
        )?;
        let output = grid_from_json_array(
            pair_obj.iter().find(|(k, _)| k == "output").ok_or("missing 'output'")?.1.clone()
        )?;
        train.push((input, output));
    }

    let test_val = obj.iter().find(|(k, _)| k == "test")
        .ok_or("missing 'test' field")?.1.clone();
    let test_arr = as_array(&test_val)?;

    let mut test = Vec::new();
    for pair in test_arr {
        let pair_obj = as_object(&pair)?;
        let input = grid_from_json_array(
            pair_obj.iter().find(|(k, _)| k == "input").ok_or("missing test 'input'")?.1.clone()
        )?;
        let output = pair_obj.iter().find(|(k, _)| k == "output")
            .map(|(_, v)| grid_from_json_array(v.clone()))
            .transpose()?;
        test.push((input, output));
    }

    Ok(ArcTask { id: id.to_string(), train, test })
}

/// Convert an ArcTask's training examples to Selph (inputs, expected) vectors.
pub fn arc_task_to_spec(task: &ArcTask) -> (Vec<Value>, Vec<Value>) {
    let inputs: Vec<Value> = task.train.iter().map(|(i, _)| i.clone()).collect();
    let expected: Vec<Value> = task.train.iter().map(|(_, o)| o.clone()).collect();
    (inputs, expected)
}

/// Generate a .selph curriculum file from a directory of ARC JSON tasks.
pub fn arc_dir_to_curriculum(tasks: &[ArcTask], depth: usize) -> String {
    let mut out = String::new();
    out.push_str(";; ARC-AGI curriculum (auto-generated)\n\n");
    for task in tasks {
        out.push_str(&format!(";; Task: {}\n", task.id));
        out.push_str(&format!("(task \"{}\" {}\n", task.id, depth));
        for (input, output) in &task.train {
            out.push_str(&format!("  ({} {})\n", grid_to_selph(input), grid_to_selph(output)));
        }
        out.push_str(")\n\n");
    }
    out
}

fn grid_to_selph(v: &Value) -> String {
    match v {
        Value::Grid(rows) => {
            let row_strs: Vec<String> = rows.iter().map(|row| {
                let cells: Vec<String> = row.iter().map(|c| c.to_string()).collect();
                format!("({})", cells.join(" "))
            }).collect();
            format!("(#grid ({}))", row_strs.join(" "))
        }
        _ => format!("{:?}", v),
    }
}

// ── Minimal JSON parser ────────────────────────────────────────────

#[derive(Clone, Debug)]
enum Json {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

fn as_object(v: &Json) -> Result<Vec<(String, Json)>, String> {
    match v { Json::Object(o) => Ok(o.clone()), _ => Err("expected JSON object".into()) }
}

fn as_array(v: &Json) -> Result<Vec<Json>, String> {
    match v { Json::Array(a) => Ok(a.clone()), _ => Err("expected JSON array".into()) }
}

fn grid_from_json_array(v: Json) -> Result<Value, String> {
    let rows = as_array(&v)?;
    let mut grid = Vec::with_capacity(rows.len());
    for row in rows {
        let cells = as_array(&row)?;
        let mut grid_row = Vec::with_capacity(cells.len());
        for cell in cells {
            match cell {
                Json::Number(n) => grid_row.push(n as i8),
                _ => return Err("grid cell must be a number".into()),
            }
        }
        grid.push(grid_row);
    }
    Ok(Value::Grid(grid))
}

fn parse_json(s: &str) -> Result<Json, String> {
    let chars: Vec<char> = s.chars().collect();
    let (val, _) = parse_value(&chars, 0)?;
    Ok(val)
}

fn skip_ws(chars: &[char], mut pos: usize) -> usize {
    while pos < chars.len() && chars[pos].is_whitespace() { pos += 1; }
    pos
}

fn parse_value(chars: &[char], pos: usize) -> Result<(Json, usize), String> {
    let pos = skip_ws(chars, pos);
    if pos >= chars.len() { return Err("unexpected end of JSON".into()); }
    match chars[pos] {
        '{' => parse_object(chars, pos),
        '[' => parse_array(chars, pos),
        '"' => parse_string(chars, pos),
        't' | 'f' => parse_bool(chars, pos),
        'n' => parse_null(chars, pos),
        _ => parse_number(chars, pos),
    }
}

fn parse_object(chars: &[char], mut pos: usize) -> Result<(Json, usize), String> {
    pos += 1; // skip '{'
    pos = skip_ws(chars, pos);
    let mut entries = Vec::new();
    if pos < chars.len() && chars[pos] == '}' { return Ok((Json::Object(entries), pos + 1)); }
    loop {
        pos = skip_ws(chars, pos);
        let (key_json, new_pos) = parse_string(chars, pos)?;
        let key = match key_json { Json::Str(s) => s, _ => unreachable!() };
        pos = skip_ws(chars, new_pos);
        if pos >= chars.len() || chars[pos] != ':' { return Err("expected ':' in object".into()); }
        pos += 1;
        let (val, new_pos) = parse_value(chars, pos)?;
        entries.push((key, val));
        pos = skip_ws(chars, new_pos);
        if pos >= chars.len() { return Err("unexpected end in object".into()); }
        if chars[pos] == '}' { return Ok((Json::Object(entries), pos + 1)); }
        if chars[pos] == ',' { pos += 1; } else { return Err("expected ',' or '}' in object".into()); }
    }
}

fn parse_array(chars: &[char], mut pos: usize) -> Result<(Json, usize), String> {
    pos += 1; // skip '['
    pos = skip_ws(chars, pos);
    let mut items = Vec::new();
    if pos < chars.len() && chars[pos] == ']' { return Ok((Json::Array(items), pos + 1)); }
    loop {
        let (val, new_pos) = parse_value(chars, pos)?;
        items.push(val);
        pos = skip_ws(chars, new_pos);
        if pos >= chars.len() { return Err("unexpected end in array".into()); }
        if chars[pos] == ']' { return Ok((Json::Array(items), pos + 1)); }
        if chars[pos] == ',' { pos += 1; } else { return Err("expected ',' or ']' in array".into()); }
    }
}

fn parse_string(chars: &[char], mut pos: usize) -> Result<(Json, usize), String> {
    pos = skip_ws(chars, pos);
    if pos >= chars.len() || chars[pos] != '"' { return Err("expected '\"'".into()); }
    pos += 1;
    let mut s = String::new();
    while pos < chars.len() && chars[pos] != '"' {
        if chars[pos] == '\\' {
            pos += 1;
            if pos >= chars.len() { return Err("unexpected end in string escape".into()); }
            match chars[pos] {
                'n' => s.push('\n'), 't' => s.push('\t'), '\\' => s.push('\\'),
                '"' => s.push('"'), '/' => s.push('/'),
                _ => { s.push('\\'); s.push(chars[pos]); }
            }
        } else {
            s.push(chars[pos]);
        }
        pos += 1;
    }
    if pos >= chars.len() { return Err("unterminated string".into()); }
    Ok((Json::Str(s), pos + 1))
}

fn parse_number(chars: &[char], mut pos: usize) -> Result<(Json, usize), String> {
    let start = pos;
    if pos < chars.len() && chars[pos] == '-' { pos += 1; }
    while pos < chars.len() && (chars[pos].is_ascii_digit() || chars[pos] == '.' || chars[pos] == 'e' || chars[pos] == 'E' || chars[pos] == '+' || chars[pos] == '-') {
        if pos > start && (chars[pos] == '-' || chars[pos] == '+') && chars[pos - 1] != 'e' && chars[pos - 1] != 'E' { break; }
        pos += 1;
    }
    let num_str: String = chars[start..pos].iter().collect();
    let n: f64 = num_str.parse().map_err(|_| format!("invalid number: {}", num_str))?;
    Ok((Json::Number(n), pos))
}

fn parse_bool(chars: &[char], pos: usize) -> Result<(Json, usize), String> {
    if chars[pos..].starts_with(&['t', 'r', 'u', 'e']) { Ok((Json::Bool(true), pos + 4)) }
    else if chars[pos..].starts_with(&['f', 'a', 'l', 's', 'e']) { Ok((Json::Bool(false), pos + 5)) }
    else { Err("expected 'true' or 'false'".into()) }
}

fn parse_null(chars: &[char], pos: usize) -> Result<(Json, usize), String> {
    if chars[pos..].starts_with(&['n', 'u', 'l', 'l']) { Ok((Json::Null, pos + 4)) }
    else { Err("expected 'null'".into()) }
}
