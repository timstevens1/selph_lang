//! S-expression parser for SELPH.
//!
//! Tokenizes source text, then parses into the Node AST.

use crate::types::Node;

#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub col: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "line {}, col {}: {}", self.line, self.col, self.message)
    }
}

#[derive(Debug, Clone)]
struct Token {
    kind: TokenKind,
    value: String,
    line: usize,
    col: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum TokenKind {
    LParen,
    RParen,
    Number,
    String,
    Symbol,
    Keyword,   // :keyword
}

// ── Tokenizer ───────────────────────────────────────────────────────

pub fn tokenize(source: &str) -> Result<Vec<Token>, ParseError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = source.chars().collect();
    let mut i = 0;
    let mut line = 1usize;
    let mut col = 1usize;

    while i < chars.len() {
        let c = chars[i];

        // Whitespace
        if c == ' ' || c == '\t' || c == '\r' {
            i += 1;
            col += 1;
            continue;
        }
        if c == '\n' {
            i += 1;
            line += 1;
            col = 1;
            continue;
        }

        // Comments
        if c == ';' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }

        // Parens
        if c == '(' {
            tokens.push(Token { kind: TokenKind::LParen, value: "(".into(), line, col });
            i += 1; col += 1;
            continue;
        }
        if c == ')' {
            tokens.push(Token { kind: TokenKind::RParen, value: ")".into(), line, col });
            i += 1; col += 1;
            continue;
        }

        // String literal
        if c == '"' {
            let start_col = col;
            i += 1; col += 1;
            let mut buf = String::new();
            while i < chars.len() && chars[i] != '"' {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    i += 1; col += 1;
                    match chars[i] {
                        'n' => buf.push('\n'),
                        't' => buf.push('\t'),
                        '\\' => buf.push('\\'),
                        '"' => buf.push('"'),
                        other => buf.push(other),
                    }
                } else {
                    if chars[i] == '\n' { line += 1; col = 0; }
                    buf.push(chars[i]);
                }
                i += 1; col += 1;
            }
            if i >= chars.len() {
                return Err(ParseError { message: "unterminated string".into(), line, col: start_col });
            }
            i += 1; col += 1; // closing "
            tokens.push(Token { kind: TokenKind::String, value: buf, line, col: start_col });
            continue;
        }

        // Keywords: :name
        if c == ':' {
            let start_col = col;
            i += 1; col += 1;
            let mut buf = String::from(":");
            while i < chars.len() && !is_delimiter(chars[i]) {
                buf.push(chars[i]);
                i += 1; col += 1;
            }
            tokens.push(Token { kind: TokenKind::Keyword, value: buf, line, col: start_col });
            continue;
        }

        // Numbers (including negative)
        if c.is_ascii_digit() || (c == '-' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit()) {
            let start_col = col;
            let mut buf = String::new();
            if c == '-' {
                buf.push('-');
                i += 1; col += 1;
            }
            while i < chars.len() && chars[i].is_ascii_digit() {
                buf.push(chars[i]);
                i += 1; col += 1;
            }
            if i < chars.len() && chars[i] == '.' {
                buf.push('.');
                i += 1; col += 1;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    buf.push(chars[i]);
                    i += 1; col += 1;
                }
            }
            tokens.push(Token { kind: TokenKind::Number, value: buf, line, col: start_col });
            continue;
        }

        // Symbols (identifiers + operators)
        if is_symbol_start(c) {
            let start_col = col;
            let mut buf = String::new();
            while i < chars.len() && is_symbol_cont(chars[i]) {
                buf.push(chars[i]);
                i += 1; col += 1;
            }
            tokens.push(Token { kind: TokenKind::Symbol, value: buf, line, col: start_col });
            continue;
        }

        return Err(ParseError {
            message: format!("unexpected character: {:?}", c),
            line, col,
        });
    }

    Ok(tokens)
}

fn is_delimiter(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\n' | '(' | ')' | '"' | ',')
}

fn is_symbol_start(c: char) -> bool {
    c.is_ascii_alphabetic() || matches!(c, '_' | '?' | '=' | '<' | '>' | '!' | '+' | '*' | '/' | '&' | '|' | '~' | '^' | '%' | '-')
}

fn is_symbol_cont(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '?' | '=' | '<' | '>' | '!' | '+' | '*' | '/' | '&' | '|' | '~' | '^' | '%')
}

// ── Parser ──────────────────────────────────────────────────────────

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos];
        self.pos += 1;
        tok
    }

    fn expect(&mut self, kind: TokenKind) -> Result<&Token, ParseError> {
        match self.peek() {
            None => Err(ParseError { message: format!("expected {:?}, got EOF", kind), line: 0, col: 0 }),
            Some(tok) if tok.kind != kind => Err(ParseError {
                message: format!("expected {:?}, got {:?}", kind, tok.kind),
                line: tok.line, col: tok.col,
            }),
            _ => {
                self.pos += 1;
                Ok(&self.tokens[self.pos - 1])
            }
        }
    }

    /// Parse a single expression, returning its index in the node pool.
    pub fn parse_expr(&mut self, nodes: &mut Vec<Node>) -> Result<usize, ParseError> {
        let tok = self.peek().ok_or(ParseError {
            message: "unexpected EOF".into(), line: 0, col: 0,
        })?.clone();

        match tok.kind {
            TokenKind::Number => {
                self.advance();
                let val: f64 = tok.value.parse().map_err(|_| ParseError {
                    message: format!("invalid number: {}", tok.value),
                    line: tok.line, col: tok.col,
                })?;
                let idx = nodes.len();
                nodes.push(Node::Num(val));
                Ok(idx)
            }
            TokenKind::String => {
                self.advance();
                let idx = nodes.len();
                nodes.push(Node::Str(tok.value.clone()));
                Ok(idx)
            }
            TokenKind::Symbol => {
                self.advance();
                let idx = nodes.len();
                if tok.value == "true" {
                    nodes.push(Node::Bool(true));
                } else if tok.value == "false" {
                    nodes.push(Node::Bool(false));
                } else {
                    nodes.push(Node::Symbol(tok.value.clone()));
                }
                Ok(idx)
            }
            TokenKind::Keyword => {
                // Keywords as symbols
                self.advance();
                let idx = nodes.len();
                nodes.push(Node::Symbol(tok.value.clone()));
                Ok(idx)
            }
            TokenKind::LParen => {
                self.advance(); // consume (

                // Check for special forms
                if let Some(head) = self.peek() {
                    let head_val = head.value.clone();
                    let head_kind = head.kind.clone();

                    if head_kind == TokenKind::Symbol {
                        match head_val.as_str() {
                            "if" => return self.parse_if(nodes),
                            "lambda" => return self.parse_lambda(nodes),
                            "let" => return self.parse_let(nodes),
                            _ => {}
                        }
                    }
                }

                // Generic list/application
                let mut children = Vec::new();
                while let Some(tok) = self.peek() {
                    if tok.kind == TokenKind::RParen {
                        break;
                    }
                    let child = self.parse_expr(nodes)?;
                    children.push(child);
                }
                self.expect(TokenKind::RParen)?;
                let idx = nodes.len();
                nodes.push(Node::App(children));
                Ok(idx)
            }
            TokenKind::RParen => {
                Err(ParseError {
                    message: "unexpected )".into(),
                    line: tok.line, col: tok.col,
                })
            }
        }
    }

    fn parse_if(&mut self, nodes: &mut Vec<Node>) -> Result<usize, ParseError> {
        self.advance(); // consume "if"
        let cond = self.parse_expr(nodes)?;
        let then_br = self.parse_expr(nodes)?;
        let else_br = self.parse_expr(nodes)?;
        self.expect(TokenKind::RParen)?;
        let idx = nodes.len();
        nodes.push(Node::If(cond, then_br, else_br));
        Ok(idx)
    }

    fn parse_lambda(&mut self, nodes: &mut Vec<Node>) -> Result<usize, ParseError> {
        self.advance(); // consume "lambda"
        // Parse params: (x y z)
        self.expect(TokenKind::LParen)?;
        let mut params = Vec::new();
        while let Some(tok) = self.peek() {
            if tok.kind == TokenKind::RParen { break; }
            let tok = self.advance().clone();
            params.push(tok.value);
        }
        self.expect(TokenKind::RParen)?;
        // Parse body
        let body = self.parse_expr(nodes)?;
        self.expect(TokenKind::RParen)?;
        let idx = nodes.len();
        nodes.push(Node::Lambda(params, body));
        Ok(idx)
    }

    fn parse_let(&mut self, nodes: &mut Vec<Node>) -> Result<usize, ParseError> {
        self.advance(); // consume "let"
        // Parse bindings: ((name val) ...)
        self.expect(TokenKind::LParen)?;
        let mut bindings = Vec::new();
        while let Some(tok) = self.peek() {
            if tok.kind == TokenKind::RParen { break; }
            self.expect(TokenKind::LParen)?;
            let name_tok = self.advance().clone();
            let val = self.parse_expr(nodes)?;
            self.expect(TokenKind::RParen)?;
            bindings.push((name_tok.value, val));
        }
        self.expect(TokenKind::RParen)?;
        // Parse body
        let body = self.parse_expr(nodes)?;
        self.expect(TokenKind::RParen)?;
        let idx = nodes.len();
        nodes.push(Node::Let(bindings, body));
        Ok(idx)
    }

    /// Parse multiple top-level expressions.
    pub fn parse_all(&mut self, nodes: &mut Vec<Node>) -> Result<Vec<usize>, ParseError> {
        let mut roots = Vec::new();
        while self.pos < self.tokens.len() {
            let root = self.parse_expr(nodes)?;
            roots.push(root);
        }
        Ok(roots)
    }
}

// ── Public API ──────────────────────────────────────────────────────

/// Parse a SELPH source string into a node pool.
/// Returns (nodes, root_index).
pub fn parse_source(source: &str) -> Result<(Vec<Node>, usize), ParseError> {
    let tokens = tokenize(source)?;
    if tokens.is_empty() {
        return Err(ParseError { message: "empty input".into(), line: 0, col: 0 });
    }
    let mut parser = Parser::new(tokens);
    let mut nodes = Vec::new();
    let root = parser.parse_expr(&mut nodes)?;
    Ok((nodes, root))
}

/// Parse a SELPH source file with multiple expressions.
/// Returns (nodes, root_indices).
pub fn parse_file(source: &str) -> Result<(Vec<Node>, Vec<usize>), ParseError> {
    let tokens = tokenize(source)?;
    if tokens.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut parser = Parser::new(tokens);
    let mut nodes = Vec::new();
    let roots = parser.parse_all(&mut nodes)?;
    Ok((nodes, roots))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_number() {
        let (nodes, root) = parse_source("42").unwrap();
        assert!(matches!(&nodes[root], Node::Num(n) if *n == 42.0));
    }

    #[test]
    fn test_parse_string() {
        let (nodes, root) = parse_source("\"hello\"").unwrap();
        assert!(matches!(&nodes[root], Node::Str(s) if s == "hello"));
    }

    #[test]
    fn test_parse_symbol() {
        let (nodes, root) = parse_source("add").unwrap();
        assert!(matches!(&nodes[root], Node::Symbol(s) if s == "add"));
    }

    #[test]
    fn test_parse_app() {
        let (nodes, root) = parse_source("(add 1 2)").unwrap();
        assert!(matches!(&nodes[root], Node::App(children) if children.len() == 3));
    }

    #[test]
    fn test_parse_lambda() {
        let (nodes, root) = parse_source("(lambda (x) (add x 1))").unwrap();
        assert!(matches!(&nodes[root], Node::Lambda(params, _) if params.len() == 1));
    }

    #[test]
    fn test_parse_if() {
        let (nodes, root) = parse_source("(if true 1 0)").unwrap();
        assert!(matches!(&nodes[root], Node::If(_, _, _)));
    }

    #[test]
    fn test_parse_let() {
        let (nodes, root) = parse_source("(let ((x 1)) x)").unwrap();
        assert!(matches!(&nodes[root], Node::Let(bindings, _) if bindings.len() == 1));
    }

    #[test]
    fn test_parse_nested() {
        let (nodes, root) = parse_source("(add (mul 2 3) 4)").unwrap();
        assert!(matches!(&nodes[root], Node::App(_)));
    }

    #[test]
    fn test_parse_negative() {
        let (nodes, root) = parse_source("-5").unwrap();
        assert!(matches!(&nodes[root], Node::Num(n) if *n == -5.0));
    }

    #[test]
    fn test_parse_comment() {
        let (nodes, root) = parse_source("; comment\n42").unwrap();
        assert!(matches!(&nodes[root], Node::Num(n) if *n == 42.0));
    }

    #[test]
    fn test_parse_bool() {
        let (nodes, root) = parse_source("true").unwrap();
        assert!(matches!(&nodes[root], Node::Bool(true)));
    }

    #[test]
    fn test_parse_defmacro() {
        let (nodes, root) = parse_source("(defmacro double (x) (add x x))").unwrap();
        assert!(matches!(&nodes[root], Node::App(children) if children.len() == 4));
    }
}
