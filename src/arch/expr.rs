//! Spec expression language: literals, references into config/metadata scopes, arithmetic,
//! comparison, boolean logic, `?:` and a fixed set of functions (see `call`). Missing references
//! evaluate to `Null`; `first(..)` picks the first non-null argument, any other use of `Null` in
//! arithmetic is an error naming the expression.

use serde_json::Value as J;
use std::fmt;

/// An evaluated value.
#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    List(Vec<Val>),
    Obj(serde_json::Map<String, J>),
}

impl Val {
    pub fn from_json(j: &J) -> Val {
        match j {
            J::Null => Val::Null,
            J::Bool(b) => Val::Bool(*b),
            J::Number(n) => Val::Num(n.as_f64().unwrap_or(f64::NAN)),
            J::String(s) => Val::Str(s.clone()),
            J::Array(a) => Val::List(a.iter().map(Val::from_json).collect()),
            J::Object(o) => Val::Obj(o.clone()),
        }
    }

    /// JSON form; integral numbers become JSON integers so serde can read them as `usize`.
    pub fn to_json(&self) -> J {
        match self {
            Val::Null => J::Null,
            Val::Bool(b) => J::Bool(*b),
            Val::Num(n) => {
                if n.fract() == 0.0 && n.abs() < 9.0e15 {
                    if *n < 0.0 {
                        J::from(*n as i64)
                    } else {
                        J::from(*n as u64)
                    }
                } else {
                    serde_json::Number::from_f64(*n)
                        .map(J::Number)
                        .unwrap_or(J::Null)
                }
            }
            Val::Str(s) => J::String(s.clone()),
            Val::List(l) => J::Array(l.iter().map(Val::to_json).collect()),
            Val::Obj(o) => J::Object(o.clone()),
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Val::Null)
    }

    pub fn truthy(&self) -> bool {
        match self {
            Val::Null => false,
            Val::Bool(b) => *b,
            Val::Num(n) => *n != 0.0,
            Val::Str(s) => !s.is_empty(),
            Val::List(l) => !l.is_empty(),
            Val::Obj(o) => !o.is_empty(),
        }
    }

    pub fn num(&self) -> Result<f64, String> {
        match self {
            Val::Num(n) => Ok(*n),
            Val::Bool(b) => Ok(*b as u8 as f64),
            other => Err(format!("expected a number, got {other}")),
        }
    }

    pub fn usize(&self) -> Result<usize, String> {
        let n = self.num()?;
        if n < 0.0 || n.fract() != 0.0 {
            return Err(format!("expected a non-negative integer, got {n}"));
        }
        Ok(n as usize)
    }

    pub fn str(&self) -> Result<&str, String> {
        match self {
            Val::Str(s) => Ok(s),
            other => Err(format!("expected a string, got {other}")),
        }
    }

    pub fn list(&self) -> Result<&[Val], String> {
        match self {
            Val::List(l) => Ok(l),
            other => Err(format!("expected a list, got {other}")),
        }
    }
}

impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Val::Null => write!(f, "null"),
            Val::Bool(b) => write!(f, "{b}"),
            Val::Num(n) if n.fract() == 0.0 && n.abs() < 1e15 => write!(f, "{}", *n as i64),
            Val::Num(n) => write!(f, "{n}"),
            Val::Str(s) => write!(f, "{s}"),
            Val::List(_) | Val::Obj(_) => write!(f, "{}", self.to_json()),
        }
    }
}

/// Name resolution for an evaluation. `lookup(root, path)` resolves `root` (a variable, or a
/// scope such as `cfg`/`gguf`/`json`/`runtime`) and a member/index path; `None` = undefined.
pub trait Env {
    fn lookup(&self, root: &str, path: &[Val]) -> Option<Val>;
    /// Whether a source tensor with this canonical name exists.
    fn has(&self, _tensor: &str) -> bool {
        false
    }
    /// Shape of a source tensor (canonical name).
    fn shape(&self, _tensor: &str) -> Option<Vec<usize>> {
        None
    }
    /// Elements per stored quantization block of a source tensor (1 for plain floats).
    fn block(&self, _tensor: &str) -> Option<usize> {
        None
    }
    /// Whether any raw (pre-rename) checkpoint tensor name matches glob `pat`.
    fn raw_has(&self, _pat: &str) -> bool {
        false
    }
    /// Values of a small source tensor (canonical name), for `values(name)`.
    fn values(&self, _tensor: &str) -> Option<Vec<f32>> {
        None
    }
    /// Stored dtype of a source tensor (`f32`, `f16`, `bf16`, `u32`, `quant`), for `dtype(name)`.
    fn dtype(&self, _tensor: &str) -> Option<&'static str> {
        None
    }
    /// Vocabulary id of a token string, for `token(s)`.
    fn token(&self, _s: &str) -> Option<u32> {
        None
    }
}

/// Follows `path` (object keys / list indices) into `v`.
pub fn walk(mut v: Val, path: &[Val]) -> Option<Val> {
    for seg in path {
        v = match (&v, seg) {
            (Val::Obj(o), Val::Str(k)) => Val::from_json(o.get(k)?),
            (Val::Obj(o), Val::Num(n)) => Val::from_json(o.get(&(*n as i64).to_string())?),
            (Val::List(l), Val::Num(n)) => {
                let i = if *n < 0.0 {
                    l.len() as i64 + *n as i64
                } else {
                    *n as i64
                };
                l.get(usize::try_from(i).ok()?)?.clone()
            }
            _ => return None,
        };
    }
    Some(v)
}

/// `parent` with one extra variable.
pub struct Bind<'a> {
    pub parent: &'a dyn Env,
    pub name: &'a str,
    pub val: Val,
}

impl Env for Bind<'_> {
    fn lookup(&self, root: &str, path: &[Val]) -> Option<Val> {
        if root == self.name {
            walk(self.val.clone(), path)
        } else {
            self.parent.lookup(root, path)
        }
    }
    fn has(&self, t: &str) -> bool {
        self.parent.has(t)
    }
    fn shape(&self, t: &str) -> Option<Vec<usize>> {
        self.parent.shape(t)
    }
    fn block(&self, t: &str) -> Option<usize> {
        self.parent.block(t)
    }
    fn raw_has(&self, p: &str) -> bool {
        self.parent.raw_has(p)
    }
    fn values(&self, t: &str) -> Option<Vec<f32>> {
        self.parent.values(t)
    }
    fn dtype(&self, t: &str) -> Option<&'static str> {
        self.parent.dtype(t)
    }
    fn token(&self, s: &str) -> Option<u32> {
        self.parent.token(s)
    }
}

/// A parsed expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Lit(Val),
    List(Vec<Expr>),
    /// `root` then `.member` / `[index]` segments.
    Ref(String, Vec<Seg>),
    Un(char, Box<Expr>),
    Bin(&'static str, Box<Expr>, Box<Expr>),
    Cond(Box<Expr>, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
    /// `e[i]` / `e.k` on a non-reference value (call result, list literal).
    Index(Box<Expr>, Box<Seg>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Seg {
    Key(String),
    Idx(Expr),
}

/// Scopes whose dotted member chain is one key (`gguf.attention.head_count`).
const DOTTED_ROOTS: &[&str] = &["gguf"];

pub fn parse(src: &str) -> Result<Expr, String> {
    let toks = lex(src)?;
    let mut p = Parser {
        t: &toks,
        i: 0,
        src,
    };
    let e = p.cond()?;
    if p.i != toks.len() {
        return Err(format!("`{src}`: unexpected `{}`", toks[p.i]));
    }
    Ok(e)
}

/// Parses and evaluates `src`.
pub fn eval_str(src: &str, env: &dyn Env) -> Result<Val, String> {
    eval(&parse(src)?, env).map_err(|e| {
        if e.contains('`') {
            e
        } else {
            format!("`{src}`: {e}")
        }
    })
}

/// Evaluates a JSON spec value: strings are expressions, numbers/bools/null literal, arrays and
/// objects element-wise. A string holding a quoted literal (`"'x'"`) is a string, and so is a bare
/// identifier no scope defines (enum variant names: `kind: "Add"`).
pub fn eval_json(j: &J, env: &dyn Env) -> Result<Val, String> {
    match j {
        J::String(s) => match parse(s)? {
            Expr::Ref(root, segs) if segs.is_empty() && env.lookup(&root, &[]).is_none() => {
                Ok(Val::Str(root))
            }
            e => eval(&e, env).map_err(|err| {
                if err.contains('`') {
                    err
                } else {
                    format!("`{s}`: {err}")
                }
            }),
        },
        J::Array(a) => Ok(Val::List(
            a.iter()
                .map(|x| eval_json(x, env))
                .collect::<Result<_, _>>()?,
        )),
        J::Object(o) => {
            let mut m = serde_json::Map::new();
            for (k, v) in o {
                m.insert(k.clone(), eval_json(v, env)?.to_json());
            }
            Ok(Val::Obj(m))
        }
        other => Ok(Val::from_json(other)),
    }
}

/// Replaces every `{expr}` in `s` with its value (`{{`/`}}` escape braces).
pub fn interp(s: &str, env: &dyn Env) -> Result<String, String> {
    if !s.contains('{') {
        return Ok(s.to_string());
    }
    let b = s.as_bytes();
    let (mut out, mut i) = (String::new(), 0);
    while i < b.len() {
        match b[i] {
            b'{' if b.get(i + 1) == Some(&b'{') => {
                out.push('{');
                i += 2;
            }
            b'}' if b.get(i + 1) == Some(&b'}') => {
                out.push('}');
                i += 2;
            }
            b'{' => {
                let (mut depth, mut j) = (1, i + 1);
                while j < b.len() && depth > 0 {
                    match b[j] {
                        b'{' => depth += 1,
                        b'}' => depth -= 1,
                        _ => {}
                    }
                    j += 1;
                }
                if depth != 0 {
                    return Err(format!("`{s}`: unclosed `{{`"));
                }
                let v = eval_str(&s[i + 1..j - 1], env)?;
                if v.is_null() {
                    return Err(format!("`{s}`: `{}` is undefined", &s[i + 1..j - 1]));
                }
                out.push_str(&v.to_string());
                i = j;
            }
            _ => {
                let c = s[i..].chars().next().unwrap();
                out.push(c);
                i += c.len_utf8();
            }
        }
    }
    Ok(out)
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Str(String),
    Id(String),
    Op(&'static str),
}

impl fmt::Display for Tok {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tok::Num(n) => write!(f, "{n}"),
            Tok::Str(s) => write!(f, "'{s}'"),
            Tok::Id(s) => write!(f, "{s}"),
            Tok::Op(s) => write!(f, "{s}"),
        }
    }
}

const OPS: &[&str] = &[
    "**", "//", "==", "!=", "<=", ">=", "&&", "||", "+", "-", "*", "/", "%", "<", ">", "!", "?",
    ":", "(", ")", "[", "]", ",", ".",
];

fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let b = src.as_bytes();
    let (mut out, mut i) = (Vec::new(), 0);
    'outer: while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() {
            let s = i;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.' || b[i] == b'_') {
                // `.` followed by a letter is member access, not a decimal point.
                if b[i] == b'.' && !b.get(i + 1).is_some_and(|x| x.is_ascii_digit()) {
                    break;
                }
                i += 1;
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                let save = i;
                i += 1;
                if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                    i += 1;
                }
                if i < b.len() && b[i].is_ascii_digit() {
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                } else {
                    i = save;
                }
            }
            let t: String = src[s..i].chars().filter(|&c| c != '_').collect();
            out.push(Tok::Num(
                t.parse()
                    .map_err(|_| format!("`{src}`: bad number `{t}`"))?,
            ));
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            let s = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push(Tok::Id(src[s..i].to_string()));
            continue;
        }
        if c == b'\'' || c == b'"' {
            let mut s = String::new();
            i += 1;
            while i < b.len() && b[i] != c {
                if b[i] == b'\\' && i + 1 < b.len() {
                    i += 1;
                }
                let ch = src[i..].chars().next().unwrap();
                s.push(ch);
                i += ch.len_utf8();
            }
            if i >= b.len() {
                return Err(format!("`{src}`: unterminated string"));
            }
            i += 1;
            out.push(Tok::Str(s));
            continue;
        }
        for op in OPS {
            if src[i..].starts_with(op) {
                out.push(Tok::Op(op));
                i += op.len();
                continue 'outer;
            }
        }
        return Err(format!("`{src}`: unexpected character `{}`", c as char));
    }
    Ok(out)
}

struct Parser<'a> {
    t: &'a [Tok],
    i: usize,
    src: &'a str,
}

impl Parser<'_> {
    fn peek_op(&self, op: &str) -> bool {
        matches!(self.t.get(self.i), Some(Tok::Op(o)) if *o == op)
    }
    fn eat(&mut self, op: &str) -> bool {
        let y = self.peek_op(op);
        if y {
            self.i += 1;
        }
        y
    }
    fn expect(&mut self, op: &str) -> Result<(), String> {
        if self.eat(op) {
            Ok(())
        } else {
            Err(format!("`{}`: expected `{op}`", self.src))
        }
    }
    fn cond(&mut self) -> Result<Expr, String> {
        let c = self.bin(0)?;
        if self.eat("?") {
            let a = self.cond()?;
            self.expect(":")?;
            let b = self.cond()?;
            return Ok(Expr::Cond(Box::new(c), Box::new(a), Box::new(b)));
        }
        Ok(c)
    }
    fn bin(&mut self, level: usize) -> Result<Expr, String> {
        const LEVELS: &[&[&str]] = &[
            &["||"],
            &["&&"],
            &["==", "!="],
            &["<", "<=", ">", ">="],
            &["+", "-"],
            &["*", "/", "//", "%"],
        ];
        if level == LEVELS.len() {
            return self.unary();
        }
        let mut l = self.bin(level + 1)?;
        'more: loop {
            for op in LEVELS[level] {
                if self.eat(op) {
                    let r = self.bin(level + 1)?;
                    l = Expr::Bin(op, Box::new(l), Box::new(r));
                    continue 'more;
                }
            }
            return Ok(l);
        }
    }
    fn unary(&mut self) -> Result<Expr, String> {
        if self.eat("!") {
            return Ok(Expr::Un('!', Box::new(self.unary()?)));
        }
        if self.eat("-") {
            return Ok(Expr::Un('-', Box::new(self.unary()?)));
        }
        let base = self.postfix()?;
        if self.eat("**") {
            // Right-associative, binds tighter than unary minus on its left operand only.
            let exp = self.unary()?;
            return Ok(Expr::Bin("**", Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }
    fn postfix(&mut self) -> Result<Expr, String> {
        let mut e = self.primary()?;
        loop {
            if self.eat("[") {
                let i = self.cond()?;
                self.expect("]")?;
                e = push_seg(e, Seg::Idx(i));
            } else if self.eat(".") {
                match self.t.get(self.i) {
                    Some(Tok::Id(k)) => {
                        let k = k.clone();
                        self.i += 1;
                        e = push_seg(e, Seg::Key(k));
                    }
                    Some(Tok::Num(n)) if n.fract() == 0.0 => {
                        let k = (*n as i64).to_string();
                        self.i += 1;
                        e = push_seg(e, Seg::Key(k));
                    }
                    _ => return Err(format!("`{}`: expected a member name after `.`", self.src)),
                }
            } else {
                return Ok(e);
            }
        }
    }
    fn primary(&mut self) -> Result<Expr, String> {
        let t = self
            .t
            .get(self.i)
            .cloned()
            .ok_or_else(|| format!("`{}`: unexpected end", self.src))?;
        self.i += 1;
        match t {
            Tok::Num(n) => Ok(Expr::Lit(Val::Num(n))),
            Tok::Str(s) => Ok(Expr::Lit(Val::Str(s))),
            Tok::Op("(") => {
                let e = self.cond()?;
                self.expect(")")?;
                Ok(e)
            }
            Tok::Op("[") => {
                let mut items = Vec::new();
                if !self.eat("]") {
                    loop {
                        items.push(self.cond()?);
                        if self.eat("]") {
                            break;
                        }
                        self.expect(",")?;
                        if self.eat("]") {
                            break;
                        }
                    }
                }
                Ok(Expr::List(items))
            }
            Tok::Id(id) => match id.as_str() {
                "true" => Ok(Expr::Lit(Val::Bool(true))),
                "false" => Ok(Expr::Lit(Val::Bool(false))),
                "null" => Ok(Expr::Lit(Val::Null)),
                _ if self.peek_op("(") => {
                    self.i += 1;
                    let mut args = Vec::new();
                    if !self.eat(")") {
                        loop {
                            args.push(self.cond()?);
                            if self.eat(")") {
                                break;
                            }
                            self.expect(",")?;
                        }
                    }
                    Ok(Expr::Call(id, args))
                }
                _ => Ok(Expr::Ref(id, Vec::new())),
            },
            Tok::Op(o) => Err(format!("`{}`: unexpected `{o}`", self.src)),
        }
    }
}

fn push_seg(e: Expr, s: Seg) -> Expr {
    match e {
        Expr::Ref(root, mut segs) => {
            // `gguf.a.b` is one dotted key: fold member names into the first segment.
            if DOTTED_ROOTS.contains(&root.as_str()) {
                if let (Seg::Key(k), Some(Seg::Key(prev))) = (&s, segs.last_mut()) {
                    prev.push('.');
                    prev.push_str(k);
                    return Expr::Ref(root, segs);
                }
            }
            segs.push(s);
            Expr::Ref(root, segs)
        }
        other => Expr::Index(Box::new(other), Box::new(s)),
    }
}

fn seg_val(s: &Seg, env: &dyn Env) -> Result<Val, String> {
    match s {
        Seg::Key(k) => Ok(Val::Str(k.clone())),
        Seg::Idx(e) => eval(e, env),
    }
}

pub fn eval(e: &Expr, env: &dyn Env) -> Result<Val, String> {
    match e {
        Expr::Lit(v) => Ok(v.clone()),
        Expr::List(items) => Ok(Val::List(
            items
                .iter()
                .map(|x| eval(x, env))
                .collect::<Result<_, _>>()?,
        )),
        Expr::Ref(root, segs) => {
            let path = segs
                .iter()
                .map(|s| seg_val(s, env))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(env.lookup(root, &path).unwrap_or(Val::Null))
        }
        Expr::Index(base, s) => {
            let b = eval(base, env)?;
            Ok(walk(b, &[seg_val(s, env)?]).unwrap_or(Val::Null))
        }
        Expr::Un('!', x) => Ok(Val::Bool(!eval(x, env)?.truthy())),
        Expr::Un(_, x) => Ok(Val::Num(-eval(x, env)?.num()?)),
        Expr::Cond(c, a, b) => {
            if eval(c, env)?.truthy() {
                eval(a, env)
            } else {
                eval(b, env)
            }
        }
        Expr::Bin(op, a, b) => {
            match *op {
                "&&" => {
                    let l = eval(a, env)?;
                    return if l.truthy() { eval(b, env) } else { Ok(l) };
                }
                "||" => {
                    let l = eval(a, env)?;
                    return if l.truthy() { Ok(l) } else { eval(b, env) };
                }
                _ => {}
            }
            let (l, r) = (eval(a, env)?, eval(b, env)?);
            binop(op, &l, &r)
        }
        Expr::Call(f, args) => call(f, args, env),
    }
}

fn binop(op: &str, l: &Val, r: &Val) -> Result<Val, String> {
    match op {
        "==" => return Ok(Val::Bool(eq(l, r))),
        "!=" => return Ok(Val::Bool(!eq(l, r))),
        "+" => {
            if let (Val::Str(a), b) | (b, Val::Str(a)) = (l, r) {
                // String concatenation keeps operand order.
                return Ok(Val::Str(match l {
                    Val::Str(_) => format!("{a}{b}"),
                    _ => format!("{b}{a}"),
                }));
            }
            if let (Val::List(a), Val::List(b)) = (l, r) {
                return Ok(Val::List(a.iter().chain(b).cloned().collect()));
            }
        }
        "<" | "<=" | ">" | ">=" => {
            if let (Val::Str(a), Val::Str(b)) = (l, r) {
                let o = a.cmp(b);
                return Ok(Val::Bool(match op {
                    "<" => o.is_lt(),
                    "<=" => o.is_le(),
                    ">" => o.is_gt(),
                    _ => o.is_ge(),
                }));
            }
        }
        _ => {}
    }
    let (a, b) = (
        l.num().map_err(|e| format!("`{op}`: {e}"))?,
        r.num().map_err(|e| format!("`{op}`: {e}"))?,
    );
    Ok(match op {
        "+" => Val::Num(a + b),
        "-" => Val::Num(a - b),
        "*" => Val::Num(a * b),
        "/" => Val::Num(a / b),
        "//" => Val::Num((a / b).floor()),
        "%" => Val::Num(a.rem_euclid(b)),
        "**" => Val::Num(a.powf(b)),
        "<" => Val::Bool(a < b),
        "<=" => Val::Bool(a <= b),
        ">" => Val::Bool(a > b),
        ">=" => Val::Bool(a >= b),
        _ => return Err(format!("unknown operator `{op}`")),
    })
}

fn eq(l: &Val, r: &Val) -> bool {
    match (l, r) {
        (Val::Num(a), Val::Bool(b)) | (Val::Bool(b), Val::Num(a)) => *a == *b as u8 as f64,
        _ => l == r,
    }
}

fn call(f: &str, args: &[Expr], env: &dyn Env) -> Result<Val, String> {
    // Lazy: `first` stops at the first defined argument.
    if f == "first" {
        for a in args {
            let v = eval(a, env)?;
            if !v.is_null() {
                return Ok(v);
            }
        }
        return Ok(Val::Null);
    }
    if f == "count" {
        // count(pattern): consecutive k = 0.. for which `pattern` with {k} names a tensor.
        let pat = eval(args.first().ok_or("count(pattern)")?, env)?;
        let pat = pat.str()?;
        let mut k = 0usize;
        loop {
            let b = Bind {
                parent: env,
                name: "k",
                val: Val::Num(k as f64),
            };
            if !env.has(&interp(pat, &b)?) {
                return Ok(Val::Num(k as f64));
            }
            k += 1;
        }
    }
    let v = args
        .iter()
        .map(|a| eval(a, env))
        .collect::<Result<Vec<_>, _>>()?;
    let arg = |i: usize| {
        v.get(i)
            .ok_or_else(|| format!("{f}(): missing argument {}", i + 1))
    };
    let nums = || v.iter().map(Val::num).collect::<Result<Vec<_>, _>>();
    Ok(match f {
        "min" | "max" => {
            let xs = match v.as_slice() {
                [Val::List(l)] => l.iter().map(Val::num).collect::<Result<Vec<_>, _>>()?,
                _ => nums()?,
            };
            let it = xs.into_iter();
            Val::Num(if f == "min" {
                it.fold(f64::INFINITY, f64::min)
            } else {
                it.fold(f64::NEG_INFINITY, f64::max)
            })
        }
        "sqrt" => Val::Num(arg(0)?.num()?.sqrt()),
        "ceil" => Val::Num(arg(0)?.num()?.ceil()),
        "floor" => Val::Num(arg(0)?.num()?.floor()),
        "round" => Val::Num(arg(0)?.num()?.round()),
        "abs" => Val::Num(arg(0)?.num()?.abs()),
        "ln" => Val::Num(arg(0)?.num()?.ln()),
        "exp" => Val::Num(arg(0)?.num()?.exp()),
        "len" => Val::Num(match arg(0)? {
            Val::List(l) => l.len(),
            Val::Str(s) => s.chars().count(),
            Val::Obj(o) => o.len(),
            Val::Null => 0,
            other => return Err(format!("len({other})")),
        } as f64),
        "at" => walk(arg(0)?.clone(), &[arg(1)?.clone()]).unwrap_or(Val::Null),
        "contains" => Val::Bool(match arg(0)? {
            Val::List(l) => l.iter().any(|x| eq(x, arg(1).unwrap())),
            Val::Str(s) => s.contains(arg(1)?.str()?),
            Val::Obj(o) => o.contains_key(arg(1)?.str()?),
            _ => false,
        }),
        "range" => {
            let (a, b) = if v.len() >= 2 {
                (arg(0)?.num()?, arg(1)?.num()?)
            } else {
                (0.0, arg(0)?.num()?)
            };
            Val::List((a as i64..b as i64).map(|x| Val::Num(x as f64)).collect())
        }
        "has" => Val::Bool(env.has(arg(0)?.str()?)),
        "shape" => match env.shape(arg(0)?.str()?) {
            Some(s) => Val::List(s.into_iter().map(|d| Val::Num(d as f64)).collect()),
            None => Val::Null,
        },
        "block" => match env.block(arg(0)?.str()?) {
            Some(b) => Val::Num(b as f64),
            None => Val::Null,
        },
        "raw_has" => Val::Bool(env.raw_has(arg(0)?.str()?)),
        "first_has" => {
            for x in &v {
                if env.has(x.str()?) {
                    return Ok(x.clone());
                }
            }
            Val::Null
        }
        "fmt" => Val::Str(interp(arg(0)?.str()?, env)?),
        "str" => Val::Str(arg(0)?.to_string()),
        "int" => Val::Num(arg(0)?.num()?.trunc()),
        "list" => {
            // list(n, x): n copies of x.
            let n = arg(0)?.usize()?;
            Val::List(vec![arg(1)?.clone(); n])
        }
        // values(name): the values of a small source tensor (null when absent).
        "values" => match env.values(arg(0)?.str()?) {
            Some(v) => Val::List(v.into_iter().map(|x| Val::Num(x as f64)).collect()),
            None => Val::Null,
        },
        // dtype(name): the stored dtype of a source tensor (null when absent).
        "dtype" => env.dtype(arg(0)?.str()?).map_or(Val::Null, |d| Val::Str(d.into())),
        // token(s): the vocabulary id of token string `s` (null when absent).
        "token" => env.token(arg(0)?.str()?).map_or(Val::Null, |t| Val::Num(t as f64)),
        // bf16(x): x rounded to the nearest bfloat16 (through f32).
        "bf16" => Val::Num(crate::model::bf16_to_f32(crate::model::f32_to_bf16(arg(0)?.num()? as f32)) as f64),
        // rfind(list, x, end): the last index below `end` (default len) holding x, else -1.
        "rfind" => {
            let l = arg(0)?.list()?;
            let end = match v.get(2) {
                Some(e) => e.usize()?.min(l.len()),
                None => l.len(),
            };
            let x = arg(1)?;
            Val::Num(l[..end].iter().rposition(|y| eq(y, x)).map_or(-1.0, |i| i as f64))
        }
        _ => return Err(format!("unknown function `{f}`")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct E(J, Vec<&'static str>);
    impl Env for E {
        fn lookup(&self, root: &str, path: &[Val]) -> Option<Val> {
            walk(Val::from_json(self.0.get(root)?), path)
        }
        fn has(&self, t: &str) -> bool {
            self.1.contains(&t)
        }
        fn shape(&self, t: &str) -> Option<Vec<usize>> {
            self.has(t).then(|| vec![4, 8])
        }
        fn values(&self, t: &str) -> Option<Vec<f32>> {
            (t == "x.w").then(|| vec![0.5, 1.0])
        }
        fn dtype(&self, t: &str) -> Option<&'static str> {
            (t == "x.w").then_some("bf16")
        }
    }
    fn env() -> E {
        E(
            json!({"cfg": {"hidden_size": 64, "heads": 4, "layer_types": ["full", "sliding"], "sub": {"x": 2.5}},
                   "gguf": {"llama.block_count": 2}, "n": 3}),
            vec!["blk.0.w", "blk.1.w", "blk.2.w", "x.w"],
        )
    }
    fn ev(s: &str) -> Val {
        eval_str(s, &env()).unwrap()
    }

    #[test]
    fn precedence_and_operators() {
        assert_eq!(ev("1 + 2 * 3"), Val::Num(7.0));
        assert_eq!(ev("(1 + 2) * 3"), Val::Num(9.0));
        assert_eq!(ev("2 ** 3 ** 2"), Val::Num(512.0));
        assert_eq!(ev("-2 ** 2"), Val::Num(-4.0));
        assert_eq!(ev("7 // 2 + 7 % 4"), Val::Num(6.0));
        assert_eq!(ev("1 < 2 && 3 >= 3 || false"), Val::Bool(true));
        assert_eq!(ev("!(1 == 1) ? 'a' : 'b'"), Val::Str("b".into()));
        assert_eq!(ev("cfg.hidden_size / cfg.heads"), Val::Num(16.0));
        assert_eq!(ev("cfg.layer_types[1] == 'sliding'"), Val::Bool(true));
        assert_eq!(ev("cfg['sub'].x"), Val::Num(2.5));
        assert_eq!(ev("1e-6 * 2"), Val::Num(2e-6));
        assert_eq!(ev("'a' + n"), Val::Str("a3".into()));
    }

    #[test]
    fn missing_keys_and_first() {
        assert_eq!(ev("cfg.missing"), Val::Null);
        assert_eq!(ev("first(cfg.missing, cfg.sub.nope, 7)"), Val::Num(7.0));
        assert_eq!(ev("first(cfg.hidden_size, 1)"), Val::Num(64.0));
        assert!(eval_str("cfg.missing + 1", &env()).is_err());
        assert_eq!(
            ev("gguf.block_count"),
            Val::Null,
            "dotted root resolves through the env, not walk"
        );
    }

    #[test]
    fn functions() {
        assert_eq!(ev("min(3, 1, 2) + max([4, 9])"), Val::Num(10.0));
        assert_eq!(ev("len(cfg.layer_types)"), Val::Num(2.0));
        assert_eq!(ev("at(cfg.layer_types, -1)"), Val::Str("sliding".into()));
        assert_eq!(ev("contains(cfg.layer_types, 'full')"), Val::Bool(true));
        assert_eq!(ev("count('blk.{k}.w')"), Val::Num(3.0));
        assert_eq!(ev("has('x.w') && !has('y.w')"), Val::Bool(true));
        assert_eq!(ev("shape('x.w')[1]"), Val::Num(8.0));
        assert_eq!(ev("first_has('y.w', 'x.w')"), Val::Str("x.w".into()));
        assert_eq!(ev("fmt('l{n * 2}')"), Val::Str("l6".into()));
        assert_eq!(ev("sqrt(16) + ceil(1.2) + floor(1.8)"), Val::Num(7.0));
        assert_eq!(
            ev("range(3)"),
            Val::List(vec![Val::Num(0.0), Val::Num(1.0), Val::Num(2.0)])
        );
        assert_eq!(ev("at(values('x.w'), 1) + len(values('y.w'))"), Val::Num(1.0));
        assert_eq!(ev("dtype('x.w') == 'bf16' && dtype('y.w') == null"), Val::Bool(true));
        assert_eq!(ev("bf16(sqrt(1536))"), Val::Num(39.25));
        assert_eq!(ev("rfind([1, 0, 1, 1, 0], 1, 3) + rfind([1, 0], 0) + rfind(cfg.layer_types, 'x')"), Val::Num(2.0));
    }

    #[test]
    fn interpolation() {
        let e = env();
        assert_eq!(
            interp("model.layers.{n - 1}.w", &e).unwrap(),
            "model.layers.2.w"
        );
        assert_eq!(interp("a{{b}}", &e).unwrap(), "a{b}");
        assert!(interp("{cfg.missing}", &e).is_err());
        assert_eq!(
            eval_json(
                &json!({"eps": "1e-5", "heads": "cfg.heads", "k": [1, "n"]}),
                &e
            )
            .unwrap()
            .to_json(),
            json!({"eps": 1e-5, "heads": 4, "k": [1, 3]})
        );
    }
}
