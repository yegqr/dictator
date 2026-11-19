// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025-2026 The Dictator authors (see AUTHORS)
//! dictate-tinyasr: the small speech model (tinyasr, github.com/dkaraush/tinyasr) for the Dictator helper.
//!
//! It speaks the "pieces" part of the protocol of `helper/parakeet.py --worker`, the only part the helper uses:
//!
//! ```text
//! dictate-tinyasr --worker DIR --packs en-ru-uk,en-es-it-pt [--threads N]
//! ```
//!
//! loads DIR/<pack>.tzf for each listed pack and DIR/silero_vad.onnx, prints `{"ready": true, "load_s": ...}` (or
//! `{"error": ...}` and exits 1), then per request reads one JSON line `{"n": BYTES, "lang": "pieces", "langs":
//! [codes, main first], "gate_db": float}` followed by BYTES (even, 0 < BYTES <= MAX_BYTES) of raw 16 kHz mono s16
//! audio (one piece of a recording, see Streamer in helper/dictate), and answers one JSON line
//! `{"cands": {code: [text, score]}, "main": code, "note", "asr_s"}`: one candidate per enabled language that a loaded
//! pack has, the main one (the first of them) first; the helper picks the language once, over all pieces of the
//! recording. score: `cand_score`. Ends (exit 0) at the end of its input.
//!
//! - a request that cannot be read (not JSON, bad size): `{"error": "bad request: ..."}` and the worker ends (exit 0),
//!   as parakeet.py; anything else wrong: `{"error": ...}` and it goes on. A "lang" other than "pieces" (parakeet.py's
//!   "auto" or a language code, never sent by the helper) is such an error: `{"error": "ValueError: lang"}`.
//!
//! A language is read by the first loaded pack in the order of `PACKS` that has it (en-ru-uk > en-es-it-pt >
//! en-ru-uz > en-fa > en-km), all of a pack's languages with one `Transcription::candidates` call (its VAD runs
//! once), beam 4, no end-of-utterance head. --threads N: the engine's threads (TINYASR_THREADS; 0: the engine's own
//! default).
//!
//! `dictate-tinyasr --netcheck` prints "blocked" only if it is in its own empty network namespace and cannot reach the
//! internet (1.1.1.1:443), else "OPEN": the helper's selftest runs it in the worker's sandbox.

use std::io::{self, BufRead, Read, Write};
use std::time::{Duration, Instant};

use tinyasr::{Files, Lang, Models, Score, Transcription};

const SR: usize = 16_000;
/// Largest request: 5 minutes of audio. The helper sends pieces of at most about 30 s, and a request takes about 22
/// bytes of memory per sample (the voice gate works in f64): ~210 MB at this size, ~1.3 GB at parakeet.py's hour.
const MAX_BYTES: usize = 2 * SR * 300;
/// Longest request line (its JSON); a longer one is a bad request.
const MAX_LINE: u64 = 1 << 16;
const BEAM: usize = 4;
/// A number that is not finite is written as this (JSON has no NaN or infinity).
const BAD_NUMBER: f64 = -1e9;
/// The language packs (models/<name>.tzf.gz of tinyasr) with their languages, in the order of priority: a language
/// several loaded packs have is read by the first. Names are checked against this list, so no other file is opened.
const PACKS: [(&str, &[Lang]); 5] = [
    ("en-ru-uk", &[Lang::En, Lang::Ru, Lang::Uk]),
    ("en-es-it-pt", &[Lang::En, Lang::Es, Lang::It, Lang::Pt]),
    ("en-ru-uz", &[Lang::En, Lang::Ru, Lang::Uz]),
    ("en-fa", &[Lang::En, Lang::Fa]),
    ("en-km", &[Lang::En, Lang::Km]),
];
const MAX_MODEL: u64 = 64 << 20;
const MAX_VAD: u64 = 8 << 20;

/// The score of one candidate (a language's reading of a piece), the number in `cands`: higher is surer. The helper
/// sums it over the pieces of a recording and divides by their seconds; `dur` is this piece's real length (n / 2 /
/// 16000 s), so each piece counts by its length.
///
/// score = (path_logp / seconds + 0.5 * max(token_logp / tokens, -5)) * dur, natural logs, with -50 per second when
/// nothing was decoded and -5 as the mean when nothing was written. Why both terms:
/// the best-path log-probability per second alone fails, as a decoder in the wrong language (English on Ukrainian or
/// Russian speech) writes almost nothing, and a path of blanks is likely, so its path score is high; the mean token
/// log-probability punishes such a reading (few, unsure tokens), floored at -5 so one empty or garbage piece cannot
/// sink a whole recording. Always finite.
fn cand_score(s: &Score, dur: f64) -> f64 {
    let per_s = if s.seconds > 0.0 {
        s.path_logp as f64 / s.seconds as f64
    } else {
        -50.0
    };
    let per_s = if per_s.is_finite() { per_s } else { -50.0 };
    let mean = if s.tokens > 0 {
        (s.token_logp as f64 / s.tokens as f64).max(-5.0)
    } else {
        -5.0
    }; // max(NaN, -5) = -5
    let dur = if dur.is_finite() { dur } else { 0.0 };
    (per_s + 0.5 * mean) * dur
}

// ------------------------------------------------------------------------------------------------ JSON
/// A JSON value (only what the protocol needs; numbers are f64).
#[derive(Debug, Clone, PartialEq)]
enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    fn get(&self, k: &str) -> Option<&J> {
        match self {
            J::Obj(v) => v.iter().rev().find(|(a, _)| a == k).map(|(_, b)| b),
            _ => None,
        }
    }
    fn num(&self) -> Option<f64> {
        match self {
            J::Num(x) => Some(*x),
            _ => None,
        }
    }
    fn str(&self) -> Option<&str> {
        match self {
            J::Str(s) => Some(s),
            _ => None,
        }
    }
}

const MAX_DEPTH: usize = 16;

/// Parses one JSON document (with surrounding white space); None if it is not exactly that.
fn parse(b: &[u8]) -> Option<J> {
    let mut p = Parser { b, i: 0 };
    let v = p.value(0)?;
    p.ws();
    if p.i == b.len() { Some(v) } else { None }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while matches!(self.b.get(self.i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }
    fn lit(&mut self, s: &[u8], v: J) -> Option<J> {
        if self.b.get(self.i..self.i + s.len())? == s {
            self.i += s.len();
            Some(v)
        } else {
            None
        }
    }
    fn value(&mut self, depth: usize) -> Option<J> {
        if depth > MAX_DEPTH {
            return None;
        }
        self.ws();
        match *self.b.get(self.i)? {
            b'n' => self.lit(b"null", J::Null),
            b't' => self.lit(b"true", J::Bool(true)),
            b'f' => self.lit(b"false", J::Bool(false)),
            b'"' => self.string().map(J::Str),
            b'[' => {
                self.i += 1;
                let mut v = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Some(J::Arr(v));
                }
                loop {
                    v.push(self.value(depth + 1)?);
                    self.ws();
                    match *self.b.get(self.i)? {
                        b',' => self.i += 1,
                        b']' => {
                            self.i += 1;
                            return Some(J::Arr(v));
                        }
                        _ => return None,
                    }
                }
            }
            b'{' => {
                self.i += 1;
                let mut v = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Some(J::Obj(v));
                }
                loop {
                    self.ws();
                    if self.b.get(self.i) != Some(&b'"') {
                        return None;
                    }
                    let k = self.string()?;
                    self.ws();
                    if self.b.get(self.i) != Some(&b':') {
                        return None;
                    }
                    self.i += 1;
                    v.push((k, self.value(depth + 1)?));
                    self.ws();
                    match *self.b.get(self.i)? {
                        b',' => self.i += 1,
                        b'}' => {
                            self.i += 1;
                            return Some(J::Obj(v));
                        }
                        _ => return None,
                    }
                }
            }
            b'-' | b'0'..=b'9' => self.number(),
            _ => None,
        }
    }
    fn number(&mut self) -> Option<J> {
        let s = self.i;
        let digits = |p: &mut Self| {
            let a = p.i;
            while matches!(p.b.get(p.i), Some(b'0'..=b'9')) {
                p.i += 1;
            }
            p.i > a
        };
        if self.b.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        if self.b.get(self.i) == Some(&b'0') {
            self.i += 1;
        } else if !digits(self) {
            return None;
        }
        if self.b.get(self.i) == Some(&b'.') {
            self.i += 1;
            if !digits(self) {
                return None;
            }
        }
        if matches!(self.b.get(self.i), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.b.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if !digits(self) {
                return None;
            }
        }
        let x: f64 = std::str::from_utf8(&self.b[s..self.i]).ok()?.parse().ok()?;
        x.is_finite().then_some(J::Num(x))
    }
    fn hex4(&mut self) -> Option<u32> {
        let h = std::str::from_utf8(self.b.get(self.i..self.i + 4)?).ok()?;
        let v = u32::from_str_radix(h, 16).ok()?;
        if !h.bytes().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        self.i += 4;
        Some(v)
    }
    fn string(&mut self) -> Option<String> {
        self.i += 1; // the opening quote
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = *self.b.get(self.i)?;
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out).ok(),
                b'\\' => {
                    let e = *self.b.get(self.i)?;
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let a = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&a) {
                                if self.b.get(self.i..self.i + 2)? != b"\\u" {
                                    return None;
                                }
                                self.i += 2;
                                let b = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&b) {
                                    return None;
                                }
                                0x10000 + ((a - 0xD800) << 10) + (b - 0xDC00)
                            } else {
                                a
                            };
                            char::from_u32(cp)?
                        }
                        _ => return None,
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                0..=0x1f => return None,
                _ => out.push(c),
            }
        }
    }
}

/// A JSON string literal: quotes, backslashes and control characters escaped, other UTF-8 as it is.
fn js(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 || c == '\u{7f}' || c == '\u{2028}' || c == '\u{2029}' => {
                o.push_str(&format!("\\u{:04x}", c as u32))
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// A JSON number, rounded to `dec` decimals; never NaN or infinite.
fn jn(x: f64, dec: i32) -> String {
    let x = if x.is_finite() { x } else { BAD_NUMBER };
    if x.abs() >= 1e15 {
        return format!("{x:e}"); // no 300 digits for a huge value
    }
    let p = 10f64.powi(dec);
    let r = (x * p).round() / p;
    if r == 0.0 { "0".into() } else { format!("{r}") }
}

fn error(msg: &str) -> String {
    format!("{{\"error\": {}}}", js(msg))
}

// ------------------------------------------------------------------------------------------------ audio
/// The voice gate, as gate() in parakeet.py: mutes what is much quieter than the speaker (other people in the room, a
/// TV). The speaker's level is the 95th percentile of 50 ms frames; frames more than drop_db below it (or loud for
/// less than 100 ms) are silenced, keeping 150 ms around every loud frame, with 10 ms ramps. Returns the share of the
/// time muted.
fn gate(a: &mut [f32], drop_db: f64) -> f64 {
    if drop_db <= 0.0 || a.len() < SR / 2 {
        return 0.0;
    }
    let hop = SR / 20;
    let n = a.len() / hop;
    let db: Vec<f64> = (0..n)
        .map(|f| {
            let e = a[f * hop..(f + 1) * hop]
                .iter()
                .map(|&x| x as f64 * x as f64)
                .sum::<f64>()
                / hop as f64;
            10.0 * (e + 1e-10).log10()
        })
        .collect();
    let mut s = db.clone();
    s.sort_by(|x, y| x.total_cmp(y));
    let pos = 0.95 * (n - 1) as f64; // numpy's percentile (linear)
    let (lo, frac) = (pos.floor() as usize, pos - pos.floor());
    let p95 = s[lo] + (s[(lo + 1).min(n - 1)] - s[lo]) * frac;
    let loud: Vec<bool> = db.iter().map(|&d| d > p95 - drop_db).collect();
    // a lone loud frame is a peak of a far voice (np.roll wraps around)
    let loud2: Vec<bool> = (0..n)
        .map(|i| loud[i] && (loud[(i + n - 1) % n] || loud[(i + 1) % n]))
        .collect();
    let keep: Vec<bool> = (0..n)
        .map(|i| loud2[i.saturating_sub(3)..(i + 4).min(n)].iter().any(|&v| v))
        .collect();
    let muted = keep.iter().filter(|&&k| !k).count() as f64 / n as f64;
    // the gain per sample (the last partial frame as the last frame), smoothed over 10 ms (np.convolve "same")
    let g: Vec<f64> = (0..a.len())
        .map(|i| if keep[(i / hop).min(n - 1)] { 1.0 } else { 0.0 })
        .collect();
    let ramp = SR / 100;
    let mut pre = vec![0.0f64; g.len() + 1];
    for (i, v) in g.iter().enumerate() {
        pre[i + 1] = pre[i] + v;
    }
    let half = (ramp - 1) / 2; // numpy centres an even kernel at (len - 1) / 2
    for (i, x) in a.iter_mut().enumerate() {
        let lo = (i + half + 1).saturating_sub(ramp);
        let hi = (i + half + 1).min(g.len());
        let gi = (pre[hi] - pre[lo]) / ramp as f64;
        *x = (*x as f64 * gi) as f32;
    }
    muted
}

// ------------------------------------------------------------------------------------------------ requests
/// Loaded packs: (name, languages); the model's own languages at run time.
type Loaded<'a> = [(&'a str, &'a [Lang])];

/// Index in `loaded` of the pack that reads `l`: the first in the order of PACKS that is loaded and has it.
fn pack_for(l: Lang, loaded: &Loaded) -> Option<usize> {
    PACKS
        .iter()
        .find_map(|(name, _)| loaded.iter().position(|(n, ls)| n == name && ls.contains(&l)))
}

/// `langs` grouped by the pack that reads them: (index in `loaded`, its languages in the order of `langs`), in the
/// order of each group's first language. Languages no loaded pack has are left out.
fn groups(langs: &[Lang], loaded: &Loaded) -> Vec<(usize, Vec<Lang>)> {
    let mut g: Vec<(usize, Vec<Lang>)> = Vec::new();
    for &l in langs {
        let Some(p) = pack_for(l, loaded) else { continue };
        match g.iter_mut().find(|(q, _)| *q == p) {
            Some((_, v)) => v.push(l),
            None => g.push((p, vec![l])),
        }
    }
    g
}

/// The first line of a request: its JSON object and the audio's size in bytes; Err: the request cannot be read
/// (the worker answers the error and ends).
fn head(line: &[u8]) -> Result<(J, usize), String> {
    let req = parse(line).ok_or("bad request: not JSON")?;
    if !matches!(req, J::Obj(_)) {
        return Err("bad request: not an object".into());
    }
    let n = req.get("n").and_then(J::num).ok_or("bad request: n")?;
    if !(n > 0.0 && n <= MAX_BYTES as f64 && n.fract() == 0.0 && (n as usize) % 2 == 0) {
        return Err("bad request: size".into());
    }
    Ok((req, n as usize))
}

/// What a request asks for.
#[derive(Debug, PartialEq)]
struct Job {
    /// The enabled languages a loaded pack has, main first (all loaded languages if the request names none).
    langs: Vec<Lang>,
    gate_db: f64,
}

/// Python's false values: parakeet.py reads fields as `req.get(k) or default`.
fn falsy(v: &J) -> bool {
    match v {
        J::Null | J::Bool(false) => true,
        J::Num(x) => *x == 0.0,
        J::Str(s) => s.is_empty(),
        J::Arr(a) => a.is_empty(),
        J::Obj(o) => o.is_empty(),
        J::Bool(true) => false,
    }
}

/// Python's float() of a field: a number, a bool, or a string holding a number.
fn pyfloat(v: &J) -> Option<f64> {
    match v {
        J::Num(x) => Some(*x),
        J::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        J::Str(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// What a request asks for; Err: an error answer (the worker goes on). "lang" must be "pieces": the helper sends
/// nothing else.
fn job(req: &J, loaded: &Loaded) -> Result<Job, String> {
    if req.get("lang").and_then(J::str) != Some("pieces") {
        return Err("ValueError: lang".into());
    }
    let gate_db = match req.get("gate_db") {
        Some(v) if !falsy(v) => pyfloat(v).ok_or("ValueError: gate_db")?,
        _ => 0.0,
    };
    let have = |l: &Lang| pack_for(*l, loaded).is_some();
    let mut langs: Vec<Lang> = Vec::new();
    match req.get("langs") {
        Some(J::Arr(v)) if !v.is_empty() => {
            for c in v {
                let c = c.str().ok_or("ValueError: langs")?;
                // a language no loaded pack has (another model's, e.g. "de") is left out
                if let Some(l) = Lang::from_code(c).filter(have) {
                    if !langs.contains(&l) {
                        langs.push(l);
                    }
                }
            }
        }
        Some(v) if !falsy(v) => return Err("ValueError: langs".into()),
        _ => {
            for (name, _) in PACKS {
                for (_, ls) in loaded.iter().filter(|(n, _)| *n == name) {
                    for &l in ls.iter() {
                        if !langs.contains(&l) {
                            langs.push(l);
                        }
                    }
                }
            }
        }
    }
    if langs.is_empty() {
        return Err("ValueError: langs: no loaded pack has any of them".into());
    }
    Ok(Job { langs, gate_db })
}

/// One language's reading of the audio.
#[derive(Debug, Clone, PartialEq)]
struct Cand {
    lang: Lang,
    text: String,
    score: f64,
}

/// The answer line: the candidates (main first), and in the note the share of the audio the voice gate muted.
fn reply(cands: &[Cand], muted: f64, asr_s: f64) -> String {
    let Some(main) = cands.first() else {
        return error("RuntimeError: no candidate");
    };
    let c: Vec<String> = cands
        .iter()
        .map(|c| format!("{}: [{}, {}]", js(c.lang.code()), js(&c.text), jn(c.score, 4)))
        .collect();
    let note = if muted != 0.0 {
        format!(", quieter voices muted {:.0}%", 100.0 * muted)
    } else {
        String::new()
    };
    let (main, note, asr_s) = (js(main.lang.code()), js(&note), jn(asr_s, 2));
    format!(
        "{{\"cands\": {{{}}}, \"main\": {main}, \"note\": {note}, \"asr_s\": {asr_s}}}",
        c.join(", ")
    )
}

// ------------------------------------------------------------------------------------------------ recognition
struct Pack {
    name: &'static str,
    m: Models,
}

fn loaded(packs: &[Pack]) -> Vec<(&str, &[Lang])> {
    packs.iter().map(|p| (p.name, p.m.langs())).collect()
}

/// Each of `langs` (a loaded pack has them all) read by its pack, in the order of `langs`.
fn read(packs: &[Pack], audio: &[f32], langs: &[Lang]) -> Vec<Cand> {
    let dur = audio.len() as f64 / SR as f64; // n / 2 / 16000: the gate keeps the length
    let mut out: Vec<Cand> = Vec::new();
    for (p, ls) in groups(langs, &loaded(packs)) {
        for c in Transcription::candidates(&packs[p].m, audio, &ls, BEAM) {
            out.push(Cand {
                lang: c.lang,
                text: c.text.trim().to_string(),
                score: cand_score(&c.score, dur),
            });
        }
    }
    langs
        .iter()
        .map(|&l| {
            let c = out.iter().find(|c| c.lang == l).cloned();
            c.unwrap_or(Cand {
                lang: l,
                text: String::new(),
                score: BAD_NUMBER,
            })
        })
        .collect()
}

/// The answer to one request (a JSON line).
fn answer(packs: &[Pack], req: &J, raw: &[u8]) -> String {
    let t0 = Instant::now();
    let job = match job(req, &loaded(packs)) {
        Ok(j) => j,
        Err(e) => return error(&e),
    };
    let mut audio: Vec<f32> = raw
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect();
    let muted = gate(&mut audio, job.gate_db);
    let cands = read(packs, &audio, &job.langs);
    reply(&cands, muted, t0.elapsed().as_secs_f64())
}

// ------------------------------------------------------------------------------------------------ process
/// One line of at most MAX_LINE bytes; None at the end of input (also inside a line), Err if it is too long.
fn read_line(inp: &mut impl BufRead) -> Result<Option<Vec<u8>>, ()> {
    let mut buf = Vec::new();
    match inp.by_ref().take(MAX_LINE).read_until(b'\n', &mut buf) {
        Ok(0) | Err(_) => Ok(None),
        Ok(_) if buf.last() == Some(&b'\n') => Ok(Some(buf)),
        Ok(_) if buf.len() as u64 >= MAX_LINE => Err(()),
        Ok(_) => Ok(None),
    }
}

fn read_file(path: &str, max: u64) -> Result<Vec<u8>, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut b = Vec::new();
    f.take(max + 1)
        .read_to_end(&mut b)
        .map_err(|e| format!("{path}: {e}"))?;
    if b.len() as u64 > max {
        return Err(format!("{path}: too large"));
    }
    Ok(b)
}

fn load(dir: &str, names: &[&'static str]) -> Result<Vec<Pack>, String> {
    let vad = read_file(&format!("{dir}/silero_vad.onnx"), MAX_VAD)?;
    let mut packs = Vec::new();
    for &name in names {
        let asr = read_file(&format!("{dir}/{name}.tzf"), MAX_MODEL)?;
        let m = Models::load(Files { asr: &asr, vad: &vad }).map_err(|e| format!("{name}: {e}"))?;
        packs.push(Pack { name, m });
    }
    Ok(packs)
}

/// Writes one answer line; false if the output is gone.
fn say(out: &mut impl Write, line: &str) -> bool {
    writeln!(out, "{line}").and_then(|_| out.flush()).is_ok()
}

fn worker(dir: &str, names: &[&'static str]) -> i32 {
    let t0 = Instant::now();
    let mut out = io::stdout().lock();
    let packs = match load(dir, names) {
        Ok(p) => p,
        Err(e) => {
            say(&mut out, &error(&format!("load: {e}")));
            return 1;
        }
    };
    if !say(
        &mut out,
        &format!("{{\"ready\": true, \"load_s\": {}}}", jn(t0.elapsed().as_secs_f64(), 2)),
    ) {
        return 0;
    }
    let mut inp = io::BufReader::with_capacity(1 << 16, io::stdin().lock());
    loop {
        let line = match read_line(&mut inp) {
            Ok(Some(l)) => l,
            Ok(None) => return 0,
            Err(()) => {
                say(&mut out, &error("bad request: line too long"));
                return 0;
            }
        };
        let (req, n) = match head(&line) {
            Ok(r) => r,
            Err(e) => {
                say(&mut out, &error(&e));
                return 0;
            }
        };
        let mut raw = vec![0u8; n];
        if inp.read_exact(&mut raw).is_err() {
            return 0;
        }
        if !say(&mut out, &answer(&packs, &req, &raw)) {
            return 0;
        }
    }
}

/// Whether /proc/self/net/dev (its text) lists no network interface but loopback: an own, empty network namespace.
fn lo_only(dev: &str) -> bool {
    dev.lines().skip(2).all(|l| l.trim_start().starts_with("lo:"))
}

/// "blocked" only if this process is in its own empty network namespace (as in the helper's sandbox) and cannot reach
/// the internet (1.1.1.1:443, by IP: no DNS needed); else "OPEN". A failed connection alone is not enough: an offline
/// computer or a firewall would pass without any sandbox.
fn netcheck() -> i32 {
    let own = std::fs::read_to_string("/proc/self/net/dev").is_ok_and(|s| lo_only(&s));
    let addr = std::net::SocketAddr::from(([1, 1, 1, 1], 443));
    let open = !own || std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(3)).is_ok();
    say(&mut io::stdout().lock(), if open { "OPEN" } else { "blocked" });
    0
}

fn usage() -> i32 {
    say(
        &mut io::stdout().lock(),
        &error("usage: dictate-tinyasr --worker DIR --packs PACK[,PACK] [--threads N] | --netcheck"),
    );
    2
}

/// The arguments after --worker: `DIR --packs a,b [--threads N]`; None if anything is wrong.
fn cli(a: &[String]) -> Option<(String, Vec<&'static str>, usize)> {
    let (dir, mut rest) = a.split_first()?;
    let (mut packs, mut threads) = (None, 0usize);
    while let [k, v, tail @ ..] = rest {
        match k.as_str() {
            "--packs" if packs.is_none() => {
                let mut p: Vec<&'static str> = Vec::new();
                for name in v.split(',') {
                    let &(n, _) = PACKS.iter().find(|(n, _)| *n == name)?;
                    if p.contains(&n) {
                        return None;
                    }
                    p.push(n);
                }
                packs = Some(p);
            }
            "--threads" => threads = v.parse().ok().filter(|&t| t <= 256)?,
            _ => return None,
        }
        rest = tail;
    }
    if !rest.is_empty() || dir.is_empty() || dir.starts_with("--") {
        return None;
    }
    Some((dir.clone(), packs?, threads))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("--netcheck") if args.len() == 1 => netcheck(),
        Some("--worker") => match cli(&args[1..]) {
            Some((dir, packs, threads)) => {
                if threads > 0 {
                    // SAFETY: no other thread exists yet (the engine starts its threads on first use)
                    unsafe { std::env::set_var("TINYASR_THREADS", threads.to_string()) };
                }
                worker(&dir, &packs)
            }
            None => usage(),
        },
        _ => usage(),
    };
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(s: &str) -> J {
        parse(s.as_bytes()).unwrap()
    }

    #[test]
    fn json_parses_requests() {
        let r = obj(r#" {"n": 32000, "lang": "pieces", "langs": ["uk", "en"], "gate_db": 0, "x": [1, {"y": null}]}"#);
        assert_eq!(r.get("n").and_then(J::num), Some(32000.0));
        assert_eq!(r.get("lang").and_then(J::str), Some("pieces"));
        assert_eq!(
            r.get("langs"),
            Some(&J::Arr(vec![J::Str("uk".into()), J::Str("en".into())]))
        );
        assert_eq!(parse(br#""\u0443\ud83d\ude00\n""#), Some(J::Str("у😀\n".into())));
        assert_eq!(parse(b"-1.5e2"), Some(J::Num(-150.0)));
        assert_eq!(obj(r#"{"n": 1, "n": 2}"#).get("n"), Some(&J::Num(2.0))); // the last one, as Python
    }

    #[test]
    fn json_rejects_garbage() {
        for bad in [
            &b""[..],
            b"{",
            b"{\"n\": }",
            b"[1,]",
            b"01",
            b"1.",
            b"\"\\ud800\"",
            b"\"\x01\"",
            b"nul",
            b"{} x",
            b"1e999",
            b"\"\\x\"",
            b"\xff",
            b"{\"a\" 1}",
            b"[\"\\u12\"]",
            b"\"\xc3\"",
            b"{\"\xff\": 1}",
            b"NaN",
            b"+1",
        ] {
            assert_eq!(parse(bad), None, "{:?}", String::from_utf8_lossy(bad));
        }
        let deep = "[".repeat(100_000) + &"]".repeat(100_000);
        assert_eq!(parse(deep.as_bytes()), None);
        let deep = format!("{{\"n\": 2, \"x\": {}1{}}}", "[".repeat(40), "]".repeat(40));
        assert_eq!(parse(deep.as_bytes()), None);
    }

    #[test]
    fn json_writes_strings_and_numbers() {
        assert_eq!(js("a\"b\\c\nd\u{1}é\t\u{7f}"), "\"a\\\"b\\\\c\\nd\\u0001é\\t\\u007f\"");
        assert_eq!(jn(f64::NAN, 2), "-1000000000");
        assert_eq!(jn(f64::NEG_INFINITY, 4), "-1000000000");
        assert_eq!(jn(-1.23456, 3), "-1.235");
        assert_eq!(jn(-0.0001, 2), "0");
        assert_eq!(jn(1e300, 4), "1e300");
        assert_eq!(jn(-1.7e308, 4), "-1.7e308");
        for s in ["x\u{2028}y\"", "\\\"\n\r\t\u{0}\u{1f}", "уї 😀 ok"] {
            assert_eq!(parse(js(s).as_bytes()), Some(J::Str(s.into())));
        }
    }

    #[test]
    fn requests_are_checked() {
        assert_eq!(head(b"{\"n\": 32000, \"lang\": \"pieces\"}\n").map(|r| r.1), Ok(32000));
        assert_eq!(
            head(format!("{{\"n\": {MAX_BYTES}}}").as_bytes()).map(|r| r.1),
            Ok(MAX_BYTES)
        );
        assert!(head(format!("{{\"n\": {}}}", MAX_BYTES + 2).as_bytes()).is_err());
        for bad in [
            &b"{\"n\": 0}"[..],
            b"{\"n\": 3}",
            b"{\"n\": -2}",
            b"{\"n\": 2.5}",
            b"{\"n\": 1e300}",
            b"{\"n\": \"2\"}",
            b"{}",
            b"[2]",
            b"2",
            b"hello",
            b"{\"n\": 2",
            b"\xff\xfe",
            b"{\"n\": 115200002}",
            b"{\"n\": null}",
        ] {
            assert!(head(bad).is_err(), "{:?}", String::from_utf8_lossy(bad));
        }
    }

    fn all() -> Vec<(&'static str, &'static [Lang])> {
        PACKS.to_vec()
    }

    #[test]
    fn jobs_are_pieces_only() {
        let two: Vec<(&str, &[Lang])> = vec![PACKS[1], PACKS[0]];
        let j = job(
            &obj(r#"{"n": 2, "lang": "pieces", "langs": ["uk", "de", "en", "uk", "ru"]}"#),
            &two,
        )
        .unwrap();
        assert_eq!(
            j,
            Job {
                langs: vec![Lang::Uk, Lang::En, Lang::Ru],
                gate_db: 0.0
            }
        );
        // no langs: every loaded language, in the order of priority
        let j = job(&obj(r#"{"n": 2, "lang": "pieces", "langs": []}"#), &two).unwrap();
        assert_eq!(
            j.langs,
            vec![Lang::En, Lang::Ru, Lang::Uk, Lang::Es, Lang::It, Lang::Pt]
        );
        // gate_db as Python's float(req.get("gate_db") or 0)
        for (v, want) in [
            ("18", 18.0),
            ("\"12\"", 12.0),
            ("true", 1.0),
            ("[]", 0.0),
            ("null", 0.0),
            ("\"\"", 0.0),
            ("-5", -5.0),
        ] {
            let j = job(&obj(&format!(r#"{{"lang": "pieces", "gate_db": {v}}}"#)), &two).unwrap();
            assert_eq!(j.gate_db, want, "{v}");
        }
        assert!(
            job(&obj(r#"{"lang": "pieces", "gate_db": "nan"}"#), &two)
                .unwrap()
                .gate_db
                .is_nan()
        );
        // only "pieces": parakeet.py's "auto" and single-language requests are errors
        for bad in [
            r#"{}"#,
            r#"{"lang": "auto", "langs": ["uk"]}"#,
            r#"{"lang": "uk", "langs": ["uk"]}"#,
            r#"{"lang": "es"}"#,
            r#"{"lang": 5}"#,
            r#"{"lang": null}"#,
            r#"{"lang": ""}"#,
            r#"{"lang": "Pieces"}"#,
            r#"{"lang": ["pieces"]}"#,
        ] {
            assert_eq!(job(&obj(bad), &two), Err(String::from("ValueError: lang")), "{bad}");
        }
        for bad in [
            r#"{"gate_db": "x"}"#,
            r#"{"gate_db": [1]}"#,
            r#"{"gate_db": {"a": 1}}"#,
            r#"{"langs": ["xx", "de"]}"#,
            r#"{"langs": ["uk", 5]}"#,
            r#"{"langs": "uk"}"#,
            r#"{"langs": {"uk": 1}}"#,
            r#"{"langs": ["fa"]}"#,
        ] {
            let bad = bad.replacen('{', r#"{"lang": "pieces", "#, 1);
            assert!(job(&obj(&bad), &two).is_err(), "{bad}");
        }
        assert!(job(&obj(r#"{"lang": "pieces", "langs": ["fa"]}"#), &all()).is_ok());
    }

    #[test]
    fn packs_by_priority() {
        let all = all();
        let name = |l: Lang, ld: &Loaded<'static>| -> Option<&'static str> { pack_for(l, ld).map(|i| ld[i].0) };
        assert_eq!(name(Lang::En, &all), Some("en-ru-uk"));
        assert_eq!(name(Lang::Ru, &all), Some("en-ru-uk"));
        assert_eq!(name(Lang::Uz, &all), Some("en-ru-uz"));
        assert_eq!(name(Lang::Pt, &all), Some("en-es-it-pt"));
        assert_eq!(name(Lang::Km, &all), Some("en-km"));
        // the order of --packs does not matter
        let rev: Vec<_> = all.iter().rev().copied().collect();
        assert_eq!(name(Lang::En, &rev), Some("en-ru-uk"));
        assert_eq!(name(Lang::Ru, &rev), Some("en-ru-uk"));
        let small: Vec<(&str, &[Lang])> = vec![PACKS[4], PACKS[3], PACKS[2]];
        assert_eq!(name(Lang::En, &small), Some("en-ru-uz"));
        assert_eq!(name(Lang::Uk, &small), None);
        let g = groups(&[Lang::Uk, Lang::Es, Lang::En, Lang::Fa, Lang::Km], &rev);
        assert_eq!(
            g,
            vec![
                (4, vec![Lang::Uk, Lang::En]),
                (3, vec![Lang::Es]),
                (1, vec![Lang::Fa]),
                (0, vec![Lang::Km])
            ]
        );
    }

    fn cand(l: Lang, t: &str, s: f64) -> Cand {
        Cand {
            lang: l,
            text: t.into(),
            score: s,
        }
    }

    #[test]
    fn replies_have_the_protocol_shape() {
        let r = reply(
            &[
                cand(Lang::Uk, "при\"віт\n", -12.345678),
                cand(Lang::En, "", f64::NEG_INFINITY),
            ],
            0.0,
            0.5,
        );
        assert!(!r.contains('\n'));
        let v = obj(&r);
        let J::Obj(c) = v.get("cands").unwrap() else {
            panic!("{r}")
        };
        assert_eq!(
            c[0],
            ("uk".into(), J::Arr(vec![J::Str("при\"віт\n".into()), J::Num(-12.3457)]))
        );
        assert_eq!(c[1], ("en".into(), J::Arr(vec![J::Str("".into()), J::Num(BAD_NUMBER)])));
        assert_eq!(v.get("main").and_then(J::str), Some("uk"));
        assert_eq!(v.get("note").and_then(J::str), Some(""));
        assert_eq!(v.get("asr_s").and_then(J::num), Some(0.5));
        let r = reply(&[cand(Lang::Uk, "a", -1.0)], 0.25, 0.0);
        assert_eq!(obj(&r).get("note").and_then(J::str), Some(", quieter voices muted 25%"));
        assert_eq!(reply(&[], 0.0, 0.0), error("RuntimeError: no candidate"));
    }

    #[test]
    fn netcheck_wants_loopback_only() {
        let head = "Inter-|   Receive\n face |bytes    packets\n";
        assert!(lo_only(&format!("{head}    lo:       0       0\n")));
        assert!(!lo_only(&format!("{head}    lo:       0       0\n  eth0: 12 3\n")));
        assert!(!lo_only(&format!("{head}wlp2s0: 1 2\n")));
    }

    #[test]
    fn scores_combine_path_and_tokens() {
        let sc = |tokens, token_logp, path_logp, seconds| Score {
            tokens,
            token_logp,
            path_logp,
            seconds,
        };
        // normal: path -2/s, mean token -0.5: (-2 + 0.5 * -0.5) * 10 s
        assert!((cand_score(&sc(20, -10.0, -20.0, 10.0), 10.0) - -22.5).abs() < 1e-9);
        // empty (no tokens written, a likely path of blanks): the mean counts as -5
        assert!((cand_score(&sc(0, 0.0, -1.0, 10.0), 10.0) - -26.0).abs() < 1e-9);
        // a very unsure mean is floored at -5
        assert!((cand_score(&sc(2, -40.0, -10.0, 10.0), 10.0) - -35.0).abs() < 1e-9);
        // nothing decoded (seconds 0): -50 per second
        assert!((cand_score(&sc(0, 0.0, 0.0, 0.0), 2.0) - -105.0).abs() < 1e-9);
        // the English reading of Ukrainian speech that wrote almost nothing now loses
        assert!(cand_score(&sc(3, -6.0, -1.0, 10.0), 10.0) < cand_score(&sc(60, -6.0, -8.0, 10.0), 10.0));
        for s in [
            sc(1, f32::NAN, f32::INFINITY, 1.0),
            sc(0, 0.0, f32::NEG_INFINITY, 1e-30),
            sc(5, -1.0, -1.0, 1.0),
        ] {
            assert!(cand_score(&s, 3.0).is_finite() && cand_score(&s, f64::NAN).is_finite());
        }
        assert_eq!(cand_score(&sc(5, -1.0, -1.0, 1.0), 0.0), 0.0);
    }

    #[test]
    fn gate_mutes_quiet_voices() {
        // 1 s loud tone, 1 s quiet tone (40 dB lower), 1 s loud
        let tone = |amp: f32, n: usize| (0..n).map(move |i| amp * (i as f32 * 0.1).sin());
        let mut a: Vec<f32> = tone(0.5, SR).chain(tone(0.005, SR)).chain(tone(0.5, SR)).collect();
        let orig = a.clone();
        let muted = gate(&mut a, 18.0);
        assert!(muted > 0.2 && muted < 0.34, "{muted}");
        assert!(a[SR + SR / 2].abs() < 1e-9);
        assert!(a[SR / 2..SR / 2 + 100].iter().any(|x| x.abs() > 0.1));
        let mut b = a.clone();
        assert_eq!(gate(&mut b, 0.0), 0.0);
        assert_eq!(a, b);
        assert_eq!(gate(&mut [0.1; 100], 18.0), 0.0);
        // as numpy: -inf below everything keeps all, NaN compares false and mutes all, a negative value is off
        let mut b = orig.clone();
        assert_eq!(gate(&mut b, f64::INFINITY), 0.0);
        assert_eq!(b[SR / 2..SR * 5 / 2], orig[SR / 2..SR * 5 / 2]); // the 10 ms ramps still fade the two ends
        assert_eq!(gate(&mut orig.clone(), -3.0), 0.0);
        let mut b = orig.clone();
        assert_eq!(gate(&mut b, f64::NAN), 1.0);
        assert!(b.iter().all(|x| *x == 0.0));
        let mut z = vec![0.0f32; SR];
        assert!(gate(&mut z, 18.0).is_finite());
    }

    #[test]
    fn read_line_limits() {
        let mut c = io::Cursor::new(b"{}\nxx".to_vec());
        assert_eq!(read_line(&mut c), Ok(Some(b"{}\n".to_vec())));
        assert_eq!(read_line(&mut c), Ok(None));
        let mut long = io::Cursor::new(vec![b'x'; MAX_LINE as usize + 10]);
        assert_eq!(read_line(&mut long), Err(()));
    }

    #[test]
    fn command_line() {
        let a = |s: &str| s.split(' ').map(String::from).collect::<Vec<_>>();
        assert_eq!(
            cli(&a("/m --packs en-ru-uk,en-fa --threads 4")),
            Some(("/m".into(), vec!["en-ru-uk", "en-fa"], 4))
        );
        assert_eq!(
            cli(&a("/m --threads 2 --packs en-km")),
            Some(("/m".into(), vec!["en-km"], 2))
        );
        for bad in [
            "/m",
            "/m --packs",
            "/m --packs en-de",
            "/m --packs en-fa,en-fa",
            "/m --packs en-fa --threads x",
            "/m --packs en-fa --threads 999",
            "--packs en-fa",
            "/m --packs en-fa,",
            "/m --packs ../x",
            "/m --packs en-fa --x 1",
        ] {
            assert_eq!(cli(&a(bad)), None, "{bad}");
        }
        assert_eq!(cli(&[]), None);
    }
}
