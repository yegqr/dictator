//! A finished recording: only the full-context pass, piece by piece.
//!
//! Silero VAD marks speech over the whole file first. Then each piece runs from the end of the last one to the end of the sentence: a pause where
//! the end-of-turn head says the speaker has finished (asked at 0.24 s into each pause, as live), or 3 s of silence; at
//! most 20 s. It is decoded with the live session's pause compression and left context; a piece cut at 20 s keeps
//! its words up to the last . ? ! (else up to its longest pause) and the rest is decoded again with the next piece,
//! as the live session does.
//!
//! Without the end-of-turn head ([`Transcription::with_eou`] false, or TINYASR_NO_EOU=1): all pieces are planned from
//! the VAD flags first (a piece ends at a pause of 0.48 s once it holds 4 s, at 3 s of silence, or at 20 s in the middle
//! of its longest pause), so no audio is encoded twice and the pieces are independent: they are decoded in parallel,
//! one per thread (single-threaded kernels inside), and given out in order.

use alloc::format;
use alloc::vec::Vec;

use crate::model::Token;

use crate::session::{CAP, FR, SR, compress, longest_pause};
use crate::text::{W, balance_quotes, cap, end_mark, is_end, show_quotes, spans, trim_trail, units, words};
use crate::vad::CHUNK;
use crate::{Lang, Models, Score, Word, allowed};

/// Frames of silence that end a sentence.
const LONG: usize = 75;
/// Without the head: a pause of this many frames (0.48 s) ends a piece that holds at least PIECE_MIN frames (4 s).
const PAUSE_CUT: usize = 12;
const PIECE_MIN: usize = 100;
/// A piece that reaches 20 s is cut in the middle of its longest pause of at least this many frames after its first
/// FORCE_FROM frames (else right at 20 s).
const FORCE_PAUSE: usize = 3;
const FORCE_FROM: usize = 125;
/// Left context of a planned piece that continues the previous one: the last CTX_VOICED speech frames before it (1.2 s
/// of the previous piece's words, not just the pause the cut sits in), reaching back at most CTX_MAX frames. Its words
/// are dropped, but the model hears what came before: its capitals at the start are in context, and the punctuation it
/// writes after the previous piece's last word (which that piece decoded without hearing what follows) replaces that
/// piece's own guess.
const CTX_VOICED: usize = 30;
const CTX_MAX: usize = 150;

/// A planned piece (no-head mode): frames [open0, b) with `ctx` frames of left context, its first speech frame, whether
/// 3 s of silence ended it.
#[derive(Clone, Copy)]
struct Plan {
    open0: usize,
    ctx: usize,
    first: usize,
    b: usize,
    end: bool,
}

/// A decoded planned piece: its words (original timeline) and the language picked for it.
#[derive(Default)]
struct Decoded {
    ws: Vec<W>,
    lang: Option<Lang>,
    end: bool,
    /// The last word of the left context as this piece's decode wrote it (its punctuation).
    ctx_last: Option<alloc::string::String>,
    score: Score,
}

/// One language's reading of a recording, from [`Transcription::candidates`].
#[derive(Clone, Debug)]
pub struct Candidate {
    pub lang: Lang,
    pub words: Vec<Word>,
    /// The words joined (a space before each word without `join`).
    pub text: alloc::string::String,
    pub score: Score,
}

/// The score of one decoded piece: its tokens from frame `from` on (not the left context's), its best path, its length.
fn piece_score(tk: &[Token], from: usize, path: Option<f32>, samples: usize) -> Score {
    let kept = tk.iter().filter(|t| t.frame as usize >= from);
    let (n, lp) = kept.fold((0, 0.0), |(n, s), t| (n + 1, s + t.logp));
    Score { tokens: n, token_logp: lp, path_logp: path.unwrap_or(0.0), seconds: samples as f32 / SR as f32 }
}

/// Recognition of a whole recording; an iterator over its pieces' final words, in order.
pub struct Transcription<'m> {
    m: &'m Models,
    audio: Vec<f32>, // whole frames
    vf: Vec<bool>,
    lang: Lang,
    langs: Vec<Lang>, // the recognizer picks among them for each piece
    beam: usize,
    open0: usize, // frame where the next piece starts
    ctx: usize,   // frames of left context for it
    ended: bool,
    last_end: f64, // end of the last word given out (s)
    eou: bool,     // ask the end-of-turn head at pauses; else planned pieces, decoded in parallel
    decoded: Option<Vec<Decoded>>,
    next_piece: usize,
    score: Score, // of the pieces given out
}

impl<'m> Transcription<'m> {
    /// `audio`: 16 kHz. `langs`: as for [`crate::Session::new`] (with several, the recognizer picks for each piece).
    /// `beam`: beam width of the decoding ([`crate::DEFAULT_BEAM`]; 1: greedy, fastest).
    pub fn new(m: &'m Models, audio: &[f32], langs: &[Lang], beam: usize) -> Transcription<'m> {
        Transcription::with_eou(m, audio, langs, beam, crate::eou_default())
    }

    /// As [`Transcription::new`]; `eou` false: no end-of-turn head (no second encoder pass per pause, see the module
    /// notes): for a recording whose end the user marked. Pieces are decoded in parallel at the first `next`.
    pub fn with_eou(m: &'m Models, audio: &[f32], langs: &[Lang], beam: usize, eou: bool) -> Transcription<'m> {
        let mut a = audio.to_vec();
        a.resize(a.len().div_ceil(FR as usize) * FR as usize, 0.0);
        // quiet stretches (a soft speaker, a far microphone) brought up to a normal level: the VAD misses speech
        // 30-40 dB below the rest, and its frames would be cut out as silence (TINYASR_AGC=0: off)
        let boosted = if agc_on() { level(&a) } else { None };
        let vin: &[f32] = boosted.as_deref().unwrap_or(&a);
        // speech flags per 40 ms frame, from p > 0.5 until p < 0.35 (as live)
        let mut vf = Vec::with_capacity(a.len() / FR as usize);
        // all chunk probabilities at once (in parallel, same values as chunk by chunk), then the flags in order
        let probs = m.vad.probs(vin);
        let (mut on, mut done, mut ci) = (false, 0, 0);
        for f in 0..a.len() / FR as usize {
            let end = (f + 1) * FR as usize;
            while done + CHUNK <= end {
                let p = probs[ci];
                ci += 1;
                on = p > 0.5 || (on && p > 0.35);
                done += CHUNK;
            }
            vf.push(on);
        }
        let a = boosted.unwrap_or(a);
        let langs = allowed(m, langs);
        Transcription { m, audio: a, vf, lang: langs[0], langs, beam: beam.max(1), open0: 0, ctx: 0, ended: true, last_end: 0.0, eou, decoded: None, next_piece: 0, score: Score::default() }
    }

    /// The language (`Model`: of the last piece).
    pub fn lang(&self) -> Lang {
        self.lang
    }

    /// How sure the recognizer was of the pieces given out so far (all of them once `next` returned None); with
    /// several languages, of the language picked for each piece. Pieces cut at 20 s count their re-decoded words twice.
    pub fn score(&self) -> Score {
        self.score
    }

    /// The whole recording read in each of `langs` that the model has (no others; none of them: empty), one full
    /// transcription per language (no end-of-turn head: planned pieces, decoded in parallel; the VAD runs once), each
    /// with its [`Score`]: so a host can pick the language itself, or between models (call this on each model and
    /// compare the candidates' scores; `Score::path_per_second` with beam > 1 compares best across models).
    /// Costs one transcription per language.
    pub fn candidates(m: &'m Models, audio: &[f32], langs: &[Lang], beam: usize) -> Vec<Candidate> {
        Transcription::candidates_after(m, audio, 0, langs, beam)
    }

    /// As [`Transcription::candidates`], for a piece of a longer recording: the first `context` samples of `audio`
    /// are the recording just before the piece, heard but not written (words from the piece on; times still count
    /// from the start of `audio`). Without it the model starts cold in the middle of speech and can skip the piece's
    /// first words, often a whole sentence; the voice detector also starts cold and can miss quiet words there. As
    /// for a piece of [`Transcription::new`], the context used is the last 1.2 s of speech before the piece (within
    /// 6 s).
    pub fn candidates_after(m: &'m Models, audio: &[f32], context: usize, langs: &[Lang], beam: usize) -> Vec<Candidate> {
        let own: Vec<Lang> = m.langs().iter().copied().filter(|l| langs.contains(l)).collect();
        let Some(&first) = own.first() else { return Vec::new() };
        let base = Transcription::with_eou(m, audio, &[first], beam, false);
        let open0 = (context / FR as usize).min(base.vf.len());
        let ctx = context_frames(&base.vf, open0);
        let last_end = (open0 as i64 * FR) as f64 / SR as f64;
        own.into_iter()
            .map(|lang| {
                let mut t = Transcription { audio: base.audio.clone(), vf: base.vf.clone(), lang, langs: alloc::vec![lang], open0, ctx, last_end, ..Transcription::empty(m, beam) };
                let mut words = Vec::new();
                for ws in &mut t {
                    words.extend(ws);
                }
                let mut text = alloc::string::String::new();
                for w in &words {
                    if !text.is_empty() && !w.join {
                        text.push(' ');
                    }
                    text.push_str(&w.text);
                }
                Candidate { lang, words, text, score: t.score }
            })
            .collect()
    }

    /// No audio yet, no-head mode (for `candidates`).
    fn empty(m: &'m Models, beam: usize) -> Transcription<'m> {
        Transcription { m, audio: Vec::new(), vf: Vec::new(), lang: Lang::En, langs: Vec::new(), beam: beam.max(1), open0: 0, ctx: 0, ended: true, last_end: 0.0, eou: false, decoded: None, next_piece: 0, score: Score::default() }
    }

    /// Audio length (s).
    pub fn duration(&self) -> f32 {
        self.audio.len() as f32 / SR as f32
    }

    /// Where the next piece starts (s): progress.
    pub fn position(&self) -> f32 {
        (self.open0 as i64 * FR) as f32 / SR as f32
    }

    /// No-head mode: every piece, from the VAD flags alone.
    fn plan(&self) -> Vec<Plan> {
        let (vf, n) = (&self.vf, self.vf.len());
        let mut out = Vec::new();
        let (mut open0, mut ctx) = (self.open0, self.ctx);
        while let Some(first) = (open0..n).find(|&i| vf[i]) {
            let cap_end = first + (CAP / FR) as usize;
            let (mut b, mut quiet, mut at_pause) = (first, 0, false);
            while b < n && b < cap_end && quiet < LONG {
                quiet = if vf[b] { 0 } else { quiet + 1 };
                b += 1;
                if quiet == PAUSE_CUT && b - first >= PIECE_MIN {
                    at_pause = true;
                    break;
                }
            }
            if !at_pause && quiet < LONG && b >= cap_end && b < n {
                // forced: the middle of the longest pause after the first FORCE_FROM frames
                let lo = (first + FORCE_FROM).min(b);
                let (mut best, mut run0, mut best_len) = (0, lo, 0);
                for i in lo..b {
                    if vf[i] {
                        run0 = i + 1;
                    } else if i + 1 - run0 > best_len {
                        best_len = i + 1 - run0;
                        best = run0 + best_len / 2;
                    }
                }
                if best_len >= FORCE_PAUSE {
                    b = best;
                }
            }
            let end = quiet >= LONG;
            out.push(Plan { open0, ctx, first, b, end });
            open0 = b;
            ctx = if end { 0 } else { context_frames(vf, b) };
        }
        out
    }

    /// No-head mode: one planned piece decoded with its left context. If the model writes nothing, it is decoded again
    /// without the left context, then (if the piece has speech after its first voiced run: a noise burst at the start
    /// can make it skip the whole piece) from the next voiced run.
    fn decode(&self, p: Plan) -> Decoded {
        let (vf, mut open0, mut ctx) = (&self.vf, p.open0, p.ctx);
        let mut from = p.first;
        loop {
            let a = open0.saturating_sub(ctx);
            let (x, map) = compress(&self.audio[a * FR as usize..p.b * FR as usize], &vf[a..p.b], a as i64);
            let (tk, picked, path) = self.m.full_scored(&x, &self.langs, self.beam);
            if tk.is_empty() {
                // a stateless RNN-T pack (en-ru-uz) can write nothing at all for a short piece that opens with the
                // previous piece's last words as its left context: decode it once more on its own
                if ctx > 0 {
                    ctx = 0;
                    continue;
                }
                // the next voiced run after the first one, if 1 s of speech follows it
                let r1 = (from..p.b).find(|&i| !vf[i]).unwrap_or(p.b);
                if let Some(s2) = (r1..p.b).find(|&i| vf[i]) {
                    if vf[s2..p.b].iter().filter(|v| **v).count() >= 25 {
                        open0 = s2;
                        from = s2;
                        ctx = 0;
                        continue;
                    }
                }
                return Decoded { end: p.end, score: piece_score(&[], 0, path, x.len()), ..Default::default() };
            }
            let tk: Vec<Token> = tk.iter().map(|t| Token { frame: map[(t.frame as usize).min(map.len() - 1)] as u32, ..*t }).collect();
            let score = piece_score(&tk, open0, path, x.len());
            let o0 = (open0 as i64 * FR) as f64 / SR as f64 - 0.02;
            let (ws, cw): (Vec<W>, Vec<W>) = words(&self.m.asr.pieces, &tk, 0.0).into_iter().partition(|w| w.t >= o0);
            let ctx_last = if ctx > 0 && open0 == p.open0 { cw.last().map(|w| w.w.clone()) } else { None };
            return Decoded { ws, lang: picked, end: p.end, ctx_last, score };
        }
    }

    /// No-head mode: all pieces decoded, longest first, in parallel (one piece per thread).
    fn decode_all(&self) -> Vec<Decoded> {
        let plan = self.plan();
        let mut order: Vec<usize> = (0..plan.len()).collect();
        order.sort_by_key(|&i| core::cmp::Reverse(plan[i].b - plan[i].open0));
        #[cfg(feature = "std")]
        {
            let slots: Vec<std::sync::Mutex<Option<Decoded>>> = plan.iter().map(|_| std::sync::Mutex::new(None)).collect();
            crate::pool::par_for(order.len(), &|i| {
                let j = order[i];
                let d = self.decode(plan[j]);
                *slots[j].lock().unwrap() = Some(d);
            });
            slots.into_iter().map(|s| s.into_inner().unwrap().unwrap()).collect()
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = order;
            plan.iter().map(|&p| self.decode(p)).collect()
        }
    }

    /// No-head mode: the next planned piece's words.
    fn next_planned(&mut self) -> Option<Vec<Word>> {
        if self.decoded.is_none() {
            self.decoded = Some(self.decode_all());
        }
        loop {
            let i = self.next_piece;
            let Decoded { mut ws, lang: picked, end, score, .. } = self.decoded.as_mut()?.get_mut(i).map(core::mem::take)?;
            self.next_piece += 1;
            self.score.add(&score);
            // the next piece heard this one's end as its left context: its punctuation after our last word wins
            // (also Persian ، ؛ ؟ and Khmer ។ ៕, which those models write instead of the Latin marks)
            let punct = |c: char| matches!(c, '.' | '?' | '!' | ',' | ';' | ':' | '…' | '،' | '؛' | '؟' | '។' | '៕');
            let mark = self.decoded.as_ref().and_then(|d| d.get(i + 1)).and_then(|d| d.ctx_last.as_deref()).map(|w| {
                let t = w.trim_end_matches(punct);
                alloc::string::String::from(&w[t.len()..])
            });
            if let (Some(mark), Some(last)) = (mark, ws.last_mut()) {
                // the next piece heard what follows this word: its punctuation (also none) replaces our guess
                let t = last.w.trim_end_matches(punct).len();
                if t > 0 {
                    last.w.truncate(t);
                    last.w.push_str(&mark);
                }
            }
            // the next piece starts with a capital (not an acronym) after a word without an end mark: the model heard a
            // new sentence there, so this one ends ("чого Хоча" -> "чого. Хоча")
            let next_cap = self.decoded.as_ref().and_then(|d| d[i + 1..].iter().find(|d| !d.ws.is_empty())).is_some_and(|d| {
                let w = &d.ws[0].w;
                let letters: Vec<char> = w.chars().filter(|c| c.is_alphabetic()).collect();
                letters.first().is_some_and(|c| c.is_uppercase()) && !(letters.len() > 1 && letters.iter().all(|c| c.is_uppercase()))
            });
            if let Some(last) = ws.last_mut() {
                if next_cap && !is_end(&last.w) && last.w.chars().last().is_some_and(|c| c.is_alphanumeric()) {
                    last.w = format!("{}{}", last.w, end_mark(self.lang));
                }
            }
            self.open0 = self.vf.len().min(self.open0.max(ws.last().map_or(0, |w| (w.te / 0.04) as usize)));
            if let Some(l) = picked {
                self.lang = l;
            }
            if ws.is_empty() {
                continue;
            }
            if end {
                if let Some(last) = ws.last_mut() {
                    if !is_end(&last.w) {
                        last.w = format!("{}{}", trim_trail(&last.w), end_mark(self.lang));
                    }
                }
            }
            let ended = is_end(&ws[ws.len() - 1].w);
            if self.ended {
                ws[0].w = cap(&ws[0].w);
            }
            let out = self.out(&balance_quotes(ws));
            self.ended = ended;
            return Some(out);
        }
    }

    /// The words out, timed by [`spans`] from the last word given out.
    fn out(&mut self, ws: &[W]) -> Vec<Word> {
        let vf = &self.vf;
        let ws = if self.lang == Lang::Km { units(ws.to_vec()) } else { ws.to_vec() }; // Khmer: timed syllables
        let sp = spans(&ws, self.last_end, |f| usize::try_from(f).ok().and_then(|i| vf.get(i).copied()));
        if let Some(&(_, e)) = sp.last() {
            self.last_end = e as f64;
        }
        ws.into_iter().zip(sp).map(|(w, (start, end))| Word { text: show_quotes(w.w, self.lang), start, end, join: w.join }).collect()
    }
}

impl Iterator for Transcription<'_> {
    type Item = Vec<Word>;

    fn next(&mut self) -> Option<Vec<Word>> {
        if !self.eou {
            return self.next_planned();
        }
        let n = self.vf.len();
        loop {
            let first = (self.open0..n).find(|&i| self.vf[i])?;
            // the piece: up to 3 s of silence, at most CAP of audio from its first speech
            let cap_end = first + (CAP / FR) as usize;
            let (mut b, mut quiet, mut eou) = (first, 0, false);
            while b < n && b < cap_end && quiet < LONG {
                quiet = if self.vf[b] { 0 } else { quiet + 1 };
                b += 1;
                // 0.24 s into a pause: has the speaker finished? (the turn so far, <= 30 s; ended at 0.48 s of silence)
                if quiet == 6 && self.m.asr.has_head() && self.vf[b..(b + 6).min(n)].iter().all(|v| !v) {
                    let from = self.open0.max(b.saturating_sub(750));
                    if self.m.eou(&self.audio[from * FR as usize..b * FR as usize], self.lang).is_some_and(|p| p > 0.5) {
                        b = (b + 6).min(n);
                        eou = true;
                        break;
                    }
                }
            }
            let end = eou || quiet >= LONG; // the sentence is over: a dot if the model wrote none
            let full = !end && b >= cap_end && b < n; // cut short: more speech may follow
            let a = self.open0.saturating_sub(self.ctx);
            let (x, map) = compress(&self.audio[a * FR as usize..b * FR as usize], &self.vf[a..b], a as i64);
            let (tk, picked, path) = self.m.full_scored(&x, &self.langs, self.beam);
            if let Some(l) = picked {
                self.lang = l;
            }
            let tk: Vec<Token> = tk.iter().map(|t| Token { frame: map[(t.frame as usize).min(map.len() - 1)] as u32, ..*t }).collect();
            self.score.add(&piece_score(&tk, self.open0, path, x.len()));
            // the left context's own words are dropped
            let o0 = (self.open0 as i64 * FR) as f64 / SR as f64 - 0.02;
            let mut ws: Vec<W> = words(&self.m.asr.pieces, &tk, 0.0).into_iter().filter(|w| w.t >= o0).collect();
            let (keep, next, ctx) = if !full || ws.len() < 2 {
                // a sentence end or the end of the audio
                if end {
                    if let Some(last) = ws.last_mut() {
                        if !is_end(&last.w) {
                            last.w = format!("{}{}", trim_trail(&last.w), end_mark(self.lang));
                        }
                    }
                }
                (ws.len(), b, 0)
            } else if let Some(j) = (0..ws.len() - 1).rev().find(|&j| is_end(&ws[j].w)) {
                (j + 1, cut(&ws, j), (SR / FR) as usize)
            } else {
                let j = longest_pause(&ws).unwrap_or(ws.len() - 2);
                (j + 1, cut(&ws, j), (4 * SR / FR) as usize)
            };
            let next = next.max(self.open0 + 1);
            ws.truncate(keep);
            self.open0 = next;
            self.ctx = ctx;
            if ws.is_empty() {
                continue;
            }
            let ended = is_end(&ws[ws.len() - 1].w);
            if self.ended {
                ws[0].w = cap(&ws[0].w);
            }
            let out = self.out(&balance_quotes(ws));
            self.ended = ended;
            return Some(out);
        }
    }
}

/// Frames of left context for a piece that starts at frame `b`: back from it until CTX_VOICED speech frames are
/// covered, at most CTX_MAX; 0 if there is no speech before it.
fn context_frames(vf: &[bool], b: usize) -> usize {
    let (mut i, mut v) = (b, 0);
    let lo = b.saturating_sub(CTX_MAX);
    while i > lo && v < CTX_VOICED {
        i -= 1;
        if vf[i] {
            v += 1;
        }
    }
    if v > 0 { b - i } else { 0 }
}

/// Frame halfway between word j and j + 1.
fn cut(ws: &[W], j: usize) -> usize {
    ((ws[j].te + ws[j + 1].t) / 2.0 / 0.04 + 0.5) as usize
}

/// Auto-gain on, unless TINYASR_AGC=0 (std builds).
pub(crate) fn agc_on() -> bool {
    #[cfg(feature = "std")]
    {
        std::env::var("TINYASR_AGC").map_or(true, |v| v != "0")
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

/// Target level of speech (dB re full scale) for the frames' 90th percentile around each frame, the most a stretch is
/// raised, and the level below which a stretch counts as silence (not raised).
pub(crate) const AGC_TARGET: f32 = -26.0;
pub(crate) const AGC_MAX: f32 = 36.0;
pub(crate) const AGC_FLOOR: f32 = -78.0;
/// Window (frames each side, 2.5 s) for the local speech level.
const AGC_WIN: usize = 62;

/// The audio with quiet stretches raised: per 40 ms frame, the local speech level (90th percentile of frame energies
/// within 2.5 s) is brought up to AGC_TARGET (never lowered, at most AGC_MAX dB, not for silence below AGC_FLOOR),
/// the gain interpolated between frame centres. None if nothing needs raising (most recordings: unchanged).
fn level(a: &[f32]) -> Option<Vec<f32>> {
    let fr = FR as usize;
    let n = a.len() / fr;
    if n == 0 {
        return None;
    }
    let db: Vec<f32> = a
        .chunks_exact(fr)
        .map(|c| {
            let e = c.iter().map(|x| x * x).sum::<f32>() / fr as f32;
            10.0 * crate::kernels::ln(e + 1e-12) / core::f32::consts::LN_10
        })
        .collect();
    let mut w = Vec::with_capacity(2 * AGC_WIN + 1);
    let gain: Vec<f32> = (0..n)
        .map(|i| {
            w.clear();
            w.extend_from_slice(&db[i.saturating_sub(AGC_WIN)..(i + AGC_WIN + 1).min(n)]);
            let k = (w.len() * 9 / 10).min(w.len() - 1);
            w.select_nth_unstable_by(k, |x, y| x.total_cmp(y));
            let lv = w[k];
            if lv < AGC_FLOOR { 0.0 } else { (AGC_TARGET - lv).clamp(0.0, AGC_MAX) }
        })
        .collect();
    if gain.iter().all(|&g| g < 1.0) {
        return None;
    }
    let lin: Vec<f32> = gain.iter().map(|g| crate::kernels::exp(g / 20.0 * core::f32::consts::LN_10)).collect();
    let mut out = a.to_vec();
    for (i, x) in out.iter_mut().enumerate() {
        // frame centres at (f + 0.5) * fr
        let pos = i as f32 / fr as f32 - 0.5;
        let f0 = (pos.max(0.0) as usize).min(n - 1);
        let f1 = (f0 + 1).min(n - 1);
        let t = (pos - f0 as f32).clamp(0.0, 1.0);
        let g = lin[f0] * (1.0 - t) + lin[f1] * t;
        *x = (*x * g).clamp(-1.0, 1.0);
    }
    Some(out)
}
