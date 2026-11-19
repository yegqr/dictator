//! Zipformer2 (icefall, causal) encoder, stateless RNN-T decoder and joiner.
//!
//! Activations are row-major `t x channels`. One set of weights runs both ways:
//! - full context (`Model::full`): the second pass over a finished segment, no attention mask, each depthwise
//!   convolution sees the whole segment;
//! - streaming (`Stream`): 32-frame chunks at 50 Hz (64 feature frames, 640 ms) with 128 frames of left context,
//!   carrying the attention keys/values, the nonlinear-attention input and the convolution tails between chunks,
//!   as icefall's `streaming_forward` and the sherpa-onnx export do.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use crate::compat::F32Ext;
use crate::fbank::{BINS, Fbank};
use crate::kernels::{Linear, Packed, Quant, Weight, exp, gemm_into, par_blocks, par_chunks, sigmoid, tanh};
use alloc::format;

// ---------------------------------------------------------------------------------------------- weights

pub struct Tensors {
    cfg: BTreeMap<String, i32>,
    t: BTreeMap<String, (Vec<usize>, Data)>,
    pub pieces: Vec<String>,
}

/// A tensor as stored: f32 values, or integer codes with one scale per `group` weights of a row (TZF2).
enum Data {
    F32(Vec<f32>),
    Int { group: usize, row: usize, scale: Vec<f32>, q: Vec<i8> }, // row = padded row length (multiple of group)
}

impl Data {
    /// f32 values (rows x k, padding dropped).
    fn f32(self, dims: &[usize]) -> Vec<f32> {
        match self {
            Data::F32(v) => v,
            Data::Int { group, row, scale, q } => {
                let n = dims[0];
                let k: usize = dims[1..].iter().product();
                let mut out = Vec::with_capacity(n * k);
                for r in 0..n {
                    for c in 0..k {
                        out.push(q[r * row + c] as f32 * scale[r * (row / group) + c / group]);
                    }
                }
                out
            }
        }
    }
}

fn f16(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // subnormal: normalize
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (s << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if e == 31 {
        (s << 31) | (0xff << 23) | (m << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

type R<T> = Result<T, &'static str>;

const TRUNCATED: &str = "model file: truncated";
const BAD: &str = "model file: corrupted tensor header";
const MISSING: &str = "model file: a tensor is missing (corrupted or wrong version?)";
const SHAPE: &str = "model file: a tensor has an unexpected shape (corrupted or wrong version?)";
const SETTING: &str = "model file: a setting is missing or out of range (corrupted or wrong version?)";

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> R<&'a [u8]> {
        let end = self.p.checked_add(n).ok_or(TRUNCATED)?;
        let s = self.b.get(self.p..end).ok_or(TRUNCATED)?;
        self.p = end;
        Ok(s)
    }
    fn align4(&mut self) {
        self.p = self.p.saturating_add((4 - self.p % 4) % 4);
    }
    fn u8(&mut self) -> R<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> R<usize> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]) as usize)
    }
    fn u32(&mut self) -> R<usize> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
    }
    fn str(&mut self) -> R<String> {
        let n = self.u16()?;
        core::str::from_utf8(self.take(n)?).map(String::from).map_err(|_| "model file: bad utf-8")
    }
}

impl Tensors {
    /// The `.tzf` file written by `export_tzf.py` (TZF1: f32) or `export_tzf2.py` (TZF2: integer weights). Every size
    /// is checked against the file, so a corrupted file is an error here, never a crash.
    pub fn parse(b: &[u8]) -> R<Tensors> {
        let mut r = Rd { b, p: 0 };
        let v2 = match r.take(4)? {
            b"TZF1" => false,
            b"TZF2" => true,
            _ => return Err("not a tinyasr model file (TZF1/TZF2)"),
        };
        let mut cfg = BTreeMap::new();
        for _ in 0..r.u32()? {
            let k = r.str()?;
            let v = r.take(4)?;
            cfg.insert(k, i32::from_le_bytes([v[0], v[1], v[2], v[3]]));
        }
        let mut t = BTreeMap::new();
        for _ in 0..r.u32()? {
            let k = r.str()?;
            let integer = v2 && r.u8()? == 1;
            let nd = r.u8()? as usize;
            let dims = (0..nd).map(|_| r.u32()).collect::<R<Vec<_>>>()?;
            let n = dims.iter().try_fold(1usize, |a, &d| a.checked_mul(d)).ok_or(BAD)?;
            if !integer {
                r.align4();
                let raw = r.take(n.checked_mul(4).ok_or(BAD)?)?;
                let v = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                t.insert(k, (dims, Data::F32(v)));
                continue;
            }
            let (bits, group, st) = (r.u8()?, r.u32()?, r.u8()?);
            if dims.is_empty() || n == 0 || group == 0 {
                return Err(BAD);
            }
            r.align4();
            let rows = dims[0];
            let row = (n / rows).div_ceil(group).checked_mul(group).ok_or(BAD)?;
            let total = rows.checked_mul(row).ok_or(BAD)?;
            let q: Vec<i8> = match bits {
                8 => r.take(total)?.iter().map(|&c| c as i8).collect(),
                4 => r.take(total.div_ceil(2))?.iter().flat_map(|&c| [(c & 15) as i8 - 8, (c >> 4) as i8 - 8]).take(total).collect(),
                _ => return Err("model file: unsupported integer width"),
            };
            r.align4();
            let ns = total / group;
            let scale = if st == 1 {
                r.take(ns.checked_mul(2).ok_or(BAD)?)?.chunks_exact(2).map(|c| f16(u16::from_le_bytes([c[0], c[1]]))).collect()
            } else {
                r.take(ns.checked_mul(4).ok_or(BAD)?)?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
            };
            t.insert(k, (dims, Data::Int { group, row, scale, q }));
        }
        let pieces = (0..r.u32()?).map(|_| r.str()).collect::<R<Vec<_>>>()?;
        Ok(Tensors { cfg, t, pieces })
    }

    /// A setting, which must lie in lo..=hi.
    fn c(&self, k: &str, lo: usize, hi: usize) -> R<usize> {
        let v = *self.cfg.get(k).ok_or(SETTING)?;
        usize::try_from(v).ok().filter(|v| (lo..=hi).contains(v)).ok_or(SETTING)
    }

    fn get(&mut self, k: &str) -> R<(Vec<usize>, Vec<f32>)> {
        let (d, v) = self.t.remove(k).ok_or(MISSING)?;
        let v = v.f32(&d); // integer tensors always have a row dimension (parse)
        Ok((d, v))
    }

    /// A tensor of exactly n values.
    fn vec(&mut self, k: &str, n: usize) -> R<Vec<f32>> {
        let v = self.get(k)?.1;
        if v.len() == n { Ok(v) } else { Err(SHAPE) }
    }

    /// A tensor of lo..=hi values.
    fn vec_in(&mut self, k: &str, lo: usize, hi: usize) -> R<Vec<f32>> {
        let v = self.get(k)?.1;
        if (lo..=hi).contains(&v.len()) { Ok(v) } else { Err(SHAPE) }
    }

    fn scalar(&mut self, k: &str) -> R<f32> {
        Ok(self.vec(k, 1)?[0])
    }

    fn has(&self, k: &str) -> bool {
        self.t.contains_key(k)
    }

    /// `k`.weight (n x kk; n None: any n >= 1) and, with `bias`, `k`.bias (n). Integer weights stay integer (`Quant`)
    /// when the groups tile the row exactly; otherwise f32.
    fn linear(&mut self, k: &str, bias: bool, n: Option<usize>, kk: usize) -> R<Linear> {
        let b = if bias { Some(alloc::format!("{k}.bias")) } else { None };
        self.linear_named(&alloc::format!("{k}.weight"), b.as_deref(), n, kk)
    }

    /// Reorders the columns of the 3x3 convolution weight `name` (cout x cin x 3 x 3: column ci*9 + a*3 + b) to
    /// tap-major order ((a*3 + b)*cin + ci), so `conv3` can build its input rows from contiguous runs of channels.
    /// Only for integer weights with one scale per row (the embed convolutions): their int8 product sums each whole
    /// row exactly in integers, so the order of the columns cannot change any result. Returns whether it did.
    fn tap_major(&mut self, name: &str, cin: usize) -> bool {
        let kk = 9 * cin;
        let Some((d, Data::Int { group, row, q, .. })) = self.t.get_mut(name) else { return false };
        let (group, row) = (*group, *row);
        if cin < 2 || group != row || row < kk || d.len() < 2 || d[1..].iter().product::<usize>() != kk {
            return false;
        }
        let n = d[0];
        if q.len() < n * row {
            return false;
        }
        let mut tmp = vec![0i8; kk];
        for r in 0..n {
            let src = &mut q[r * row..r * row + kk];
            for ci in 0..cin {
                for tap in 0..9 {
                    tmp[tap * cin + ci] = src[ci * 9 + tap];
                }
            }
            src.copy_from_slice(&tmp);
        }
        true
    }

    /// As `linear`, for tensors not named X.weight / X.bias (the LSTM's weight_ih_l0 ...).
    fn linear_named(&mut self, name: &str, bias: Option<&str>, n: Option<usize>, kk: usize) -> R<Linear> {
        let (d, data) = self.t.remove(name).ok_or(MISSING)?;
        let rows = *d.first().ok_or(SHAPE)?;
        let cols: usize = d[1..].iter().product();
        if rows == 0 || n.is_some_and(|n| n != rows) || cols != kk || kk == 0 {
            return Err(SHAPE);
        }
        let b = match bias {
            Some(k) => Some(self.vec(k, rows)?),
            None => None,
        };
        let w = match data {
            // integer weights stay integer only where they multiply in int8; otherwise they are widened once here
            // (not on every call)
            Data::Int { group, row, scale, q } if row == kk && Quant::int8_for(group) => Weight::Int(Quant::new(rows, kk, group, scale, q)),
            // one scale per row and a row length the int8 path cannot tile (the embed convolutions, k = 72 / 288): rows
            // zero-padded to a multiple of 64 so they multiply in int8; the caller feeds rows of `Linear::k()` values
            Data::Int { group, row, scale, q } if row == kk && group == kk && kk >= 64 && kk % 64 != 0 && Quant::int8_for(kk.next_multiple_of(64)) => {
                let kp = kk.next_multiple_of(64);
                let mut qp = vec![0i8; rows * kp];
                for r in 0..rows {
                    qp[r * kp..r * kp + kk].copy_from_slice(&q[r * kk..(r + 1) * kk]);
                }
                Weight::Int(Quant::new(rows, kp, kp, scale, qp))
            }
            data => Weight::F32(Packed::from_rows(&data.f32(&d), rows, kk)),
        };
        Ok(Linear { w, b })
    }
}

// ---------------------------------------------------------------------------------------------- pieces

/// SwooshL / SwooshR activations (see `kernels::swoosh`).
fn swoosh_l(x: &mut [f32]) {
    crate::kernels::swoosh(x, 4.0, 0.035);
}

fn swoosh_r(x: &mut [f32]) {
    crate::kernels::swoosh(x, 1.0, 0.313_261_7);
}

/// BiasNorm: x * exp(log_scale) / rms(x - bias), per row.
struct BiasNorm {
    bias: Vec<f32>,
    scale: f32,
}

impl BiasNorm {
    fn load(w: &mut Tensors, k: &str, dim: usize) -> R<BiasNorm> {
        Ok(BiasNorm { bias: w.vec(&alloc::format!("{k}.bias"), dim)?, scale: exp(w.scalar(&alloc::format!("{k}.log_scale"))?) })
    }
    fn apply(&self, x: &mut [f32]) {
        let d = self.bias.len();
        for r in x.chunks_exact_mut(d) {
            let ms = r.iter().zip(&self.bias).map(|(v, b)| (v - b) * (v - b)).sum::<f32>() / d as f32;
            let s = self.scale / ms.sqrt();
            r.iter_mut().for_each(|v| *v *= s);
        }
    }
}

/// orig + (x - orig) * scale, per channel, into x.
fn bypass(orig: &[f32], x: &mut [f32], scale: &[f32]) {
    let d = scale.len();
    for (xr, or) in x.chunks_exact_mut(d).zip(orig.chunks_exact(d)) {
        for ((v, o), s) in xr.iter_mut().zip(or).zip(scale) {
            *v = o + (*v - o) * s;
        }
    }
}

fn add(x: &mut [f32], y: &[f32]) {
    x.iter_mut().zip(y).for_each(|(a, b)| *a += b);
}

/// `f` of the relative position encodings for positions -before ..= after (rows of `dim`), from a table kept across
/// calls (every layer of a pass asks for the same rows).
fn pe_table(before: usize, after: usize, dim: usize, f: &dyn Fn(&[f32]) -> Vec<f32>) -> Vec<f32> {
    #[cfg(feature = "std")]
    {
        use std::sync::{Arc, Mutex};
        static T: Mutex<Vec<(usize, usize, Arc<Vec<f32>>)>> = Mutex::new(Vec::new()); // (dim, half, rows -half..=half)
        let need = before.max(after);
        let tab = {
            let mut all = T.lock().unwrap();
            match all.iter().find(|e| e.0 == dim && e.1 >= need) {
                Some(e) => e.2.clone(),
                None => {
                    let half = (need * 2).max(256);
                    let mut v = vec![0.0; (2 * half + 1) * dim];
                    for (n, row) in v.chunks_exact_mut(dim).enumerate() {
                        rel_pe(n as f32 - half as f32, dim, row);
                    }
                    let e = Arc::new(v);
                    all.retain(|x| x.0 != dim);
                    all.push((dim, half, e.clone()));
                    e
                }
            }
        };
        let half = (tab.len() / dim - 1) / 2;
        return f(&tab[(half - before) * dim..(half + after + 1) * dim]);
    }
    #[cfg(not(feature = "std"))]
    {
        let np = before + after + 1;
        let mut pe = vec![0.0; np * dim];
        for n in 0..np {
            rel_pe(n as f32 - before as f32, dim, &mut pe[n * dim..(n + 1) * dim]);
        }
        f(&pe)
    }
}

/// Compact relative positional encoding (CompactRelPositionalEncoding) of relative position v.
fn rel_pe(v: f32, dim: usize, out: &mut [f32]) {
    let cl = (dim as f32).sqrt();
    let xc = cl * v.signum() * ((v.abs() + cl).ln() - cl.ln());
    let ls = dim as f32 / (2.0 * core::f32::consts::PI);
    let a = (xc / ls).atan();
    for f in 0..dim / 2 {
        let (s, c) = (a * (f + 1) as f32).sin_cos();
        out[2 * f] = c;
        out[2 * f + 1] = s;
    }
    out[dim - 1] = 1.0;
}

struct FeedForward {
    inp: Linear,
    out: Linear,
}

impl FeedForward {
    /// x += FFN(x), then `tail(first row, rows)` on the result: the whole chain row-wise, in parallel row blocks with
    /// one pool synchronization.
    fn add_into(&self, x: &mut [f32], t: usize, tail: &(dyn Fn(usize, &mut [f32]) + Sync)) {
        let d = x.len() / t.max(1);
        let work = t * d * self.inp.n() * 2;
        par_blocks(x, d, work, &|r0, rows| {
            let nr = rows.len() / d;
            let mut h = self.inp.apply(rows, nr);
            swoosh_l(&mut h);
            let y = self.out.apply(&h, nr);
            add(rows, &y);
            tail(r0, rows);
        });
    }
}

/// Rows `keep` from the end of cat(cache, new) become the new cache; returns the concatenation.
fn cat_cache(cache: Option<&mut Vec<f32>>, new: Vec<f32>) -> Vec<f32> {
    match cache {
        None => new,
        Some(c) => {
            let keep = c.len();
            let mut all = core::mem::take(c);
            all.extend_from_slice(&new);
            *c = all[all.len() - keep..].to_vec();
            all
        }
    }
}

struct AttnWeights {
    in_proj: Linear,
    pos: Linear,
    heads: usize,
    qd: usize,
    pd: usize,
}

impl AttnWeights {
    /// Attention weights heads x t x (left + t). `keys`: cached keys (left x heads*qd), updated.
    /// `masked`: key slots to ignore (len left + t).
    fn apply(&self, x: &[f32], t: usize, keys: Option<&mut Vec<f32>>, left: usize, masked: &[bool], pos_dim: usize) -> Vec<f32> {
        let (h, qd, pd) = (self.heads, self.qd, self.pd);
        let y = self.in_proj.apply(x, t);
        let w = y.len() / t;
        let qdim = h * qd;
        let mut k = vec![0.0; t * qdim];
        for i in 0..t {
            k[i * qdim..(i + 1) * qdim].copy_from_slice(&y[i * w + qdim..i * w + 2 * qdim]);
        }
        let k = cat_cache(keys, k);
        let tk = left + t;
        // projected encodings of relative positions -(left + t - 1) ..= t - 1
        let np = left + 2 * t - 1;
        let p = pe_table(left + t - 1, t - 1, pos_dim, &|pe| self.pos.apply(pe, np));
        // per head: keys and position projections packed as gemm weights (scores = q K^T, pos = pq P^T)
        let kp: Vec<Packed> = (0..h)
            .map(|hh| {
                let mut kh = vec![0.0; tk * qd];
                for j in 0..tk {
                    kh[j * qd..(j + 1) * qd].copy_from_slice(&k[j * qdim + hh * qd..j * qdim + (hh + 1) * qd]);
                }
                Packed::from_rows(&kh, tk, qd)
            })
            .collect();
        let pw = h * pd;
        let pp: Vec<Packed> = (0..h)
            .map(|hh| {
                let mut ph = vec![0.0; np * pd];
                for n in 0..np {
                    ph[n * pd..(n + 1) * pd].copy_from_slice(&p[n * pw + hh * pd..n * pw + (hh + 1) * pd]);
                }
                Packed::from_rows(&ph, np, pd)
            })
            .collect();
        let mut out = vec![0.0; h * t * tk];
        let any_masked = masked.iter().take(tk).any(|&b| b);
        par_chunks(&mut out, tk, h * t * tk * (qd + pd + 8), &|r0, rows| {
            let nrows = rows.len() / tk;
            let mut ri = 0;
            while ri < nrows {
                let (hh, i) = ((r0 + ri) / t, (r0 + ri) % t);
                let cnt = (t - i).min(nrows - ri); // rows of this head in the chunk
                let mut qh = vec![0.0; cnt * qd];
                let mut pq = vec![0.0; cnt * pd];
                for c in 0..cnt {
                    let ii = i + c;
                    qh[c * qd..(c + 1) * qd].copy_from_slice(&y[ii * w + hh * qd..ii * w + (hh + 1) * qd]);
                    pq[c * pd..(c + 1) * pd].copy_from_slice(&y[ii * w + 2 * qdim + hh * pd..ii * w + 2 * qdim + (hh + 1) * pd]);
                }
                let seg = &mut rows[ri * tk..(ri + cnt) * tk];
                gemm_into(&qh, &kp[hh], seg);
                let mut pqs = vec![0.0; cnt * np];
                gemm_into(&pq, &pp[hh], &mut pqs);
                for (c, row) in seg.chunks_exact_mut(tk).enumerate() {
                    let ii = i + c;
                    let prow = &pqs[c * np + (t - 1 - ii)..c * np + (t - 1 - ii) + tk];
                    if any_masked {
                        for (j, (v, pv)) in row.iter_mut().zip(prow).enumerate() {
                            *v = if masked.get(j).copied().unwrap_or(false) { -1000.0 } else { *v + pv };
                        }
                    } else {
                        row.iter_mut().zip(prow).for_each(|(v, pv)| *v += pv);
                    }
                    crate::kernels::softmax_row(row);
                }
                ri += cnt;
            }
        });
        out
    }
}


/// (rows x cols) -> (cols x rows).
fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0; x.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = x[r * cols + c];
        }
    }
    t
}

/// Per head: values (tk x heads*d, head slice) packed as gemm weights with d output columns (see `weighted_rows`).
fn pack_values(v: &[f32], heads: usize, tk: usize, d: usize) -> Vec<Packed> {
    let vd = v.len() / tk;
    (0..heads)
        .map(|hh| {
            let mut vt = vec![0.0; d * tk];
            for j in 0..tk {
                for c in 0..d {
                    vt[c * tk + j] = v[j * vd + hh * d + c];
                }
            }
            Packed::from_rows(&vt, d, tk)
        })
        .collect()
}

/// Rows i0.. of `weighted` (rows.len() / (heads * d) of them) into `rows`, from values packed by `pack_values`.
fn weighted_rows(wts: &[f32], vp: &[Packed], t: usize, tk: usize, d: usize, i0: usize, rows: &mut [f32]) {
    let heads = vp.len();
    let cnt = rows.len() / (heads * d);
    let mut tmp = vec![0.0; cnt * d];
    for (hh, vph) in vp.iter().enumerate() {
        gemm_into(&wts[(hh * t + i0) * tk..(hh * t + i0 + cnt) * tk], vph, &mut tmp);
        for c in 0..cnt {
            rows[c * heads * d + hh * d..c * heads * d + (hh + 1) * d].copy_from_slice(&tmp[c * d..(c + 1) * d]);
        }
    }
}

struct SelfAttn {
    inp: Linear,
    out: Linear,
    heads: usize,
}

impl SelfAttn {
    /// x += SelfAttn(x): the value projection in parallel row blocks, then weighting + output projection + residual
    /// add in parallel row blocks (two pool synchronizations).
    fn add_into(&self, x: &mut [f32], t: usize, wts: &[f32], tk: usize, cache: Option<&mut Vec<f32>>) {
        let vdim = self.inp.n();
        let xd = x.len() / t.max(1);
        let mut vn = vec![0.0; t * vdim];
        {
            let xs: &[f32] = x;
            par_blocks(&mut vn, vdim, t * xd * vdim, &|r0, vs| {
                let nr = vs.len() / vdim;
                vs.copy_from_slice(&self.inp.apply(&xs[r0 * xd..(r0 + nr) * xd], nr));
            });
        }
        let v = cat_cache(cache, vn);
        let d = vdim / self.heads;
        let vp = pack_values(&v, self.heads, tk, d);
        par_blocks(x, xd, t * tk * vdim + t * vdim * self.out.n(), &|i0, xs| {
            let nr = xs.len() / xd;
            let mut o = vec![0.0; nr * vdim];
            weighted_rows(wts, &vp, t, tk, d, i0, &mut o);
            let y = self.out.apply(&o, nr);
            add(xs, &y);
        });
    }
}

struct NonlinAttn {
    inp: Linear,
    out: Linear,
    hid: usize,
}

impl NonlinAttn {
    /// x += NonlinAttn(x) (head 0 of the attention weights): the input projection + tanh gate in parallel row blocks,
    /// then weighting + gate + output projection + residual add in parallel row blocks (two pool synchronizations).
    fn add_into(&self, x: &mut [f32], t: usize, wts: &[f32], tk: usize, cache: Option<&mut Vec<f32>>) {
        let hid = self.hid;
        let xd = x.len() / t.max(1);
        // z rows [a | b | c]: a is replaced by the gated input b * tanh(a)
        let mut z = vec![0.0; t * 3 * hid];
        {
            let xs: &[f32] = x;
            par_blocks(&mut z, 3 * hid, t * xd * 3 * hid, &|r0, zs| {
                let nr = zs.len() / (3 * hid);
                zs.copy_from_slice(&self.inp.apply(&xs[r0 * xd..(r0 + nr) * xd], nr));
                for r in zs.chunks_exact_mut(3 * hid) {
                    for c in 0..hid {
                        r[c] = r[hid + c] * tanh(r[c]);
                    }
                }
            });
        }
        let mut xs = vec![0.0; t * hid];
        for i in 0..t {
            xs[i * hid..(i + 1) * hid].copy_from_slice(&z[i * 3 * hid..i * 3 * hid + hid]);
        }
        let xs = cat_cache(cache, xs);
        let vp = pack_values(&xs, 1, tk, hid);
        par_blocks(x, xd, t * tk * hid + t * hid * self.out.n(), &|i0, xr| {
            let nr = xr.len() / xd;
            let mut o = vec![0.0; nr * hid];
            weighted_rows(&wts[..t * tk], &vp, t, tk, hid, i0, &mut o);
            for (ii, orow) in o.chunks_exact_mut(hid).enumerate() {
                let zc = &z[(i0 + ii) * 3 * hid + 2 * hid..(i0 + ii + 1) * 3 * hid];
                orow.iter_mut().zip(zc).for_each(|(v, g)| *v *= g);
            }
            let y = self.out.apply(&o, nr);
            add(xr, &y);
        });
    }
}

struct Conv {
    inp: Linear,
    // weights tap-major (tap x dim), so the inner loops run over contiguous channels
    causal_w: Vec<f32>, // half x dim
    causal_b: Vec<f32>,
    chunk_w: Vec<f32>, // kernel x dim
    chunk_b: Vec<f32>,
    scale: Vec<f32>, // 2 x kernel x dim
    out: Linear,
    dim: usize,
    kernel: usize,
}

impl Conv {
    /// x += Conv(x): the input product + gating in parallel row blocks, then the depthwise convolutions + swoosh +
    /// output product + residual add in parallel row blocks (two pool synchronizations).
    /// `cache`: the last kernel/2 rows of gated input before this chunk (streaming), updated.
    fn add_into(&self, x: &mut [f32], t: usize, cache: Option<&mut Vec<f32>>) {
        let (d, k) = (self.dim, self.kernel);
        let lp = k / 2;
        let xd = x.len() / t.max(1);
        let mut g = vec![0.0; t * d];
        {
            let xs: &[f32] = x;
            par_blocks(&mut g, d, t * xd * 2 * d, &|r0, gs| {
                let nr = gs.len() / d;
                let z = self.inp.apply(&xs[r0 * xd..(r0 + nr) * xd], nr);
                for (i, gr) in gs.chunks_exact_mut(d).enumerate() {
                    crate::kernels::glu(&z[i * 2 * d..i * 2 * d + d], &z[i * 2 * d + d..(i + 1) * 2 * d], gr);
                }
            });
        }
        let mut pad = match &cache {
            Some(c) => (*c).clone(),
            None => vec![0.0; lp * d],
        };
        pad.extend_from_slice(&g);
        if let Some(c) = cache {
            *c = pad[pad.len() - lp * d..].to_vec();
        }
        let half = lp + 1;
        let work = t * d * (half + k) + t * d * self.out.n();
        par_blocks(x, xd, work, &|i0, xs| {
            let nr = xs.len() / xd;
            let mut y = vec![0.0; nr * d];
            let mut b = vec![0.0; d];
            for (ii, a) in y.chunks_exact_mut(d).enumerate() {
                let i = i0 + ii;
                crate::kernels::dw_taps(a, Some(&self.causal_b), &self.causal_w, &pad[i * d..(i + half) * d], half);
                let s0 = i.saturating_sub(lp);
                let s1 = (i + k - lp).min(t);
                let u0 = s0 + lp - i;
                crate::kernels::dw_taps(&mut b, Some(&self.chunk_b), &self.chunk_w[u0 * d..], &g[s0 * d..s1 * d], s1 - s0);
                let (l, r) = if t < k { (Some(i), Some(k - t + i)) } else { ((i < k).then_some(i), (i >= t - k).then(|| i - (t - k))) };
                let row = |j: Option<usize>, half: usize| j.map(|j| &self.scale[(half * k + j) * d..(half * k + j + 1) * d]);
                match (row(l, 0), row(r, 1)) {
                    (None, None) => a.iter_mut().zip(&b).for_each(|(a, b)| *a += b),
                    (Some(le), None) => a.iter_mut().zip(&b).zip(le).for_each(|((a, b), le)| *a += b * (1.0 + le)),
                    (None, Some(re)) => a.iter_mut().zip(&b).zip(re).for_each(|((a, b), re)| *a += b * (1.0 + re)),
                    (Some(le), Some(re)) => a.iter_mut().zip(&b).zip(le).zip(re).for_each(|(((a, b), le), re)| *a += b * (1.0 + le + re)),
                }
            }
            swoosh_r(&mut y);
            let o = self.out.apply(&y, nr);
            add(xs, &o);
        });
    }
}

/// Streaming state of one layer.
#[derive(Clone)]
pub struct LayerState {
    key: Vec<f32>,
    nonlin: Vec<f32>,
    val1: Vec<f32>,
    val2: Vec<f32>,
    conv1: Vec<f32>,
    conv2: Vec<f32>,
}

struct Layer {
    attn: AttnWeights,
    ff1: FeedForward,
    ff2: FeedForward,
    ff3: FeedForward,
    nonlin: NonlinAttn,
    sa1: SelfAttn,
    sa2: SelfAttn,
    conv1: Conv,
    conv2: Conv,
    norm: BiasNorm,
    bypass: Vec<f32>,
    bypass_mid: Vec<f32>,
}

impl Layer {
    fn forward(&self, x: &mut Vec<f32>, t: usize, st: Option<&mut LayerState>, left: usize, masked: &[bool], pos_dim: usize) {
        let orig = x.clone();
        let tk = left + t;
        let (mut key, mut nl, mut v1, mut v2, mut c1, mut c2) = (None, None, None, None, None, None);
        if let Some(s) = st {
            key = Some(&mut s.key);
            nl = Some(&mut s.nonlin);
            v1 = Some(&mut s.val1);
            v2 = Some(&mut s.val2);
            c1 = Some(&mut s.conv1);
            c2 = Some(&mut s.conv2);
        }
        let w = self.attn.apply(x, t, key, left, masked, pos_dim);
        self.ff1.add_into(x, t, &|_, _| {});
        self.nonlin.add_into(x, t, &w, tk, nl);
        self.sa1.add_into(x, t, &w, tk, v1);
        self.conv1.add_into(x, t, c1);
        let d = x.len() / t.max(1);
        self.ff2.add_into(x, t, &|r0, rows| bypass(&orig[r0 * d..r0 * d + rows.len()], rows, &self.bypass_mid));
        self.sa2.add_into(x, t, &w, tk, v2);
        self.conv2.add_into(x, t, c2);
        self.ff3.add_into(x, t, &|r0, rows| {
            self.norm.apply(rows);
            bypass(&orig[r0 * d..r0 * d + rows.len()], rows, &self.bypass);
        });
    }
}

struct Stack {
    layers: Vec<Layer>,
    dim: usize,
    ds: usize,
    down: Vec<f32>, // softmax of the downsample bias
    combine: Vec<f32>,
}

fn softmax(b: &[f32]) -> Vec<f32> {
    let m = b.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = b.iter().map(|v| exp(v - m)).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

/// SimpleDownsample: weighted sum of `w.len()` consecutive rows (the last row repeated as padding).
fn downsample(x: &[f32], t: usize, d: usize, w: &[f32]) -> (Vec<f32>, usize) {
    let ds = w.len();
    let dt = t.div_ceil(ds);
    let mut out = vec![0.0; dt * d];
    for i in 0..dt {
        for (k, wk) in w.iter().enumerate() {
            let s = (i * ds + k).min(t - 1);
            out[i * d..(i + 1) * d].iter_mut().zip(&x[s * d..(s + 1) * d]).for_each(|(o, v)| *o += wk * v);
        }
    }
    (out, dt)
}

impl Stack {
    /// x: t x dim. `masked`: left-context slots at 50 Hz (len left_full) that hold no audio yet.
    fn forward(&self, x: &mut Vec<f32>, t: usize, st: Option<&mut [LayerState]>, left_full: usize, masked_full: &[bool], pos_dim: usize) {
        let d = self.dim;
        let left = left_full / self.ds;
        let mut masked: Vec<bool> = (0..left).map(|j| masked_full.get(j * self.ds).copied().unwrap_or(false)).collect();
        if self.ds == 1 {
            masked.resize(left + t, false);
            run(&self.layers, x, t, st, left, &masked, pos_dim);
            return;
        }
        let orig = x.clone();
        let (mut y, dt) = downsample(x, t, d, &self.down);
        masked.resize(left + dt, false);
        run(&self.layers, &mut y, dt, st, left, &masked, pos_dim);
        let mut up = vec![0.0; t * d];
        for i in 0..t {
            up[i * d..(i + 1) * d].copy_from_slice(&y[(i / self.ds) * d..(i / self.ds + 1) * d]);
        }
        bypass(&orig, &mut up, &self.combine);
        *x = up;
    }
}

fn run(layers: &[Layer], x: &mut Vec<f32>, t: usize, st: Option<&mut [LayerState]>, left: usize, masked: &[bool], pos_dim: usize) {
    match st {
        Some(s) => layers.iter().zip(s.iter_mut()).for_each(|(l, s)| l.forward(x, t, Some(s), left, masked, pos_dim)),
        None => layers.iter().for_each(|l| l.forward(x, t, None, left, masked, pos_dim)),
    }
}

// ---------------------------------------------------------------------------------------------- front end

/// Conv2dSubsampling + ConvNeXt, channels-last images (t x f x c).
struct Embed {
    c0: Linear, // 1 -> 8, 3x3 (im2col)
    c0_rows: Vec<f32>, // c0's weights, 8 x 9 (row-major), for the direct convolution
    c1: Linear, // 8 -> 32, 3x3 stride 2
    c2: Linear, // 32 -> 128, 3x3 stride (1, 2)
    tap: [bool; 2], // c1 / c2 weights in tap-major column order (`Tensors::tap_major`)
    dw_w: Vec<f32>, // 7 x 7 x 128 (tap-major)
    dw_b: Vec<f32>,
    pw1: Linear,
    pw2: Linear,
    out: Linear, // input order (f, c) (permuted at load from PyTorch's (c, f))
    norm: BiasNorm,
}

const FW: usize = 19; // frequency width after the convolutions
const C3: usize = 128;

/// 3x3 convolution of a channels-last image (t x f x cin), no time padding, `pf` frequency padding. `tap`: the
/// weights' columns are tap-major (`Tensors::tap_major`), else PyTorch's order.
#[allow(clippy::too_many_arguments)]
fn conv3(x: &[f32], t: usize, f: usize, cin: usize, lin: &Linear, st: usize, sf: usize, pf: usize, tap: bool) -> (Vec<f32>, usize, usize) {
    let to = (t - 3) / st + 1;
    let fo = (f + 2 * pf - 3) / sf + 1;
    let kl = lin.k(); // 9 cin, or more: zero padding of the weights' rows
    let n = lin.n();
    let mut y = vec![0.0; to * fo * n];
    // input rows built for a few output time steps at a time, so they (and the product) stay in cache
    let sub = (CONV_SUB / (fo * kl)).max(1);
    // im2col, product and activation per block of output time steps (row-wise: one pool synchronization)
    par_blocks(&mut y, fo * n, to * fo * kl * n, &|i0, ys| {
        let nt = ys.len() / (fo * n);
        // zeros beyond 9 cin (the padding) are never written
        let mut colbuf = vec![0.0; sub.min(nt) * fo * kl];
        let mut s0 = 0;
        while s0 < nt {
            let ns = sub.min(nt - s0);
            let col = &mut colbuf[..ns * fo * kl];
            for ii in 0..ns {
                let i = i0 + s0 + ii;
                for j in 0..fo {
                    let row = &mut col[(ii * fo + j) * kl..(ii * fo + j) * kl + 9 * cin];
                    if tap {
                        for a in 0..3 {
                            let tt = i * st + a;
                            let f0 = (j * sf) as isize - pf as isize;
                            if f0 >= 0 && f0 as usize + 3 <= f {
                                let s = (tt * f + f0 as usize) * cin;
                                row[a * 3 * cin..(a + 1) * 3 * cin].copy_from_slice(&x[s..s + 3 * cin]);
                            } else {
                                for b in 0..3 {
                                    let ff = f0 + b as isize;
                                    let dst = &mut row[(a * 3 + b) * cin..(a * 3 + b + 1) * cin];
                                    if ff >= 0 && (ff as usize) < f {
                                        let s = (tt * f + ff as usize) * cin;
                                        dst.copy_from_slice(&x[s..s + cin]);
                                    } else {
                                        dst.fill(0.0);
                                    }
                                }
                            }
                        }
                    } else {
                        for ci in 0..cin {
                            for a in 0..3 {
                                for b in 0..3 {
                                    let (tt, ff) = (i * st + a, (j * sf + b) as isize - pf as isize);
                                    // PyTorch weight layout (cout, cin, 3, 3) -> column index ci*9 + a*3 + b
                                    row[ci * 9 + a * 3 + b] = if ff >= 0 && (ff as usize) < f { x[(tt * f + ff as usize) * cin + ci] } else { 0.0 };
                                }
                            }
                        }
                    }
                }
            }
            let mut z = lin.apply(col, ns * fo);
            swoosh_r(&mut z);
            ys[s0 * fo * n..(s0 + ns) * fo * n].copy_from_slice(&z);
            s0 += ns;
        }
    });
    (y, to, fo)
}

/// About the floats of im2col rows built at a time in `conv3` (fits the L2 cache with the product).
const CONV_SUB: usize = 48 * 1024;
/// Time steps per fused ConvNeXt block, rows per `Embed::finish` block (cache-sized).
const CNX_SUB: usize = 2;
const FINISH_SUB: usize = 32;

/// c0 (1 -> 8 channels, 3x3, frequency padding 1, stride 1) computed directly, the same arithmetic as the f32 product
/// of its im2col rows (bias, then a fused multiply-add per tap in PyTorch's column order), then SwooshR.
fn conv0_direct(x: &[f32], t: usize, f: usize, w: &[f32], bias: &[f32]) -> (Vec<f32>, usize, usize) {
    let (to, fo) = (t - 2, f);
    let mut y = vec![0.0; to * fo * 8];
    par_blocks(&mut y, fo * 8, to * fo * 72, &|i0, ys| {
        let nt = ys.len() / (fo * 8);
        crate::kernels::conv0_rows(x, f, i0, nt, w, bias, ys);
        swoosh_r(ys);
    });
    (y, to, fo)
}

impl Embed {
    /// Convolutions up to (not including) ConvNeXt: features (t x 80) -> (t' x 19 x 128), t' = (t - 7) / 2 + 1.
    fn convs(&self, feats: &[f32], t: usize) -> (Vec<f32>, usize) {
        let (x, t, f) = if crate::kernels::conv0_direct_ok() && self.c0_rows.len() == 72 && self.c0.b.as_ref().is_some_and(|b| b.len() == 8) {
            conv0_direct(feats, t, 80, &self.c0_rows, self.c0.b.as_deref().unwrap_or(&[]))
        } else {
            conv3(feats, t, 80, 1, &self.c0, 1, 1, 1, false)
        };
        let (x, t, f) = conv3(&x, t, f, 8, &self.c1, 2, 2, 0, self.tap[0]);
        let (x, t, _) = conv3(&x, t, f, 32, &self.c2, 1, 2, 0, self.tap[1]);
        (x, t)
    }

    /// ConvNeXt over `rows` (t_in x 19 x 128) giving rows `[3, t_in - 3)` of the time-padded result: the caller
    /// supplies 3 rows of context (zeros or cache) on each side.
    fn convnext(&self, rows: &[f32], t_in: usize) -> Vec<f32> {
        let t = t_in - 6;
        let row = FW * C3;
        let mut z = vec![0.0; t * row];
        let work = t * row * 49 + t * FW * C3 * self.pw1.n() * 2;
        // depthwise 7x7, pointwise products, activation and residual add per block of time steps (row-wise in time:
        // one pool synchronization)
        par_blocks(&mut z, row, work, &|i0, zsall| {
            let ntall = zsall.len() / row;
            let mut y = vec![0.0; CNX_SUB.min(ntall) * row];
            // a few time steps at a time: depthwise, pointwise products and activation stay in cache
            for (sb, zs) in zsall.chunks_mut(CNX_SUB * row).enumerate() {
              let i0 = i0 + sb * CNX_SUB;
              let nt = zs.len() / row;
              let y = &mut y[..nt * row];
              for (ii, yr) in y.chunks_exact_mut(row).enumerate() {
                  let i = i0 + ii;
                  for f in 0..FW {
                      let a = &mut yr[f * C3..(f + 1) * C3];
                      a.copy_from_slice(&self.dw_b);
                      // taps df whose column f + df - 3 is inside 0..FW
                      let (df0, df1) = (3usize.saturating_sub(f), 7.min(FW + 3 - f));
                      for dt in 0..7 {
                          let x0 = (i + dt) * row + (f + df0 - 3) * C3;
                          let w0 = (dt * 7 + df0) * C3;
                          crate::kernels::dw_taps(a, None, &self.dw_w[w0..], &rows[x0..x0 + (df1 - df0) * C3], df1 - df0);
                      }
                  }
              }
              let mut h = self.pw1.apply(y, nt * FW);
              swoosh_l(&mut h);
              let mut o = self.pw2.apply(&h, nt * FW);
              add(&mut o, &rows[(3 + i0) * row..(3 + i0 + nt) * row]);
              zs.copy_from_slice(&o);
            }
        });
        z
    }

    fn finish(&self, x: &[f32], t: usize) -> Vec<f32> {
        let (n, kx) = (self.out.n(), x.len() / t.max(1));
        let mut y = vec![0.0; t * n];
        par_blocks(&mut y, n, t * kx * n, &|r0, ysall| {
            for (sb, ys) in ysall.chunks_mut(FINISH_SUB * n).enumerate() {
                let r0 = r0 + sb * FINISH_SUB;
                let nr = ys.len() / n;
                let mut o = self.out.apply(&x[r0 * kx..(r0 + nr) * kx], nr);
                self.norm.apply(&mut o);
                ys.copy_from_slice(&o);
            }
        });
        y
    }
}

// ---------------------------------------------------------------------------------------------- model

/// A recognized token: vocabulary id, output frame (40 ms) and log-probability.
#[derive(Clone, Copy, Debug)]
pub struct Token {
    pub id: u32,
    pub frame: u32,
    pub logp: f32,
}


struct Joiner {
    enc: Linear,
    dec: Linear,
    out: Vec<(Linear, Vec<u32>)>, // per language: output rows of the allowed ids (shared + English + own), the ids
}

/// End-of-turn head (eou_head.py): P(the speaker has finished) from the second pass's last `HEAD_K` encoder frames
/// (attention pooled + the last one) and the decoder output after the last text tokens. Optional in the file.
pub const HEAD_K: usize = 50;
struct Head {
    e: Linear,
    pos: Vec<f32>, // HEAD_K x hidden
    att: Linear,
    d: Linear,
    out0: Linear,
    out2: Linear,
}

/// erf (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7), for the exact GELU the head was trained with.
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0 - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_7) * t - 0.284_496_74) * t + 0.254_829_6) * t * exp(-x * x);
    if x < 0.0 { -y } else { y }
}

fn gelu(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v = 0.5 * *v * (1.0 + erf(*v * core::f32::consts::FRAC_1_SQRT_2)));
}

impl Head {
    /// `enc_dim`: encoder output width, `dec_dim`: decoder output width (before the joiner).
    fn load(w: &mut Tensors, enc_dim: usize, dec_dim: usize) -> R<Option<Head>> {
        if !w.has("eou_head.e.0.weight") {
            return Ok(None);
        }
        let e = w.linear("eou_head.e.0", true, None, enc_dim)?;
        let hd = e.n();
        let d = w.linear("eou_head.d.0", true, None, dec_dim)?;
        let out0 = w.linear("eou_head.out.0", true, None, 2 * hd + d.n())?;
        Ok(Some(Head {
            pos: w.vec("eou_head.pos", HEAD_K * hd)?,
            att: w.linear("eou_head.att", true, Some(1), hd)?,
            out2: w.linear("eou_head.out.2", true, None, out0.n())?,
            e,
            d,
            out0,
        }))
    }

    /// enc: HEAD_K x 512 right-aligned (the first HEAD_K - valid rows unused), dec: decoder output before the joiner.
    fn prob(&self, enc: &[f32], valid: usize, dec: &[f32]) -> f32 {
        let k = HEAD_K;
        let valid = valid.clamp(1, k);
        let d = enc.len() / k;
        let mut h = self.e.apply(&enc[(k - valid) * d..], valid);
        gelu(&mut h);
        let hd = h.len() / valid;
        for (i, row) in h.chunks_exact_mut(hd).enumerate() {
            let p = &self.pos[(k - valid + i) * hd..(k - valid + i + 1) * hd];
            row.iter_mut().zip(p).for_each(|(a, b)| *a += b);
        }
        let sc = self.att.apply(&h, valid);
        let m = sc.iter().cloned().fold(f32::MIN, f32::max);
        let wts: Vec<f32> = sc.iter().map(|v| exp(v - m)).collect();
        let z: f32 = wts.iter().sum();
        let mut x = vec![0.0; 2 * hd];
        for (i, row) in h.chunks_exact(hd).enumerate() {
            row.iter().enumerate().for_each(|(j, v)| x[j] += wts[i] / z * v);
        }
        x[hd..].copy_from_slice(&h[(valid - 1) * hd..]);
        let mut dd = self.d.apply(dec, 1);
        gelu(&mut dd);
        x.extend_from_slice(&dd);
        let mut y = self.out0.apply(&x, 1);
        gelu(&mut y);
        sigmoid(self.out2.apply(&y, 1)[0])
    }
}

/// v5.9t hybrid decoder: an LSTM over the whole token history, added to the stateless output through a scalar gate
/// (decoder.py: embedding_out + gate * lstm_proj(lstm(embedding(y)))). One step per emitted token.
struct Lstm {
    ih: Linear, // 4H x D, bias = b_ih + b_hh
    hh: Linear, // 4H x H
    proj: Linear, // D x H
    gate: f32,
    hidden: usize,
}

impl Lstm {
    /// One step of torch.nn.LSTM (gate order i, f, g, o) on input x; updates h, c.
    fn step(&self, x: &[f32], h: &mut [f32], c: &mut [f32]) {
        let n = self.hidden;
        let mut g = self.ih.apply(x, 1);
        let r = self.hh.apply(h, 1);
        g.iter_mut().zip(&r).for_each(|(a, b)| *a += b);
        crate::kernels::lstm_cell(&g, &mut c[..n], &mut h[..n]);
    }
}

/// Most zero-duration tokens on one frame in TDT greedy before it is forced on (the server scorer's max_sym).
const TDT_MAX_SYM: usize = 5;
/// Frames a beam-search decoder state is kept for reuse (more than the longest TDT duration, 4).
const MEMO_FRAMES: usize = 8;

/// One beam-search hypothesis: decoder state, total log-probability, tokens.
#[derive(Clone)]
pub struct Hyp {
    dec: alloc::sync::Arc<Decoding>, // shared with sibling hypotheses until advanced (copy on write)
    logp: f32,
    toks: Vec<Token>,
    pending: bool, // TDT beam: the last token is not yet in `dec` (advanced only if the hypothesis survives pruning)
}

pub struct Model {
    head: Option<Head>,
    embed: Embed,
    stacks: Vec<Stack>,
    out_down: Vec<f32>,
    lang_vec: Vec<Vec<f32>>,
    dec_emb: Vec<f32>,
    dec_conv: Vec<f32>, // dim x 4 x context (groups of 4 channels)
    dec_dim: usize,
    context: usize,
    joiner: Joiner,
    lstm: Option<Lstm>,
    /// Per token id: the LSTM input product ih(embedding) + bias (4 x hidden), filled on first use. The same values
    /// as computing it each step (int8 rows are quantized and multiplied independently of the batch).
    #[cfg(feature = "std")]
    ih_cache: Vec<std::sync::OnceLock<alloc::boxed::Box<[f32]>>>,
    /// Unique per loaded model (never reused, unlike the address): the key of the per-thread `decoder_y_into` cache.
    #[cfg(feature = "std")]
    uid: usize,
    dur: Option<Linear>, // TDT duration head (durations 0..n-1) on the joiner hidden; None = RNN-T
    pos_dim: usize,
    pub left: usize,
    pub chunk: usize,
    pub pieces: Vec<String>,
    /// The languages, in the model's order (its lang_embed rows and output blocks).
    pub langs: Vec<crate::Lang>,
}

impl Model {
    /// `l`'s place among the model's languages (its first, English, for one it doesn't have).
    pub fn lang_index(&self, l: crate::Lang) -> usize {
        self.langs.iter().position(|&m| m == l).unwrap_or(0)
    }
}

/// Most tokens emitted on one encoder frame in greedy search (from the MAX_SYMBOLS env var in std builds).
#[cfg(feature = "std")]
static MAX_SYMBOLS_ENV: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
#[cfg(feature = "std")]
fn max_symbols() -> usize {
    *MAX_SYMBOLS_ENV.get_or_init(|| std::env::var("MAX_SYMBOLS").ok().and_then(|v| v.parse().ok()).unwrap_or(3))
}
#[cfg(not(feature = "std"))]
fn max_symbols() -> usize {
    3
}

impl Model {
    /// The recognizer from its .tzf file. Every shape the model runs with is checked here against the settings and
    /// the other tensors, so a corrupted file or one for another version of the library is an error, never a crash
    /// later.
    pub fn from_bytes(b: &[u8]) -> Result<Model, &'static str> {
        let mut w = Tensors::parse(b)?;
        let ns = w.c("n_stacks", 1, 16)?;
        let pos_dim = w.c("pos_dim", 2, 1024)?;
        let left = w.c("left_context", 0, 1024)?;
        let chunk = w.c("chunk", 1, 512)?;
        let context = w.c("context", 1, 16)?;
        let mut stacks: Vec<Stack> = Vec::new();
        for s in 0..ns {
            let g = |w: &Tensors, k: &str, lo, hi| w.c(&format!("{k}.{s}"), lo, hi);
            let dim = g(&w, "encoder_dim", 4, 4096)?;
            let nl = g(&w, "num_encoder_layers", 1, 64)?;
            let ds = g(&w, "downsampling_factor", 1, 64)?;
            let heads = g(&w, "num_heads", 1, 64)?;
            let qd = g(&w, "query_head_dim", 1, 1024)?;
            let pd = g(&w, "pos_head_dim", 1, 1024)?;
            let kernel = g(&w, "cnn_module_kernel", 1, 255)?;
            let half = kernel / 2 + 1;
            let hid = 3 * dim / 4;
            let base = if ds == 1 { format!("encoder.encoders.{s}") } else { format!("encoder.encoders.{s}.encoder") };
            let mut layers: Vec<Layer> = Vec::new();
            for l in 0..nl {
                let p = format!("{base}.layers.{l}");
                let ff = |w: &mut Tensors, n: &str| -> R<FeedForward> {
                    let inp = w.linear(&format!("{p}.{n}.in_proj"), true, None, dim)?;
                    let out = w.linear(&format!("{p}.{n}.out_proj"), true, Some(dim), inp.n())?;
                    Ok(FeedForward { inp, out })
                };
                let conv = |w: &mut Tensors, n: &str| -> R<Conv> {
                    let q = |x: &str| format!("{p}.{n}.{x}");
                    let sc = w.vec(&q("depthwise_conv.chunkwise_conv_scale"), 2 * dim * kernel)?;
                    let (l, r) = sc.split_at(dim * kernel);
                    let mut scale = transpose(l, dim, kernel);
                    scale.extend(transpose(r, dim, kernel));
                    Ok(Conv {
                        inp: w.linear(&q("in_proj"), true, Some(2 * dim), dim)?,
                        causal_w: transpose(&w.vec(&q("depthwise_conv.causal_conv.weight"), dim * half)?, dim, half),
                        causal_b: w.vec(&q("depthwise_conv.causal_conv.bias"), dim)?,
                        chunk_w: transpose(&w.vec(&q("depthwise_conv.chunkwise_conv.weight"), dim * kernel)?, dim, kernel),
                        chunk_b: w.vec(&q("depthwise_conv.chunkwise_conv.bias"), dim)?,
                        scale,
                        out: w.linear(&q("out_proj"), true, Some(dim), dim)?,
                        dim,
                        kernel,
                    })
                };
                let sa = |w: &mut Tensors, n: &str| -> R<SelfAttn> {
                    let inp = w.linear(&format!("{p}.{n}.in_proj"), true, None, dim)?;
                    if inp.n() % heads != 0 {
                        return Err(SHAPE);
                    }
                    let out = w.linear(&format!("{p}.{n}.out_proj"), true, Some(dim), inp.n())?;
                    Ok(SelfAttn { inp, out, heads })
                };
                let layer = Layer {
                    attn: AttnWeights {
                        in_proj: w.linear(&format!("{p}.self_attn_weights.in_proj"), true, Some(2 * heads * qd + heads * pd), dim)?,
                        pos: w.linear(&format!("{p}.self_attn_weights.linear_pos"), false, Some(heads * pd), pos_dim)?,
                        heads,
                        qd,
                        pd,
                    },
                    ff1: ff(&mut w, "feed_forward1")?,
                    ff2: ff(&mut w, "feed_forward2")?,
                    ff3: ff(&mut w, "feed_forward3")?,
                    nonlin: NonlinAttn {
                        inp: w.linear(&format!("{p}.nonlin_attention.in_proj"), true, Some(3 * hid), dim)?,
                        out: w.linear(&format!("{p}.nonlin_attention.out_proj"), true, Some(dim), hid)?,
                        hid,
                    },
                    sa1: sa(&mut w, "self_attn1")?,
                    sa2: sa(&mut w, "self_attn2")?,
                    conv1: conv(&mut w, "conv_module1")?,
                    conv2: conv(&mut w, "conv_module2")?,
                    norm: BiasNorm::load(&mut w, &format!("{p}.norm"), dim)?,
                    bypass: w.vec(&format!("{p}.bypass.bypass_scale"), dim)?,
                    bypass_mid: w.vec(&format!("{p}.bypass_mid.bypass_scale"), dim)?,
                };
                // the streaming state of every layer is laid out as layer 0's
                if let Some(l0) = layers.first() {
                    if layer.sa1.inp.n() != l0.sa1.inp.n() || layer.sa2.inp.n() != l0.sa2.inp.n() {
                        return Err(SHAPE);
                    }
                }
                layers.push(layer);
            }
            let (down, combine) = if ds == 1 {
                (vec![], vec![])
            } else {
                (softmax(&w.vec(&format!("encoder.encoders.{s}.downsample.bias"), ds)?), w.vec(&format!("encoder.encoders.{s}.out_combiner.bypass_scale"), dim)?)
            };
            stacks.push(Stack { layers, dim, ds, down, combine });
        }
        let d0 = stacks[0].dim;
        let enc_dim = stacks.iter().map(|s| s.dim).max().unwrap_or(d0); // the full-dim output (`encode`)
        let out_down = softmax(&w.vec_in("encoder.downsample_output.bias", 1, 64)?);
        // front end: im2col weights stay PyTorch (cout, cin*9); `out` takes (f, c) instead of (c, f)
        let (od, ow) = w.get("encoder_embed.out.weight")?;
        if od.first() != Some(&d0) || ow.len() != d0 * FW * C3 {
            return Err(SHAPE);
        }
        let ok = FW * C3;
        let mut ow2 = vec![0.0; ow.len()];
        for n in 0..d0 {
            for c in 0..C3 {
                for f in 0..FW {
                    ow2[n * ok + f * C3 + c] = ow[n * ok + c * FW + f];
                }
            }
        }
        let pw1 = w.linear("encoder_embed.convnext.pointwise_conv1", true, None, C3)?;
        let tap = [w.tap_major("encoder_embed.conv.4.weight", 8), w.tap_major("encoder_embed.conv.7.weight", 32)];
        let c0 = w.linear("encoder_embed.conv.0", true, Some(8), 9)?;
        let embed = Embed {
            c0_rows: match &c0.w {
                Weight::F32(p) => p.rows(),
                Weight::Int(_) => Vec::new(),
            },
            c0,
            c1: w.linear("encoder_embed.conv.4", true, Some(32), 8 * 9)?,
            c2: w.linear("encoder_embed.conv.7", true, Some(C3), 32 * 9)?,
            tap,
            dw_w: transpose(&w.vec("encoder_embed.convnext.depthwise_conv.weight", C3 * 49)?, C3, 49),
            dw_b: w.vec("encoder_embed.convnext.depthwise_conv.bias", C3)?,
            pw2: w.linear("encoder_embed.convnext.pointwise_conv2", true, Some(C3), pw1.n())?,
            pw1,
            out: Linear { w: Weight::F32(Packed::from_rows(&ow2, d0, ok)), b: Some(w.vec("encoder_embed.out.bias", d0)?) },
            norm: BiasNorm::load(&mut w, "encoder_embed.out_norm", d0)?,
        };
        use crate::Lang;
        // the language set: "lang.K" = Lang index of the K-th language (files from en-ru-uz on); older files by its size
        let n = w.c("n_langs", 1, Lang::ALL.len())?;
        let langs = if w.cfg.contains_key("lang.0") {
            let mut v = alloc::vec::Vec::new();
            for k in 0..n {
                let i = w.c(&alloc::format!("lang.{k}"), 0, Lang::ALL.len() - 1)?;
                v.push(Lang::ALL[i]);
            }
            v
        } else {
            match n {
                3 => alloc::vec![Lang::En, Lang::Ru, Lang::Uk],
                4 => alloc::vec![Lang::En, Lang::Es, Lang::It, Lang::Pt],
                _ => return Err("model file: unknown language set"),
            }
        };
        let le = w.get("lang_embed.weight")?.1;
        if le.len() < n * d0 {
            return Err(SHAPE);
        }
        let lang_vec = (0..n).map(|k| le[k * d0..(k + 1) * d0].to_vec()).collect();
        // decoder: embedding (vocabulary x dec_dim, dec_dim in groups of 4) and its convolution over `context` tokens
        let (dd, dec_emb) = w.get("decoder.embedding.weight")?;
        let dec_dim: usize = dd.get(1..).map_or(0, |r| r.iter().product());
        if dd.len() < 2 || dec_dim == 0 || dec_dim % 4 != 0 || dd[0] == 0 {
            return Err(SHAPE);
        }
        let dec_conv = w.vec("decoder.conv.weight", dec_dim * 4 * context)?;
        // joiner: every id a language can emit (< shared + n * lang_size) has an output row and a decoder embedding
        let shared = w.c("shared", 0, 1 << 16)?;
        let size = w.c("lang_size", 0, 1 << 16)?;
        let enc = w.linear("joiner.encoder_proj", true, None, enc_dim)?;
        let jdim = enc.n();
        // the output rows stay integer where the int8 product is available (a quarter of the memory read per step)
        let (jd, jraw) = w.t.remove("joiner.output_linear.weight").ok_or(MISSING)?;
        let jint = match jraw {
            Data::Int { group, row, scale, q } if row == jdim && jd.len() == 2 && Quant::int8_for(group) => Some((group, scale, q)),
            raw => {
                w.t.insert("joiner.output_linear.weight".into(), (jd.clone(), raw));
                None
            }
        };
        let jw = if jint.is_none() { w.get("joiner.output_linear.weight")?.1 } else { Vec::new() };
        let vocab = jd.first().copied().unwrap_or(0);
        if vocab == 0 || (jint.is_none() && jw.len() != vocab * jdim) || shared + size == 0 || shared + n * size > vocab || vocab > dd[0] {
            return Err(SHAPE);
        }
        if let Some((g, s, q)) = &jint {
            if q.len() != vocab * jdim || s.len() != vocab * (jdim / g) {
                return Err(SHAPE);
            }
        }
        let jb = w.vec("joiner.output_linear.bias", vocab)?;
        let (shared, size) = (shared as u32, size as u32);
        let out = (0..n as u32)
            .map(|k| {
                // shared block + English; ґ pieces in the shared block (reserved ids, tok_dir/extra.json) only for Ukrainian
                let uk_only = |i: u32| i < shared && w.pieces.get(i as usize).is_some_and(|p| p.contains('ґ') || p.contains('Ґ'));
                let mut ids: Vec<u32> = (0..shared + size).filter(|&i| langs[k as usize] == Lang::Uk || !uk_only(i)).collect();
                if k != 0 {
                    ids.extend(shared + k * size..shared + (k + 1) * size);
                }
                let b = ids.iter().map(|&i| jb[i as usize]).collect();
                if let Some((g, s, q)) = &jint {
                    let ng = jdim / g;
                    let mut rq = Vec::with_capacity(ids.len() * jdim);
                    let mut rs = Vec::with_capacity(ids.len() * ng);
                    for &i in &ids {
                        rq.extend_from_slice(&q[i as usize * jdim..(i as usize + 1) * jdim]);
                        rs.extend_from_slice(&s[i as usize * ng..(i as usize + 1) * ng]);
                    }
                    return (Linear { w: Weight::Int(Quant::new(ids.len(), jdim, *g, rs, rq)), b: Some(b) }, ids);
                }
                let mut rows = Vec::with_capacity(ids.len() * jdim);
                for &i in &ids {
                    rows.extend_from_slice(&jw[i as usize * jdim..(i as usize + 1) * jdim]);
                }
                (Linear { w: Weight::F32(Packed::from_rows(&rows, ids.len(), jdim)), b: Some(b) }, ids)
            })
            .collect::<Vec<_>>();
        if out.iter().any(|(_, ids)| ids.first() != Some(&0)) {
            return Err(SHAPE); // the blank (0) is every language's first output
        }
        let joiner = Joiner { enc, dec: w.linear("joiner.decoder_proj", true, Some(jdim), dec_dim)?, out };
        let dur = if w.has("joiner.duration_linear.weight") { Some(w.linear("joiner.duration_linear", true, None, jdim)?) } else { None };
        if dur.as_ref().is_some_and(|d| d.n() < 2) {
            return Err(SHAPE);
        }
        let lstm = if w.has("decoder.lstm.weight_ih_l0") {
            let b2 = w.vec_in("decoder.lstm.bias_hh_l0", 4, 1 << 16)?;
            let hidden = b2.len() / 4;
            if b2.len() % 4 != 0 {
                return Err(SHAPE);
            }
            let b1 = w.vec("decoder.lstm.bias_ih_l0", 4 * hidden)?;
            let mut ih = w.linear_named("decoder.lstm.weight_ih_l0", None, Some(4 * hidden), dec_dim)?;
            ih.b = Some(b1.iter().zip(&b2).map(|(a, b)| a + b).collect());
            let hh = w.linear_named("decoder.lstm.weight_hh_l0", None, Some(4 * hidden), hidden)?;
            let proj = w.linear_named("decoder.lstm_proj.weight", None, Some(dec_dim), hidden)?;
            Some(Lstm { ih, hh, proj, gate: w.scalar("decoder.lstm_gate_m.g")?, hidden })
        } else {
            None
        };
        let head = Head::load(&mut w, enc_dim, dec_dim)?;
        #[cfg(feature = "std")]
        let ih_n = if lstm.is_some() { dec_emb.len() / dec_dim } else { 0 };
        Ok(Model {
            head,
            embed,
            stacks,
            out_down,
            lang_vec,
            dec_emb,
            dec_conv,
            dec_dim,
            context,
            joiner,
            #[cfg(feature = "std")]
            ih_cache: (0..ih_n).map(|_| std::sync::OnceLock::new()).collect(),
            #[cfg(feature = "std")]
            uid: {
                static NEXT: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(1);
                NEXT.fetch_add(1, core::sync::atomic::Ordering::Relaxed)
            },
            lstm,
            dur,
            pos_dim,
            left,
            chunk,
            pieces: core::mem::take(&mut w.pieces),
            langs,
        })
    }

    /// Stacks + output combination: x (t x dim0) at 50 Hz -> (ceil(t / 2) x 512) at 25 Hz.
    fn encode(&self, x: Vec<f32>, t: usize, mut states: Option<&mut [Vec<LayerState>]>, masked: &[bool]) -> (Vec<f32>, usize) {
        let mut outs: Vec<Vec<f32>> = Vec::new();
        let mut cur = x;
        let mut cd = self.stacks[0].dim;
        let left = if states.is_some() { self.left } else { 0 };
        for (s, st) in self.stacks.iter().enumerate() {
            if st.dim != cd {
                // convert_num_channels: truncate or zero-pad to this stack's width
                let mut y = vec![0.0; t * st.dim];
                let m = st.dim.min(cd);
                for i in 0..t {
                    y[i * st.dim..i * st.dim + m].copy_from_slice(&cur[i * cd..i * cd + m]);
                }
                cur = y;
                cd = st.dim;
            }
            let ls = states.as_deref_mut().map(|all| &mut all[s][..]);
            st.forward(&mut cur, t, ls, left, masked, self.pos_dim);
            outs.push(cur.clone());
        }
        // full-dim output: the last stack, then the extra channels of earlier, wider stacks
        let last = self.stacks.len() - 1;
        let mut pieces = vec![(last, 0, self.stacks[last].dim)];
        let mut od = self.stacks[last].dim;
        for i in (0..last).rev() {
            let d = self.stacks[i].dim;
            if d > od {
                pieces.push((i, od, d));
                od = d;
            }
        }
        let mut full = vec![0.0; t * od];
        for i in 0..t {
            let mut o = 0;
            for &(s, a, b) in &pieces {
                let sd = self.stacks[s].dim;
                full[i * od + o..i * od + o + b - a].copy_from_slice(&outs[s][i * sd + a..i * sd + b]);
                o += b - a;
            }
        }
        downsample(&full, t, od, &self.out_down)
    }

    /// Full-context encoder output (frames at 25 Hz x 512) for fbank features (t x 80).
    pub fn encode_full(&self, feats: &[f32], t: usize, lang: usize) -> (Vec<f32>, usize) {
        let (x, tc) = self.embed.convs(feats, t);
        let row = FW * C3;
        let mut rows = vec![0.0; (tc + 6) * row];
        rows[3 * row..(3 + tc) * row].copy_from_slice(&x);
        let x = self.embed.convnext(&rows, tc + 6);
        let mut e = self.embed.finish(&x, tc);
        let lv = &self.lang_vec[lang];
        e.chunks_exact_mut(lv.len()).for_each(|r| add(r, lv));
        self.encode(e, tc, None, &[])
    }

    fn decoder_out(&self, ctx: &[u32], h: &[f32]) -> Vec<f32> {
        let mut y = self.decoder_y(ctx);
        if let Some(l) = &self.lstm {
            let p = l.proj.apply(h, 1);
            y.iter_mut().zip(&p).for_each(|(a, b)| *a += l.gate * b);
        }
        self.joiner.dec.apply(&y, 1)
    }

    /// `advance` for several hypotheses at once: the LSTM, its projection and the joiner's decoder projection run as
    /// one batch.
    fn advance_batch(&self, decs: &mut [&mut Decoding], ids: &[u32]) {
        let n = decs.len();
        if n == 0 {
            return;
        }
        let d = self.dec_dim;
        let mut ys: Vec<f32> = Vec::with_capacity(n * d);
        for (dec, &id) in decs.iter_mut().zip(ids) {
            self.push_ctx(dec, id);
            self.decoder_y_into(&dec.ctx, &mut ys);
        }
        if let Some(l) = &self.lstm {
            let hn = l.hidden;
            let mut hp = Vec::with_capacity(n * hn);
            for dec in decs.iter() {
                hp.extend_from_slice(&dec.h);
            }
            let mut g = self.lstm_in(l, ids);
            let r = l.hh.apply(&hp, n);
            g.iter_mut().zip(&r).for_each(|(a, b)| *a += b);
            for (b, dec) in decs.iter_mut().enumerate() {
                let dec = &mut **dec;
                crate::kernels::lstm_cell(&g[b * 4 * hn..(b + 1) * 4 * hn], &mut dec.c, &mut dec.h);
            }
            let mut hs = Vec::with_capacity(n * hn);
            decs.iter().for_each(|dec| hs.extend_from_slice(&dec.h));
            let p = l.proj.apply(&hs, n);
            ys.iter_mut().zip(&p).for_each(|(a, b)| *a += l.gate * b);
        }
        let o = self.joiner.dec.apply(&ys, n);
        let jd = o.len() / n;
        for (b, dec) in decs.iter_mut().enumerate() {
            dec.out = o[b * jd..(b + 1) * jd].to_vec();
        }
    }

    /// Joiner hidden of every hypothesis at encoder frame i (rows), from the projected encoder output `ep`.
    fn joint_rows(&self, ep: &[f32], i: usize, decs: &[&Decoding]) -> Vec<f32> {
        let jd = self.joiner.enc.n();
        let mut h = vec![0.0; decs.len() * jd];
        for (dec, row) in decs.iter().zip(h.chunks_exact_mut(jd)) {
            crate::kernels::tanh_sum(&ep[i * jd..(i + 1) * jd], &dec.out, row);
        }
        h
    }

    /// Emit `id`: extend the history, advance the LSTM, recompute the decoder output.
    fn advance(&self, dec: &mut Decoding, id: u32) {
        self.push_ctx(dec, id);
        if let Some(l) = &self.lstm {
            let mut g = self.lstm_in(l, &[id]);
            let r = l.hh.apply(&dec.h, 1);
            g.iter_mut().zip(&r).for_each(|(a, b)| *a += b);
            let n = l.hidden;
            crate::kernels::lstm_cell(&g, &mut dec.c[..n], &mut dec.h[..n]);
        }
        dec.out = self.decoder_out(&dec.ctx, &dec.h);
    }


    /// Append `id` to the decoder context, keeping only the last `context` tokens (all the decoder reads).
    fn push_ctx(&self, dec: &mut Decoding, id: u32) {
        dec.ctx.push(id);
        if dec.ctx.len() > self.context {
            dec.ctx.remove(0);
        }
    }

    /// LSTM input products ih(embedding(id)) + bias for `ids` (n x 4 hidden): from the per-token cache, the missing
    /// ones computed as one batch.
    fn lstm_in(&self, l: &Lstm, ids: &[u32]) -> Vec<f32> {
        let d = self.dec_dim;
        #[cfg(feature = "std")]
        if !self.ih_cache.is_empty() {
            let mut miss: Vec<u32> = Vec::new();
            for &id in ids {
                if self.ih_cache[id as usize].get().is_none() && !miss.contains(&id) {
                    miss.push(id);
                }
            }
            if !miss.is_empty() {
                let mut x = Vec::with_capacity(miss.len() * d);
                for &id in &miss {
                    x.extend_from_slice(&self.dec_emb[id as usize * d..(id as usize + 1) * d]);
                }
                let g = l.ih.apply(&x, miss.len());
                let w = g.len() / miss.len();
                for (k, &id) in miss.iter().enumerate() {
                    let _ = self.ih_cache[id as usize].set(g[k * w..(k + 1) * w].into());
                }
            }
            let mut g = Vec::with_capacity(ids.len() * 4 * l.hidden);
            for &id in ids {
                g.extend_from_slice(self.ih_cache[id as usize].get().unwrap());
            }
            return g;
        }
        let mut x = Vec::with_capacity(ids.len() * d);
        for &id in ids {
            x.extend_from_slice(&self.dec_emb[id as usize * d..(id as usize + 1) * d]);
        }
        l.ih.apply(&x, ids.len())
    }

    /// `decoder_y` appended to `out`; with a two-token context, memoized per thread by the token pair (exact).
    fn decoder_y_into(&self, ctx: &[u32], out: &mut Vec<f32>) {
        #[cfg(feature = "std")]
        if self.context == 2 {
            use std::cell::RefCell;
            use std::collections::HashMap;
            std::thread_local! {
                static Y: RefCell<(usize, HashMap<u64, alloc::boxed::Box<[f32]>>)> = RefCell::new((0, HashMap::new()));
            }
            let key = ((ctx[ctx.len() - 2] as u64) << 32) | ctx[ctx.len() - 1] as u64;
            // a cache belongs to one model; not its address, which a model loaded after this one is dropped may reuse
            let me = self.uid;
            return Y.with(|y| {
                let mut y = y.borrow_mut();
                if y.0 != me || y.1.len() >= 4096 {
                    y.1.clear();
                    y.0 = me;
                }
                let v = y.1.entry(key).or_insert_with(|| self.decoder_y(ctx).into());
                out.extend_from_slice(v);
            });
        }
        out.extend(self.decoder_y(ctx));
    }

    /// Decoder output before the joiner's projection (dec_dim), for the last `context` tokens of `ctx`.
    fn decoder_y(&self, ctx: &[u32]) -> Vec<f32> {
        let d = self.dec_dim;
        let c = self.context;
        let mut y = vec![0.0; d];
        for o in 0..d {
            let g = o / 4;
            let mut a = 0.0;
            for i in 0..4 {
                for (k, &tok) in ctx[ctx.len() - c..].iter().enumerate() {
                    a += self.dec_conv[(o * 4 + i) * c + k] * self.dec_emb[tok as usize * d + g * 4 + i];
                }
            }
            y[o] = a.max(0.0);
        }
        y
    }

    /// Greedy RNN-T over encoder frames, continuing from `dec`.
    fn greedy(&self, enc: &[f32], t: usize, lang: usize, dec: &mut Decoding, frame0: u32) -> Vec<Token> {
        if let Some(dur) = &self.dur {
            return self.greedy_tdt(dur, enc, t, lang, dec, frame0);
        }
        let ep = self.joiner.enc.apply(enc, t);
        let jd = self.joiner.enc.n();
        let (out, ids) = &self.joiner.out[lang];
        let mut toks = Vec::new();
        let mut h = vec![0.0; jd];
        for i in 0..t {
            // up to max_symbols() tokens per 40 ms frame (fast speech needs more than one)
            for _ in 0..max_symbols() {
                for (k, v) in h.iter_mut().enumerate() {
                    *v = tanh(ep[i * jd + k] + dec.out[k]);
                }
                let logit = out.apply(&h, 1);
                let (best, &m) = logit.iter().enumerate().fold((0, &f32::MIN), |a, b| if b.1 > a.1 { b } else { a });
                let id = ids[best];
                if id == 0 {
                    break;
                }
                let lse = m + logit.iter().map(|v| exp(v - m)).sum::<f32>().ln();
                toks.push(Token { id, frame: frame0 + i as u32, logp: m - lse });
                self.advance(dec, id);
            }
        }
        toks
    }

    /// modified_beam_search (icefall beam_search.py): per frame, every hypothesis is extended by its log-softmax over the
    /// language's tokens; the `beam` best (hypothesis, token) pairs survive, blank keeping the text; at most one token
    /// per frame; hypotheses with the same tokens are merged (log-add). Continues from `hyps`.
    fn beam_rnnt(&self, enc: &[f32], t: usize, lang: usize, hyps: &mut Vec<Hyp>, frame0: u32, beam: usize) {
        let ep = self.joiner.enc.apply(enc, t);
        let (out, ids) = &self.joiner.out[lang];
        for i in 0..t {
            let mut cand: Vec<(f32, usize, usize, f32)> = Vec::new(); // (score, hyp, output index, token logp)
            let nh = hyps.len();
            let hrows = self.joint_rows(&ep, i, &hyps.iter().map(|h| &*h.dec).collect::<Vec<_>>());
            let lg = out.apply(&hrows, nh);
            let vn = lg.len() / nh;
            for (hi, hy) in hyps.iter().enumerate() {
                let logit = &lg[hi * vn..(hi + 1) * vn];
                let m = logit.iter().cloned().fold(f32::MIN, f32::max);
                let lse = m + logit.iter().map(|v| exp(v - m)).sum::<f32>().ln();
                for (k, &v) in logit.iter().enumerate() {
                    cand.push((hy.logp + v - lse, hi, k, v - lse));
                }
            }
            let n = beam.min(cand.len());
            cand.select_nth_unstable_by(n - 1, |a, b| b.0.total_cmp(&a.0));
            cand.truncate(n);
            let mut next: Vec<Hyp> = Vec::with_capacity(n);
            for &(score, hi, k, lp) in &cand {
                let id = ids[k];
                let mut hy = hyps[hi].clone();
                hy.logp = score;
                if id != 0 {
                    hy.toks.push(Token { id, frame: frame0 + i as u32, logp: lp });
                    hy.pending = true;
                }
                if let Some(o) = next.iter_mut().find(|o| o.toks.len() == hy.toks.len() && o.toks.iter().zip(&hy.toks).all(|(a, b)| a.id == b.id)) {
                    let (a, b) = (o.logp.max(hy.logp), o.logp.min(hy.logp));
                    o.logp = a + (1.0 + exp(b - a)).ln();
                } else {
                    next.push(hy);
                }
            }
            let pend: Vec<u32> = next.iter().filter(|h| h.pending).map(|h| h.toks.last().unwrap().id).collect();
            let mut decs: Vec<&mut Decoding> = next.iter_mut().filter(|h| h.pending).map(|h| { h.pending = false; alloc::sync::Arc::make_mut(&mut h.dec) }).collect();
            self.advance_batch(&mut decs, &pend);
            *hyps = next;
        }
    }

    fn beam_tdt(&self, dur: &Linear, enc: &[f32], t: usize, lang: usize, beam: usize) -> (Vec<Token>, f32) {
        let ep = self.joiner.enc.apply(enc, t);
        let (out, ids) = &self.joiner.out[lang];
        let lse = crate::kernels::logsumexp;
        // reused per hypothesis: its best tokens (logit, index) and (score, token, duration) candidates
        let mut top: Vec<(f32, usize)> = Vec::with_capacity(beam + 1);
        let mut cand: Vec<(f32, usize, usize)> = Vec::new();
        let same = |a: &Hyp, b: &Hyp| a.toks.len() == b.toks.len() && a.toks.iter().zip(&b.toks).all(|(x, y)| x.id == y.id);
        // merge `hy` into `list` (log-add on equal text)
        let push = |list: &mut Vec<Hyp>, hy: Hyp| {
            if let Some(o) = list.iter_mut().find(|o| same(o, &hy)) {
                let (a, b) = (o.logp.max(hy.logp), o.logp.min(hy.logp));
                o.logp = a + (1.0 + exp(b - a)).ln();
            } else {
                list.push(hy);
            }
        };
        let mut buckets: Vec<Vec<Hyp>> = (0..t).map(|_| Vec::new()).collect();
        let mut done: Vec<Hyp> = Vec::new();
        if t == 0 {
            return (Vec::new(), 0.0);
        }
        buckets[0] = self.start_beam();
        // decoder states advanced in this search, by token ids (with the frame they were made on, to forget old ones)
        let mut memo: alloc::collections::BTreeMap<Vec<u32>, (usize, alloc::sync::Arc<Decoding>)> = alloc::collections::BTreeMap::new();
        for i in 0..t {
            // a hypothesis waits at most the longest duration (one bucket per frame ahead): older states are not asked for
            memo.retain(|_, (f, _)| *f + MEMO_FRAMES >= i);
            let mut cur = core::mem::take(&mut buckets[i]);
            for _round in 0..=TDT_MAX_SYM {
                if cur.is_empty() {
                    break;
                }
                cur.sort_by(|a, b| b.logp.total_cmp(&a.logp));
                cur.truncate(beam);
                {  // decoder step for the survivors only, as one batch
                    // a text already advanced in this search (the same candidate kept with another duration, in another
                    // frame's bucket) reuses that decoder state: it depends on the tokens only
                    let mut todo: Vec<bool> = Vec::with_capacity(cur.len());
                    for h in cur.iter_mut() {
                        if h.pending {
                            if let Some((_, d)) = memo.get(&h.toks.iter().map(|t| t.id).collect::<Vec<u32>>()) {
                                h.dec = d.clone();
                                h.pending = false;
                            }
                        }
                        todo.push(h.pending);
                    }
                    let pend: Vec<u32> = cur.iter().filter(|h| h.pending).map(|h| h.toks.last().unwrap().id).collect();
                    let mut decs: Vec<&mut Decoding> = cur.iter_mut().filter(|h| h.pending).map(|h| { h.pending = false; alloc::sync::Arc::make_mut(&mut h.dec) }).collect();
                    self.advance_batch(&mut decs, &pend);
                    for (h, _) in cur.iter().zip(&todo).filter(|(_, t)| **t) {
                        memo.insert(h.toks.iter().map(|t| t.id).collect(), (i, h.dec.clone()));
                    }
                }
                let mut stay: Vec<Hyp> = Vec::new(); // zero-duration tokens: same frame, next round
                let nh = cur.len();
                let hrows = self.joint_rows(&ep, i, &cur.iter().map(|h| &*h.dec).collect::<Vec<_>>());
                let lgall = out.apply(&hrows, nh);
                let ldall = dur.apply(&hrows, nh);
                let (vn, dn) = (lgall.len() / nh, ldall.len() / nh);
                for (hi, hy) in cur.iter().enumerate() {
                    // log-softmax of tokens and durations, taken only where used
                    let (lt, ld) = (&lgall[hi * vn..(hi + 1) * vn], &ldall[hi * dn..(hi + 1) * dn]);
                    let (zt, zd) = (lse(lt), lse(ld));
                    // this hypothesis' best `beam` tokens: one pass keeping a short sorted list
                    let kk = beam.min(vn);
                    top.clear();
                    // blocks of 16 whose max cannot enter the list are skipped (the same list as a full scan)
                    let mut base = 0;
                    for blk in lt.chunks(16) {
                        // "some value above the list's last" (as "the block's max above it": the same blocks, but a
                        // branch-free compare over the block instead of a chain of 16 dependent max operations)
                        let enter = top.len() < kk || {
                            let thr = top[top.len() - 1].0;
                            let mut any = false;
                            for &v in blk {
                                any |= v > thr;
                            }
                            any
                        };
                        if enter {
                            for (j, &v) in blk.iter().enumerate() {
                                if top.len() < kk || v > top[top.len() - 1].0 {
                                    let at = top.iter().position(|e| v > e.0).unwrap_or(top.len());
                                    top.insert(at, (v, base + j));
                                    top.truncate(kk);
                                }
                            }
                        }
                        base += blk.len();
                    }
                    // its best `beam` (token, duration) pairs
                    cand.clear();
                    for &(v, k) in &top {
                        for (d, &l) in ld.iter().enumerate() {
                            if ids[k] == 0 && d == 0 {
                                continue;
                            }
                            cand.push((hy.logp + (v - zt) + (l - zd), k, d));
                        }
                    }
                    let n = beam.min(cand.len());
                    cand.select_nth_unstable_by(n - 1, |a, b| b.0.total_cmp(&a.0));
                    for &(score, k, d) in &cand[..n] {
                        let id = ids[k];
                        let list = if d == 0 {
                            &mut stay
                        } else if i + d >= t {
                            &mut done
                        } else {
                            &mut buckets[i + d]
                        };
                        // equal text already there: log-add (as `push`), without building the new hypothesis
                        let ext = id != 0;
                        let nt = hy.toks.len() + ext as usize;
                        let found = list.iter_mut().find(|o| {
                            o.toks.len() == nt
                                && (!ext || o.toks[nt - 1].id == id)
                                && o.toks.iter().zip(&hy.toks).all(|(x, y)| x.id == y.id)
                        });
                        if let Some(o) = found {
                            let (a, b) = (o.logp.max(score), o.logp.min(score));
                            o.logp = a + (1.0 + exp(b - a)).ln();
                            continue;
                        }
                        let mut nh = hy.clone();
                        nh.logp = score;
                        if ext {
                            nh.toks.push(Token { id, frame: i as u32, logp: lt[k] - zt });
                            nh.pending = true;
                        }
                        list.push(nh);
                    }
                }
                cur = stay;
            }
            for mut hy in cur {  // still on this frame after TDT_MAX_SYM zero-duration rounds: forced on by one frame
                if hy.pending {
                    let id = hy.toks.last().map(|x| x.id).unwrap_or(0);
                    self.advance(alloc::sync::Arc::make_mut(&mut hy.dec), id);
                    hy.pending = false;
                }
                if i + 1 >= t { push(&mut done, hy) } else { push(&mut buckets[i + 1], hy) }
            }
        }
        Model::best_scored(&done)
    }

    fn start_beam(&self) -> Vec<Hyp> {
        vec![Hyp { dec: alloc::sync::Arc::new(Decoding::new(self)), logp: 0.0, toks: Vec::new(), pending: false }]
    }

    /// The most probable hypothesis: its tokens and its log-probability (the whole path: tokens, blanks, durations).
    fn best_scored(hyps: &[Hyp]) -> (Vec<Token>, f32) {
        hyps.iter().max_by(|a, b| a.logp.total_cmp(&b.logp)).map_or((Vec::new(), 0.0), |h| (h.toks.clone(), h.logp))
    }

    /// TDT greedy (tdt_decode.py): after a token or blank the frame pointer moves on by the predicted duration (a blank
    /// by at least 1; after TDT_MAX_SYM zero-duration tokens in a row by 1). A jump past this chunk's last frame is
    /// carried into the next chunk (`dec.skip`).
    fn greedy_tdt(&self, dur: &Linear, enc: &[f32], t: usize, lang: usize, dec: &mut Decoding, frame0: u32) -> Vec<Token> {
        let ep = self.joiner.enc.apply(enc, t);
        let jd = self.joiner.enc.n();
        let (out, ids) = &self.joiner.out[lang];
        let mut toks = Vec::new();
        let mut h = vec![0.0; jd];
        let argmax = |v: &[f32], from: usize| (from..v.len()).fold(from, |a, j| if v[j] > v[a] { j } else { a });
        let mut i = dec.skip;
        while i < t {
            for (k, v) in h.iter_mut().enumerate() {
                *v = tanh(ep[i * jd + k] + dec.out[k]);
            }
            let logit = out.apply(&h, 1);
            let dl = dur.apply(&h, 1);
            let best = argmax(&logit, 0);
            let id = ids[best];
            let mut d;
            if id == 0 {
                d = argmax(&dl, 1); // a blank always moves on
                dec.run = 0;
            } else {
                d = argmax(&dl, 0);
                let m = logit[best];
                let lse = m + logit.iter().map(|v| exp(v - m)).sum::<f32>().ln();
                toks.push(Token { id, frame: frame0 + i as u32, logp: m - lse });
                self.advance(dec, id);
                dec.run = if d == 0 { dec.run + 1 } else { 0 };
                if dec.run >= TDT_MAX_SYM {
                    d = 1;
                    dec.run = 0;
                }
            }
            i += d;
        }
        dec.skip = i - t;
        toks
    }


    /// Second pass: full-context transcription of a finished segment (16 kHz); beam > 1: beam search of that width
    /// (RNN-T: modified_beam_search; TDT: over tokens and durations), else greedy. Also returns the log-probability of
    /// the best beam-search path (tokens, blanks, durations; natural log; None for greedy or audio too short to decode).
    pub fn full_scored(&self, audio: &[f32], lang: usize, beam: usize) -> (Vec<Token>, Option<f32>) {
        let fb = Fbank::new();
        let t = Fbank::num_frames(audio.len());
        if t < 16 {
            return (Vec::new(), None);
        }
        let feats = fb.frames(audio, 0, t);
        let (enc, te) = self.encode_full(&feats, t, lang);
        if beam > 1 {
            let (tk, lp) = if let Some(dur) = &self.dur {
                self.beam_tdt(dur, &enc, te, lang, beam)
            } else {
                let mut hyps = self.start_beam();
                self.beam_rnnt(&enc, te, lang, &mut hyps, 0, beam);
                Model::best_scored(&hyps)
            };
            return (tk, Some(lp));
        }
        let mut dec = Decoding::new(self);
        (self.greedy(&enc, te, lang, &mut dec, 0), None)
    }

    /// Second pass + P(the speaker has finished) from the end-of-turn head (None if the file has no head): the last
    /// HEAD_K encoder frames and the decoder output after the last text tokens (<eou>, id 12, skipped).
    pub fn full_head(&self, audio: &[f32], lang: usize) -> (Vec<Token>, Option<f32>) {
        let fb = Fbank::new();
        let t = Fbank::num_frames(audio.len());
        if t < 16 {
            return (Vec::new(), None);
        }
        let feats = fb.frames(audio, 0, t);
        let (enc, te) = self.encode_full(&feats, t, lang);
        let mut dec = Decoding::new(self);
        let toks = self.greedy(&enc, te, lang, &mut dec, 0);
        let p = self.head.as_ref().map(|h| {
            let d = enc.len() / te.max(1);
            let n = te.min(HEAD_K);
            let mut last = vec![0.0; HEAD_K * d];
            last[(HEAD_K - n) * d..].copy_from_slice(&enc[(te - n) * d..te * d]);
            let ids: Vec<u32> = toks.iter().map(|t| t.id).filter(|&i| i != 12).collect();
            let mut ctx = vec![0u32; self.context];
            ctx.extend(&ids);
            let mut y = self.decoder_y(&ctx);
            if let Some(l) = &self.lstm {  // hybrid decoder: + the LSTM output after [blank] + the text (eou_head.py)
                let dd = self.dec_dim;
                let (mut hs, mut cs) = (vec![0.0; l.hidden], vec![0.0; l.hidden]);
                for &i in [0].iter().chain(&ids) {
                    l.step(&self.dec_emb[i as usize * dd..(i as usize + 1) * dd], &mut hs, &mut cs);
                }
                let p = l.proj.apply(&hs, 1);
                y.iter_mut().zip(&p).for_each(|(a, b)| *a += l.gate * b);
            }
            h.prob(&last, n, &y)
        });
        (toks, p)
    }

    pub fn has_head(&self) -> bool {
        self.head.is_some()
    }

    pub fn stream(&self, lang: usize) -> Stream<'_> {
        Stream::new(self, lang)
    }

}

/// Decoder context (last `context` tokens) and its projected output.
#[derive(Clone)]
pub struct Decoding {
    ctx: Vec<u32>,
    out: Vec<f32>,
    h: Vec<f32>, // LSTM state (empty without the LSTM)
    c: Vec<f32>,
    skip: usize, // TDT: frames of the next chunk already jumped over
    run: usize,  // TDT: zero-duration tokens in a row on the current frame
}

impl Decoding {
    fn new(m: &Model) -> Decoding {
        let ctx = vec![0; m.context];
        let n = m.lstm.as_ref().map_or(0, |l| l.hidden);
        let (mut h, mut c) = (vec![0.0; n], vec![0.0; n]);
        if let Some(l) = &m.lstm {  // training runs the LSTM over [blank] + labels: the start state has seen blank
            l.step(&m.dec_emb[..m.dec_dim], &mut h, &mut c);
        }
        let out = m.decoder_out(&ctx, &h);
        Decoding { ctx, out, h, c, skip: 0, run: 0 }
    }
}

// ---------------------------------------------------------------------------------------------- streaming

/// Feature frames per chunk window and per step: 2 * chunk + 13 (7 for the convolutions, 6 for ConvNeXt), and
/// 2 * chunk.
pub struct Stream<'m> {
    m: &'m Model,
    lang: usize,
    fb: Fbank,
    audio: Vec<f32>,
    feats: Vec<f32>, // frames computed so far
    nfeat: usize,
    chunks: usize,
    embed_cache: Vec<f32>, // 3 x 19 x 128
    states: Vec<Vec<LayerState>>,
    processed: usize, // 50 Hz frames through the encoder
    dec: Decoding,
    pub tokens: Vec<Token>,
    finished: bool,
    /// Encoder output of every chunk so far (frames x 512), kept when `keep_enc` is set (tests).
    pub keep_enc: bool,
    pub enc: Vec<f32>,
}

impl<'m> Stream<'m> {
    fn new(m: &'m Model, lang: usize) -> Stream<'m> {
        let states = m
            .stacks
            .iter()
            .map(|st| {
                let left = m.left / st.ds;
                let d = st.dim;
                let l0 = &st.layers[0];
                let s = LayerState {
                    key: vec![0.0; left * l0.attn.heads * l0.attn.qd],
                    nonlin: vec![0.0; left * l0.nonlin.hid],
                    val1: vec![0.0; left * l0.sa1.inp.n()],
                    val2: vec![0.0; left * l0.sa2.inp.n()],
                    conv1: vec![0.0; (l0.conv1.kernel / 2) * d],
                    conv2: vec![0.0; (l0.conv2.kernel / 2) * d],
                };
                vec![s; st.layers.len()]
            })
            .collect();
        Stream {
            m,
            lang,
            fb: Fbank::new(),
            audio: Vec::new(),
            feats: Vec::new(),
            nfeat: 0,
            chunks: 0,
            embed_cache: vec![0.0; 3 * FW * C3],
            states,
            processed: 0,
            dec: Decoding::new(m),
            tokens: Vec::new(),
            finished: false,
            keep_enc: false,
            enc: Vec::new(),
        }
    }

    pub fn window(&self) -> usize {
        2 * self.m.chunk + 13
    }

    /// Feed 16 kHz audio; returns the tokens decoded from the chunks it completed.
    pub fn accept(&mut self, x: &[f32]) -> Vec<Token> {
        self.audio.extend_from_slice(x);
        let ready = if self.finished { Fbank::num_frames(self.audio.len()) } else { Fbank::num_ready(self.audio.len()) };
        if ready > self.nfeat {
            let f = self.fb.frames(&self.audio, self.nfeat, ready);
            self.feats.extend_from_slice(&f);
            self.nfeat = ready;
        }
        let mut out = Vec::new();
        let (win, step) = (self.window(), 2 * self.m.chunk);
        while self.chunks * step + win <= self.nfeat {
            let s = self.chunks * step;
            let t = self.chunk(&self.feats[s * BINS..(s + win) * BINS].to_vec());
            out.extend(t);
            self.chunks += 1;
        }
        self.tokens.extend_from_slice(&out);
        out
    }

    /// End of input: pad with silence so the last audio goes through (like sherpa-onnx's tail padding).
    pub fn finish(&mut self) -> Vec<Token> {
        let pad = vec![0.0; (self.window() * 160).max(16000)];
        let mut out = self.accept(&pad);
        self.finished = true;
        out.extend(self.accept(&[]));
        out
    }

    fn chunk(&mut self, feats: &[f32]) -> Vec<Token> {
        let m = self.m;
        let win = self.window();
        let (x, tc) = m.embed.convs(feats, win);
        let row = FW * C3;
        let mut rows = core::mem::take(&mut self.embed_cache);
        rows.extend_from_slice(&x);
        let t = tc - 3; // = chunk
        // cache (3 rows) + 35 new rows: the 7x7 time context comes from the window itself (13 frames overlap)
        let y = m.embed.convnext(&rows, tc + 3);
        self.embed_cache = rows[t * row..(t + 3) * row].to_vec();
        let mut e = m.embed.finish(&y, t);
        let lv = &m.lang_vec[self.lang];
        e.chunks_exact_mut(lv.len()).for_each(|r| add(r, lv));
        let left = m.left;
        let masked: Vec<bool> = (0..left).map(|j| j + self.processed < left).collect();
        self.processed += t;
        let (enc, te) = m.encode(e, t, Some(&mut self.states), &masked);
        if self.keep_enc {
            self.enc.extend_from_slice(&enc);
        }
        let frame0 = (self.chunks * t / 2) as u32;
        m.greedy(&enc, te, self.lang, &mut self.dec, frame0)
    }
}
