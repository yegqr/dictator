//! x86-64 code paths chosen at run time, so one binary built for plain `x86-64` runs the best kernels each CPU has:
//!
//! | level        | CPUs (examples)                                        | int8 product                         | f32 kernels |
//! |--------------|--------------------------------------------------------|--------------------------------------|-------------|
//! | `Avx512Vnni` | Ice/Tiger Lake, Sapphire Rapids, AMD Zen 4/5           | `vpdpbusd` zmm (in `kernels`)        | AVX-512     |
//! | `AvxVnni`    | Alder/Raptor/Meteor/Arrow Lake, Zen 5 without AVX-512  | `vpdpbusd` ymm (VEX, here)           | AVX2 + FMA  |
//! | `Avx2`       | Haswell .. Comet Lake, Zen 1-3, Skylake-X              | `vpmaddubsw` + `vpmaddwd` ymm (here) | AVX2 + FMA  |
//! | `Sse`        | anything else                                          | none: weights widened to f32         | upstream    |
//!
//! The AVX-512 kernels stay in `kernels.rs`; this module holds the 256-bit ones and the run-time choice. Env
//! overrides (std builds, for testing): `TINYASR_ISA=avx512vnni|avxvnni|avx2|sse` picks that level or the best lower one
//! the CPU has (never a level it lacks); `TINYASR_INT8=0` is the same as `TINYASR_ISA=sse` (the upstream code paths).
//!
//! All 256-bit int8 kernels use the weight layout of `kernels::PackI8` (16-column panels, 4 bytes of each column per 4
//! inputs = 64 bytes), handled as two 8-column halves, and the same arithmetic (activations x + 128 as unsigned bytes,
//! each group's sum started at -128 x the group's weight sum, scaled per group with one fused multiply-add), so their
//! results equal the AVX-512 path bit for bit. The f32 kernels repeat the AVX-512 arithmetic 8 lanes at a time (same
//! results, except sums over a row - softmax, logsumexp - which add in a different order).

#![allow(unsafe_code)]

use core::arch::x86_64::*;

#[cfg(not(feature = "std"))]
use crate::compat::F32Ext;

/// The code path level, best last.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Isa {
    Sse,
    Avx2,
    AvxVnni,
    Avx512Vnni,
}

/// What the CPU has (cached).
#[derive(Clone, Copy)]
struct Cpu {
    isa: Isa,
    /// The VEX `vpdpbusd` (AVX-VNNI). Without it the `AvxVnni` level (only when forced by TINYASR_ISA on an AVX-512 VNNI
    /// CPU) runs the same 256-bit kernel with the EVEX encoding (AVX-512 VNNI + VL), so the 256-bit path
    /// can be tested on AVX-512 machines.
    vex_vnni: bool,
}

#[cfg(feature = "std")]
fn cpu() -> Cpu {
    static C: std::sync::OnceLock<(Isa, bool)> = std::sync::OnceLock::new();
    let &(isa, vex_vnni) = C.get_or_init(|| {
        let avx2 = std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
        let a512 = avx2
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni");
        let vex = avx2 && std::arch::is_x86_feature_detected!("avxvnni");
        let evex256 = a512 && std::arch::is_x86_feature_detected!("avx512vl");
        let has = |l: Isa| match l {
            Isa::Sse => true,
            Isa::Avx2 => avx2,
            Isa::AvxVnni => vex || evex256,
            Isa::Avx512Vnni => a512,
        };
        let want = match std::env::var("TINYASR_INT8").as_deref() {
            Ok("0") => Isa::Sse,
            _ => match std::env::var("TINYASR_ISA").as_deref() {
                Ok("sse") | Ok("scalar") | Ok("none") => Isa::Sse,
                Ok("avx2") => Isa::Avx2,
                Ok("avxvnni") => Isa::AvxVnni,
                _ => Isa::Avx512Vnni,
            },
        };
        let best = [Isa::Avx512Vnni, Isa::AvxVnni, Isa::Avx2, Isa::Sse].into_iter().find(|&l| l <= want && has(l)).unwrap_or(Isa::Sse);
        // the natural AVX-VNNI level (not forced) needs the VEX instruction; an AVX-512 CPU picks Avx512Vnni first
        if std::env::var_os("TINYASR_ISA_LOG").is_some() {
            std::eprintln!("tinyasr x86 level: {best:?} (vex vnni {vex})");
        }
        (best, vex)
    });
    Cpu { isa, vex_vnni }
}

#[cfg(not(feature = "std"))]
fn cpu() -> Cpu {
    let avx2 = cfg!(all(target_feature = "avx2", target_feature = "fma"));
    let isa = if avx2 && cfg!(all(target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512vnni")) {
        Isa::Avx512Vnni
    } else if avx2 && cfg!(target_feature = "avxvnni") {
        Isa::AvxVnni
    } else if avx2 {
        Isa::Avx2
    } else {
        Isa::Sse
    };
    Cpu { isa, vex_vnni: cfg!(target_feature = "avxvnni") }
}

/// The code path level of this run.
pub fn isa() -> Isa {
    cpu().isa
}

/// The 256-bit paths of this module (AVX2 + FMA at least, not AVX-512): the f32 kernels here run.
#[inline]
pub fn ymm() -> bool {
    matches!(isa(), Isa::Avx2 | Isa::AvxVnni)
}

// ------------------------------------------------------------------------------------------------- int8 product

/// For the AVX2 path (`vpmaddubsw`, which saturates its i16 pair sums): how many 4-input steps of `vpmaddubsw` results
/// may be added in i16 before widening to i32, for weights `q` in groups of `g`; 0 only if `g` is not a multiple of 4
/// (then the tensor is not packed). A pair sum is at most 255 (|w1| + |w2|) <= 510 max|w| (int4 weights: 4080), so s steps stay
/// exact while s x 510 max|w| <= 32767; s is a power of two dividing g / 4.
/// Larger weights (int8, max|w| > 64) get `SPLIT` | s: the activations are split into 4-bit halves (a = 16 hi + lo),
/// each multiplied by its own `vpmaddubsw` (pair sums <= 2 x 15 x 128 = 3840, so s <= 8 steps stay exact).
pub fn i16_steps(q: &[i8], g: usize) -> usize {
    let maxw = q.iter().map(|&w| (w as i32).abs()).max().unwrap_or(0).max(1) as usize;
    if g % 4 != 0 || g < 4 {
        return 0;
    }
    let mut limit = 32767 / (510 * maxw);
    let split = if limit == 0 {
        limit = 32767 / (30 * maxw);
        SPLIT
    } else {
        0
    };
    let mut s = 1;
    while s * 2 <= limit && (g / 4) % (s * 2) == 0 {
        s *= 2;
    }
    split | s
}

/// Flag in the `i16_steps` value: the AVX2 product splits the activations into 4-bit halves.
pub const SPLIT: usize = 1 << 16;

/// Whether a `PackI8` with these `i16_steps` multiplies on this level (always on the VNNI levels; AVX2 needs steps > 0).
pub fn pack_usable(steps: usize) -> bool {
    match isa() {
        Isa::Avx512Vnni | Isa::AvxVnni => true,
        Isa::Avx2 => steps > 0,
        Isa::Sse => false,
    }
}

/// Activation rows per 256-bit tile and rows per cache block (as `kernels::PR` / `PB`).
const PR: usize = 4;
const PB: usize = 64;

/// `kernels::pk_block` for the 256-bit levels: out (rows x n, row stride n) = unsigned activations (rows x k, see
/// `kernels::quantize_rows_u8`) x packed weights (`w`, `sw`, `ws` as `kernels::PackI8`) (+ bias), rows r0..r1 and
/// 16-column panels p_lo..p_hi only.
/// Safety: `out` points at r1 x n writable floats, and no other thread touches this block meanwhile.
#[allow(clippy::too_many_arguments)]
pub unsafe fn pk_block(w: &[i8], sw: &[f32], ws: &[i32], steps: usize, aq: &[u8], sa: &[f32], r0: usize, r1: usize, p_lo: usize, p_hi: usize, n: usize, k: usize, g: usize, bias: Option<&[f32]>, out: *mut f32) {
    let c = cpu();
    assert!(matches!(c.isa, Isa::Avx2 | Isa::AvxVnni), "pk_block: no 256-bit int8 path on this CPU");
    let (np, ng) = (n.div_ceil(16), k / g);
    assert!(g >= 4 && g % 4 == 0 && k % g == 0 && r0 <= r1 && p_lo <= p_hi && p_hi <= np);
    assert!(aq.len() >= r1 * k && sa.len() >= r1 * ng);
    assert!(w.len() >= np * k * 16 && sw.len() >= np * ng * 16 && ws.len() >= np * ng * 16);
    assert!(bias.is_none_or(|b| b.len() >= n));
    if c.isa == Isa::Avx2 {
        let s = steps & !SPLIT;
        assert!(s > 0 && s <= 64 && (g / 4) % s == 0, "pk_block: weights too large for the AVX2 int8 path");
    }
    for i0 in (r0..r1).step_by(PB) {
        let i1 = (i0 + PB).min(r1);
        for p in p_lo..p_hi {
            for h in 0..2 {
                let j0 = p * 16 + h * 8;
                if j0 >= n {
                    break;
                }
                let cols = (n - j0).min(8);
                let mut i = i0;
                while i < i1 {
                    let r = (i1 - i).min(PR);
                    // SAFETY: rows i..i+r < r1, panel p < np and its half h, columns j0..j0+cols <= n: all inside the
                    // slices asserted above and the r1 x n output the caller vouches for. The level was checked above.
                    unsafe {
                        let t = Tile {
                            a: aq.as_ptr().add(i * k),
                            sa: sa.as_ptr().add(i * ng),
                            w: w.as_ptr().add(p * k * 16 + h * 32),
                            sw: sw.as_ptr().add(p * ng * 16 + h * 8),
                            ws: ws.as_ptr().add(p * ng * 16 + h * 8),
                            k,
                            g,
                            steps,
                            out: out.add(i * n + j0),
                            n,
                            cols,
                            bias: bias.map(|b| b.as_ptr().add(j0)),
                        };
                        match (c.isa, c.vex_vnni, r) {
                            (Isa::Avx2, _, 4) => tile_avx2::<4>(&t),
                            (Isa::Avx2, _, 3) => tile_avx2::<3>(&t),
                            (Isa::Avx2, _, 2) => tile_avx2::<2>(&t),
                            (Isa::Avx2, _, _) => tile_avx2::<1>(&t),
                            (_, true, 4) => tile_vex::<4>(&t),
                            (_, true, 3) => tile_vex::<3>(&t),
                            (_, true, 2) => tile_vex::<2>(&t),
                            (_, true, _) => tile_vex::<1>(&t),
                            (_, false, 4) => tile_evex::<4>(&t),
                            (_, false, 3) => tile_evex::<3>(&t),
                            (_, false, 2) => tile_evex::<2>(&t),
                            (_, false, _) => tile_evex::<1>(&t),
                        }
                    }
                    i += r;
                }
            }
        }
    }
}

/// One tile: R activation rows x one 8-column half panel. Pointers at the tile's first row / column; the weights at the
/// half's first 4-byte column of input 0 (inputs 4 apart are 64 bytes apart), scales and sums at group 0 (groups 16
/// apart).
struct Tile {
    a: *const u8,
    sa: *const f32,
    w: *const i8,
    sw: *const f32,
    ws: *const i32,
    k: usize,
    g: usize,
    steps: usize,
    out: *mut f32,
    n: usize,
    cols: usize,
    bias: Option<*const f32>,
}

/// Stores R rows of 8 results (+ bias) at t.out (row stride n), only `cols` lanes.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn store_rows<const R: usize>(t: &Tile, accf: &[__m256; R]) {
    // SAFETY: the caller guarantees R rows of `cols` writable floats at t.out (stride n) and `cols` bias values.
    unsafe {
        let b = match t.bias {
            Some(b) if t.cols == 8 => _mm256_loadu_ps(b),
            Some(b) => {
                let mut l = [0f32; 8];
                core::ptr::copy_nonoverlapping(b, l.as_mut_ptr(), t.cols);
                _mm256_loadu_ps(l.as_ptr())
            }
            None => _mm256_setzero_ps(),
        };
        for r in 0..R {
            let v = _mm256_add_ps(accf[r], b);
            let o = t.out.add(r * t.n);
            if t.cols == 8 {
                _mm256_storeu_ps(o, v);
            } else {
                let mut l = [0f32; 8];
                _mm256_storeu_ps(l.as_mut_ptr(), v);
                core::ptr::copy_nonoverlapping(l.as_ptr(), o, t.cols);
            }
        }
    }
}

/// AVX2 tile: per 4 inputs and row, `vpmaddubsw` (unsigned activation bytes x signed weight bytes, adjacent pairs added
/// into i16) added up in i16 for `steps` steps (exact, see `i16_steps`), then `vpmaddwd` by 1 widens pairs to i32.
#[target_feature(enable = "avx2,fma")]
unsafe fn tile_avx2<const R: usize>(t: &Tile) {
    if t.steps & SPLIT != 0 {
        // SAFETY: the same tile and level (AVX2 + FMA), only the arithmetic differs.
        return unsafe { tile_avx2_split::<R>(t) };
    }
    let (k, g) = (t.k, t.g);
    let ng = k / g;
    // SAFETY: reads R rows of k bytes at t.a and ng scales at t.sa; weights at t.w + kk 16 + 0..32 (kk < k), scales /
    // sums at group c 16 + 0..8 (c < ng): inside the panel (asserted by `pk_block`).
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut accf = [_mm256_setzero_ps(); R];
        for c in 0..ng {
            let init = _mm256_loadu_si256(t.ws.add(c * 16).cast());
            let mut acci = [init; R];
            let mut kk = c * g;
            while kk < (c + 1) * g {
                let mut s16 = [_mm256_setzero_si256(); R];
                for s in 0..t.steps {
                    let q = kk + 4 * s;
                    let wv = _mm256_loadu_si256(t.w.add(q * 16).cast());
                    for r in 0..R {
                        let av = _mm256_set1_epi32(core::ptr::read_unaligned(t.a.add(r * k + q) as *const i32));
                        s16[r] = _mm256_add_epi16(s16[r], _mm256_maddubs_epi16(av, wv));
                    }
                }
                for r in 0..R {
                    acci[r] = _mm256_add_epi32(acci[r], _mm256_madd_epi16(s16[r], ones));
                }
                kk += 4 * t.steps;
            }
            let s = _mm256_loadu_ps(t.sw.add(c * 16));
            for r in 0..R {
                let sc = _mm256_mul_ps(s, _mm256_set1_ps(*t.sa.add(r * ng + c)));
                accf[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acci[r]), sc, accf[r]);
            }
        }
        store_rows::<R>(t, &accf);
    }
}

/// The VNNI 256-bit tile: one `vpdpbusd` per 4 inputs and row. `$dp` is the VEX (AVX-VNNI) or EVEX (AVX-512 VNNI + VL)
/// form of the same instruction.
/// `tile_avx2` for int8 weights (max|w| up to 128): each activation byte a = 16 hi + lo (4-bit halves), both halves
/// multiplied by `vpmaddubsw` and summed in i16 for `steps` steps (exact, see `i16_steps`), then widened by `vpmaddwd`
/// with 1 and 16. The same integer sums as the VNNI paths, so the same results.
#[target_feature(enable = "avx2,fma")]
unsafe fn tile_avx2_split<const R: usize>(t: &Tile) {
    let (k, g) = (t.k, t.g);
    let ng = k / g;
    let steps = t.steps & !SPLIT;
    // SAFETY: the reads of `tile_avx2` (weights at (kk + 4 s) 16 .. + 32 inside the panel half, 4 activation bytes at
    // r k + kk + 4 s < r k + k, scales and sums of group c < ng), checked by `pk_block`.
    unsafe {
        let (ones, sixteen, nib) = (_mm256_set1_epi16(1), _mm256_set1_epi16(16), _mm256_set1_epi8(15));
        let mut accf = [_mm256_setzero_ps(); R];
        for c in 0..ng {
            let init = _mm256_loadu_si256(t.ws.add(c * 16).cast());
            let mut acci = [init; R];
            let mut kk = c * g;
            while kk < (c + 1) * g {
                let mut lo = [_mm256_setzero_si256(); R];
                let mut hi = [_mm256_setzero_si256(); R];
                for s in 0..steps {
                    let q = kk + 4 * s;
                    let wv = _mm256_loadu_si256(t.w.add(q * 16).cast());
                    for r in 0..R {
                        let av = _mm256_set1_epi32(core::ptr::read_unaligned(t.a.add(r * k + q) as *const i32));
                        let (al, ah) = (_mm256_and_si256(av, nib), _mm256_and_si256(_mm256_srli_epi16(av, 4), nib));
                        lo[r] = _mm256_add_epi16(lo[r], _mm256_maddubs_epi16(al, wv));
                        hi[r] = _mm256_add_epi16(hi[r], _mm256_maddubs_epi16(ah, wv));
                    }
                }
                for r in 0..R {
                    let sum = _mm256_add_epi32(_mm256_madd_epi16(lo[r], ones), _mm256_madd_epi16(hi[r], sixteen));
                    acci[r] = _mm256_add_epi32(acci[r], sum);
                }
                kk += 4 * steps;
            }
            let s = _mm256_loadu_ps(t.sw.add(c * 16));
            for r in 0..R {
                let sc = _mm256_mul_ps(s, _mm256_set1_ps(*t.sa.add(r * ng + c)));
                accf[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acci[r]), sc, accf[r]);
            }
        }
        store_rows::<R>(t, &accf);
    }
}

macro_rules! tile_vnni256 {
    ($name:ident, $feat:literal, $dp:ident) => {
        #[target_feature(enable = $feat)]
        unsafe fn $name<const R: usize>(t: &Tile) {
            let (k, g) = (t.k, t.g);
            let ng = k / g;
            // SAFETY: as `tile_avx2`: R rows of k bytes / ng scales, weights and group values inside the panel.
            unsafe {
                let mut accf = [_mm256_setzero_ps(); R];
                for c in 0..ng {
                    let init = _mm256_loadu_si256(t.ws.add(c * 16).cast());
                    let mut acci = [init; R];
                    let mut kk = c * g;
                    while kk < (c + 1) * g {
                        let wv = _mm256_loadu_si256(t.w.add(kk * 16).cast());
                        for r in 0..R {
                            let av = _mm256_set1_epi32(core::ptr::read_unaligned(t.a.add(r * k + kk) as *const i32));
                            acci[r] = $dp(acci[r], av, wv);
                        }
                        kk += 4;
                    }
                    let s = _mm256_loadu_ps(t.sw.add(c * 16));
                    for r in 0..R {
                        let sc = _mm256_mul_ps(s, _mm256_set1_ps(*t.sa.add(r * ng + c)));
                        accf[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acci[r]), sc, accf[r]);
                    }
                }
                store_rows::<R>(t, &accf);
            }
        }
    };
}
tile_vnni256!(tile_vex, "avx2,fma,avxvnni", _mm256_dpbusd_avx_epi32);
tile_vnni256!(tile_evex, "avx2,fma,avx512f,avx512vl,avx512vnni", _mm256_dpbusd_epi32);

/// `kernels::quantize_u8_into` with AVX2 (the same arithmetic, 8 values at a time). Returns false (nothing done) when
/// this level has no AVX2 or `g` is not a multiple of 8.
pub fn quantize_u8(a: &[f32], g: usize, q: &mut [u8], s: &mut [f32]) -> bool {
    if !ymm() || g == 0 || g % 8 != 0 {
        return false;
    }
    assert!(q.len() >= a.len() && s.len() >= a.len() / g);
    // SAFETY: AVX2 + FMA checked by `ymm()`; sizes asserted above.
    unsafe { quantize_u8_avx2(a, g, q, s) };
    true
}

#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_u8_avx2(a: &[f32], g: usize, q: &mut [u8], s: &mut [f32]) {
    // SAFETY: group gi reads a[gi g .. gi g + g] (a.chunks_exact), writes q[gi g ..][..g] and s[gi]; q.len() >= a.len()
    // and s.len() >= a.len() / g (checked by the caller).
    unsafe {
        let half = _mm256_set1_ps(0.5);
        let signm = _mm256_set1_epi32(i32::MIN);
        let absm = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MAX));
        let off = _mm256_set1_epi32(128);
        let gather = _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0);
        for (gi, x) in a.chunks_exact(g).enumerate() {
            let p = x.as_ptr();
            let mut mx = _mm256_setzero_ps();
            let mut i = 0;
            while i < g {
                mx = _mm256_max_ps(mx, _mm256_and_ps(_mm256_loadu_ps(p.add(i)), absm));
                i += 8;
            }
            let max = hmax(mx);
            *s.get_unchecked_mut(gi) = max / 127.0;
            let inv = _mm256_set1_ps(if max > 0.0 { 127.0 / max } else { 0.0 });
            let o = q.as_mut_ptr().add(gi * g);
            let mut i = 0;
            while i < g {
                let r = _mm256_mul_ps(_mm256_loadu_ps(p.add(i)), inv);
                // r + copysign(0.5, r), truncated toward zero, + 128: 1..=255
                let h = _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(half), _mm256_and_si256(_mm256_castps_si256(r), signm)));
                let v = _mm256_add_epi32(_mm256_cvttps_epi32(_mm256_add_ps(r, h)), off);
                // 8 x i32 -> 8 bytes: pack within 128-bit lanes, then the two lanes' first 4 bytes side by side
                let b = _mm256_packus_epi16(_mm256_packs_epi32(v, v), _mm256_setzero_si256());
                let b = _mm256_permutevar8x32_epi32(b, gather);
                _mm_storel_epi64(o.add(i).cast(), _mm256_castsi256_si128(b));
                i += 8;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------- f32 kernels

/// Max of 8 lanes.
#[target_feature(enable = "avx2,fma")]
#[inline]
fn hmax(v: __m256) -> f32 {
    let m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    let m = _mm_max_ss(m, _mm_shuffle_ps::<1>(m, m));
    _mm_cvtss_f32(m)
}

/// Sum of 8 lanes.
#[target_feature(enable = "avx2,fma")]
#[inline]
fn hsum(v: __m256) -> f32 {
    let m = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let m = _mm_add_ps(m, _mm_movehl_ps(m, m));
    let m = _mm_add_ss(m, _mm_shuffle_ps::<1>(m, m));
    _mm_cvtss_f32(m)
}

/// exp of 8 lanes, the arithmetic of `kernels::exp` (clamp, 2^floor, degree-5 polynomial, no fused multiply-add).
#[target_feature(enable = "avx2,fma")]
#[inline]
fn exp8(x: __m256) -> __m256 {
    let t = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(x, _mm256_set1_ps(core::f32::consts::LOG2_E)), _mm256_set1_ps(-126.0)), _mm256_set1_ps(126.0));
    let fi = _mm256_floor_ps(t);
    let f = _mm256_sub_ps(t, fi);
    let mut q = _mm256_mul_ps(f, _mm256_set1_ps(0.001_333_355));
    q = _mm256_mul_ps(f, _mm256_add_ps(_mm256_set1_ps(0.009_618_129), q));
    q = _mm256_mul_ps(f, _mm256_add_ps(_mm256_set1_ps(0.055_504_11), q));
    q = _mm256_mul_ps(f, _mm256_add_ps(_mm256_set1_ps(0.240_226_5), q));
    #[allow(clippy::approx_constant)]
    let c0 = 0.693_147_2;
    q = _mm256_mul_ps(f, _mm256_add_ps(_mm256_set1_ps(c0), q));
    q = _mm256_add_ps(_mm256_set1_ps(1.0), q);
    let pw = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(_mm256_cvttps_epi32(fi), _mm256_set1_epi32(127))));
    _mm256_mul_ps(pw, q)
}

/// tanh as `kernels::tanh`: 1 - 2 / (exp(2x) + 1).
#[target_feature(enable = "avx2,fma")]
#[inline]
fn tanh8(x: __m256) -> __m256 {
    let e = exp8(_mm256_mul_ps(_mm256_set1_ps(2.0), x));
    _mm256_sub_ps(_mm256_set1_ps(1.0), _mm256_div_ps(_mm256_set1_ps(2.0), _mm256_add_ps(e, _mm256_set1_ps(1.0))))
}

/// sigmoid as `kernels::sigmoid`: 1 / (1 + exp(-x)).
#[target_feature(enable = "avx2,fma")]
#[inline]
fn sigmoid8(x: __m256) -> __m256 {
    let e = exp8(_mm256_sub_ps(_mm256_setzero_ps(), x));
    _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_add_ps(_mm256_set1_ps(1.0), e))
}

/// `kernels::swoosh` for x[..len / 8 * 8]; returns how many values it did (0 without AVX2).
pub fn swoosh(x: &mut [f32], shift: f32, offset: f32) -> usize {
    if !ymm() {
        return 0;
    }
    // SAFETY: AVX2 + FMA checked by `ymm()`.
    unsafe { swoosh_avx2(x, shift, offset) }
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::approx_constant)]
unsafe fn swoosh_avx2(x: &mut [f32], shift: f32, offset: f32) -> usize {
    let n8 = x.len() / 8 * 8;
    // SAFETY: loads and stores at i .. i + 8 <= n8 <= x.len().
    unsafe {
        let p = x.as_mut_ptr();
        let (sh, of, k08) = (_mm256_set1_ps(shift), _mm256_set1_ps(offset), _mm256_set1_ps(0.08));
        let (one, two, zero) = (_mm256_set1_ps(1.0), _mm256_set1_ps(2.0), _mm256_setzero_ps());
        let absm = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MAX));
        let ln2 = _mm256_set1_ps(core::f32::consts::LN_2);
        let d = [0.666_666_7f32, 0.4, 0.285_714_3, 0.222_222_2, 0.181_818_2].map(|v| _mm256_set1_ps(v));
        let sqh = _mm256_set1_epi32(0x3f35_04f3);
        let mut i = 0;
        while i < n8 {
            let v = _mm256_loadu_ps(p.add(i));
            let y = _mm256_sub_ps(v, sh);
            let e = exp8(_mm256_sub_ps(zero, _mm256_and_ps(y, absm)));
            // ln(1 + e) as `kernels::ln`
            let xl = _mm256_add_ps(one, e);
            let b = _mm256_castps_si256(xl);
            let ex = _mm256_srai_epi32::<23>(_mm256_sub_epi32(b, sqh));
            let m = _mm256_castsi256_ps(_mm256_sub_epi32(b, _mm256_slli_epi32::<23>(ex)));
            let s = _mm256_div_ps(_mm256_sub_ps(m, one), _mm256_add_ps(m, one));
            let z = _mm256_mul_ps(s, s);
            let mut r = _mm256_mul_ps(z, d[4]);
            r = _mm256_mul_ps(z, _mm256_add_ps(d[3], r));
            r = _mm256_mul_ps(z, _mm256_add_ps(d[2], r));
            r = _mm256_mul_ps(z, _mm256_add_ps(d[1], r));
            r = _mm256_mul_ps(z, _mm256_add_ps(d[0], r));
            r = _mm256_add_ps(two, r);
            let lnv = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(ex), ln2), _mm256_mul_ps(s, r));
            let sp = _mm256_add_ps(_mm256_max_ps(y, zero), lnv);
            _mm256_storeu_ps(p.add(i), _mm256_sub_ps(_mm256_sub_ps(sp, _mm256_mul_ps(k08, v)), of));
            i += 8;
        }
    }
    n8
}

/// `kernels::logsumexp` (None without AVX2 or for fewer than 8 values).
pub fn logsumexp(v: &[f32]) -> Option<f32> {
    if !ymm() || v.len() < 8 {
        return None;
    }
    // SAFETY: AVX2 + FMA checked by `ymm()`.
    Some(unsafe { logsumexp_avx2(v) })
}

#[target_feature(enable = "avx2,fma")]
unsafe fn logsumexp_avx2(v: &[f32]) -> f32 {
    let n8 = v.len() / 8 * 8;
    // SAFETY: loads at i .. i + 8 <= n8 <= v.len().
    unsafe {
        let p = v.as_ptr();
        let mut mx = _mm256_set1_ps(f32::MIN);
        let mut i = 0;
        while i < n8 {
            mx = _mm256_max_ps(mx, _mm256_loadu_ps(p.add(i)));
            i += 8;
        }
        let mut m = hmax(mx);
        for &x in &v[n8..] {
            m = m.max(x);
        }
        let mv = _mm256_set1_ps(m);
        let mut s = _mm256_setzero_ps();
        let mut i = 0;
        while i < n8 {
            s = _mm256_add_ps(s, exp8(_mm256_sub_ps(_mm256_loadu_ps(p.add(i)), mv)));
            i += 8;
        }
        let mut sum = hsum(s);
        for &x in &v[n8..] {
            sum += crate::kernels::exp(x - m);
        }
        m + sum.ln()
    }
}

/// `kernels::softmax_row`; false (nothing done) without AVX2 or for fewer than 8 values.
pub fn softmax_row(row: &mut [f32]) -> bool {
    if !ymm() || row.len() < 8 {
        return false;
    }
    // SAFETY: AVX2 + FMA checked by `ymm()`.
    unsafe { softmax_row_avx2(row) };
    true
}

#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_row_avx2(row: &mut [f32]) {
    let n = row.len();
    let n8 = n / 8 * 8;
    // SAFETY: vector accesses at i .. i + 8 <= n8 <= n; the tail by index below n.
    unsafe {
        let p = row.as_mut_ptr();
        let mut mx = _mm256_set1_ps(f32::MIN);
        let mut i = 0;
        while i < n8 {
            mx = _mm256_max_ps(mx, _mm256_loadu_ps(p.add(i)));
            i += 8;
        }
        let mut m = hmax(mx);
        for j in n8..n {
            m = m.max(*p.add(j));
        }
        let mv = _mm256_set1_ps(m);
        let mut s = _mm256_setzero_ps();
        let mut i = 0;
        while i < n8 {
            let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(p.add(i)), mv));
            _mm256_storeu_ps(p.add(i), e);
            s = _mm256_add_ps(s, e);
            i += 8;
        }
        let mut sum = hsum(s);
        for j in n8..n {
            let e = crate::kernels::exp(*p.add(j) - m);
            *p.add(j) = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        let iv = _mm256_set1_ps(inv);
        let mut i = 0;
        while i < n8 {
            _mm256_storeu_ps(p.add(i), _mm256_mul_ps(_mm256_loadu_ps(p.add(i)), iv));
            i += 8;
        }
        for j in n8..n {
            *p.add(j) *= inv;
        }
    }
}

/// `kernels::tanh_sum` for the first n / 8 * 8 values (n = the shortest length); returns how many it did.
pub fn tanh_sum(a: &[f32], b: &[f32], out: &mut [f32]) -> usize {
    if !ymm() {
        return 0;
    }
    let n8 = a.len().min(b.len()).min(out.len()) / 8 * 8;
    let mut i = 0;
    // SAFETY: AVX2 + FMA checked by `ymm()`; accesses at i .. i + 8 <= n8 <= every length.
    unsafe {
        while i < n8 {
            let x = _mm256_add_ps(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i)));
            _mm256_storeu_ps(out.as_mut_ptr().add(i), tanh8_tf(x));
            i += 8;
        }
    }
    n8
}

/// `tanh8` callable from functions without the target feature (only after `ymm()`).
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh8_tf(x: __m256) -> __m256 {
    tanh8(x)
}

/// `kernels::lstm_cell` for j < c.len() / 8 * 8 (g: 4 n gate values, h: n); returns how many it did.
pub fn lstm_cell(g: &[f32], c: &mut [f32], h: &mut [f32]) -> usize {
    let n = c.len();
    assert!(g.len() >= 4 * n && h.len() >= n);
    if !ymm() {
        return 0;
    }
    // SAFETY: AVX2 + FMA checked by `ymm()`; sizes asserted above.
    unsafe { lstm_cell_avx2(g, c, h) }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn lstm_cell_avx2(g: &[f32], c: &mut [f32], h: &mut [f32]) -> usize {
    let n = c.len();
    let n8 = n / 8 * 8;
    let gp = g.as_ptr();
    let mut j = 0;
    // SAFETY: j + 8 <= n8 <= n; g read at up to 3 n + j + 8 <= 4 n <= g.len(), h.len() >= n (asserted by the caller).
    unsafe {
        while j < n8 {
            let i = sigmoid8(_mm256_loadu_ps(gp.add(j)));
            let f = sigmoid8(_mm256_loadu_ps(gp.add(n + j)));
            let gg = tanh8(_mm256_loadu_ps(gp.add(2 * n + j)));
            let o = sigmoid8(_mm256_loadu_ps(gp.add(3 * n + j)));
            let cv = _mm256_add_ps(_mm256_mul_ps(f, _mm256_loadu_ps(c.as_ptr().add(j))), _mm256_mul_ps(i, gg));
            _mm256_storeu_ps(c.as_mut_ptr().add(j), cv);
            _mm256_storeu_ps(h.as_mut_ptr().add(j), _mm256_mul_ps(o, tanh8(cv)));
            j += 8;
        }
    }
    n8
}

/// `kernels::glu` for i < out.len() / 8 * 8; returns how many it did.
pub fn glu(a: &[f32], b: &[f32], out: &mut [f32]) -> usize {
    let n = out.len();
    assert!(a.len() >= n && b.len() >= n);
    if !ymm() {
        return 0;
    }
    // SAFETY: AVX2 + FMA checked by `ymm()`; sizes asserted above.
    unsafe { glu_avx2(a, b, out) }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn glu_avx2(a: &[f32], b: &[f32], out: &mut [f32]) -> usize {
    let n8 = out.len() / 8 * 8;
    let mut i = 0;
    // SAFETY: i + 8 <= n8 <= out.len() <= a.len(), b.len() (asserted by the caller).
    unsafe {
        while i < n8 {
            let s = sigmoid8(_mm256_loadu_ps(b.as_ptr().add(i)));
            _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(_mm256_loadu_ps(a.as_ptr().add(i)), s));
            i += 8;
        }
    }
    n8
}

/// `kernels::dw_taps` (needs out.len() % 8 == 0); false (nothing done) otherwise or without AVX2.
pub fn dw_taps(out: &mut [f32], init: Option<&[f32]>, w: &[f32], x: &[f32], taps: usize) -> bool {
    let d = out.len();
    if !ymm() || d % 8 != 0 {
        return false;
    }
    assert!(w.len() >= taps * d && x.len() >= taps * d && init.is_none_or(|v| v.len() >= d));
    // SAFETY: AVX2 + FMA checked by `ymm()`; sizes asserted above.
    unsafe { dw_taps_avx2(out, init, w, x, taps) };
    true
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dw_taps_avx2(out: &mut [f32], init: Option<&[f32]>, w: &[f32], x: &[f32], taps: usize) {
    let d = out.len();
    // SAFETY: w, x >= taps d, init >= d, d % 8 == 0 (checked by the caller); every access is below those bounds.
    unsafe {
        let (op, wp, xp) = (out.as_mut_ptr(), w.as_ptr(), x.as_ptr());
        let sp = init.map_or(op as *const f32, |v| v.as_ptr());
        let mut c = 0;
        while c + 32 <= d {
            let mut acc = [_mm256_loadu_ps(sp.add(c)), _mm256_loadu_ps(sp.add(c + 8)), _mm256_loadu_ps(sp.add(c + 16)), _mm256_loadu_ps(sp.add(c + 24))];
            for u in 0..taps {
                let (wr, xr) = (wp.add(u * d + c), xp.add(u * d + c));
                for q in 0..4 {
                    acc[q] = _mm256_fmadd_ps(_mm256_loadu_ps(wr.add(8 * q)), _mm256_loadu_ps(xr.add(8 * q)), acc[q]);
                }
            }
            for q in 0..4 {
                _mm256_storeu_ps(op.add(c + 8 * q), acc[q]);
            }
            c += 32;
        }
        while c < d {
            let mut acc = _mm256_loadu_ps(sp.add(c));
            for u in 0..taps {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(wp.add(u * d + c)), _mm256_loadu_ps(xp.add(u * d + c)), acc);
            }
            _mm256_storeu_ps(op.add(c), acc);
            c += 8;
        }
    }
}

/// `kernels::conv0_rows` with AVX2 (one output column = 8 channels = one register; the same fused multiply-adds in the
/// same order as the AVX-512 version). False (nothing done) without AVX2.
pub fn conv0_rows(x: &[f32], f: usize, i0: usize, nt: usize, w: &[f32], bias: &[f32], out: &mut [f32]) -> bool {
    if !ymm() {
        return false;
    }
    assert!(w.len() >= 72 && bias.len() >= 8 && out.len() >= nt * f * 8 && x.len() >= (i0 + nt + 2) * f && f >= 1);
    // SAFETY: AVX2 + FMA checked by `ymm()`; sizes asserted above.
    unsafe { conv0_avx2(x, f, i0, nt, w, bias, out) };
    true
}

#[target_feature(enable = "avx2,fma")]
unsafe fn conv0_avx2(x: &[f32], f: usize, i0: usize, nt: usize, w: &[f32], bias: &[f32], out: &mut [f32]) {
    // SAFETY: x read at rows i0 + ii + a < i0 + nt + 2 and columns 0 .. f (by index, checked by the caller); out written
    // at (ii f + j) 8 .. + 8 < nt f 8.
    unsafe {
        let mut wv = [_mm256_setzero_ps(); 9];
        for (p, v) in wv.iter_mut().enumerate() {
            let l: [f32; 8] = core::array::from_fn(|co| w[co * 9 + p]);
            *v = _mm256_loadu_ps(l.as_ptr());
        }
        let bv = _mm256_loadu_ps(bias.as_ptr());
        let at = |r: usize, c: isize| -> f32 { if c >= 0 && (c as usize) < f { x[r * f + c as usize] } else { 0.0 } };
        for ii in 0..nt {
            let i = i0 + ii;
            for j in 0..f {
                let mut acc = bv;
                for a in 0..3 {
                    for b in 0..3 {
                        let c = j as isize + b as isize - 1;
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(at(i + a, c)), wv[a * 3 + b], acc);
                    }
                }
                let o = &mut out[(ii * f + j) * 8..(ii * f + j) * 8 + 8];
                _mm256_storeu_ps(o.as_mut_ptr(), acc);
            }
        }
    }
}

/// Rows of the f32 micro-kernel per call (`kernels::blocked_avx512` steps by this on the 256-bit levels).
pub const F32_ROWS: usize = 6;

/// The f32 micro-kernel for `rows` (1..=6) rows of `a` (k each) x one panel (k x 16) with start values `init`, written
/// to `cols` (<= 16) columns of each row of `out` (row stride n). Returns false (nothing done) without AVX2.
#[allow(clippy::too_many_arguments)]
pub fn kernel_f32(rows: usize, a: &[f32], k: usize, panel: &[f32], init: &[f32; 16], out: &mut [f32], n: usize, cols: usize) -> bool {
    if !ymm() {
        return false;
    }
    assert!((1..=F32_ROWS).contains(&rows) && a.len() >= rows * k && panel.len() >= k * 16 && cols <= 16 && out.len() >= (rows - 1) * n + cols);
    let args = (a, k, panel, init, out, n, cols);
    // SAFETY: AVX2 + FMA checked by `ymm()`; sizes asserted above.
    unsafe {
        match rows {
            6 => kernel_avx2::<6>(args),
            5 => kernel_avx2::<5>(args),
            4 => kernel_avx2::<4>(args),
            3 => kernel_avx2::<3>(args),
            2 => kernel_avx2::<2>(args),
            _ => kernel_avx2::<1>(args),
        }
    }
    true
}

#[target_feature(enable = "avx2,fma")]
unsafe fn kernel_avx2<const R: usize>((a, k, panel, init, out, n, cols): (&[f32], usize, &[f32], &[f32; 16], &mut [f32], usize, usize)) {
    // SAFETY: reads a[r k + p] (r < R, p < k), panel[p 16 .. p 16 + 16] (p < k); writes `cols` values of R rows at stride
    // n: all inside the slices (checked by the caller).
    unsafe {
        let (ap, bp) = (a.as_ptr(), panel.as_ptr());
        let start = [_mm256_loadu_ps(init.as_ptr()), _mm256_loadu_ps(init.as_ptr().add(8))];
        let mut acc = [start; R];
        for p in 0..k {
            let w0 = _mm256_loadu_ps(bp.add(p * 16));
            let w1 = _mm256_loadu_ps(bp.add(p * 16 + 8));
            for r in 0..R {
                let x = _mm256_set1_ps(*ap.add(r * k + p));
                acc[r][0] = _mm256_fmadd_ps(x, w0, acc[r][0]);
                acc[r][1] = _mm256_fmadd_ps(x, w1, acc[r][1]);
            }
        }
        for r in 0..R {
            let o = out.as_mut_ptr().add(r * n);
            if cols == 16 {
                _mm256_storeu_ps(o, acc[r][0]);
                _mm256_storeu_ps(o.add(8), acc[r][1]);
            } else {
                let mut l = [0f32; 16];
                _mm256_storeu_ps(l.as_mut_ptr(), acc[r][0]);
                _mm256_storeu_ps(l.as_mut_ptr().add(8), acc[r][1]);
                core::ptr::copy_nonoverlapping(l.as_ptr(), o, cols);
            }
        }
    }
}
