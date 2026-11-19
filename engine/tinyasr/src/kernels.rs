//! Float kernels. The matrix-multiply micro-kernel is explicit NEON on aarch64; elsewhere it is
//! written so LLVM auto-vectorizes it (SIMD128 on wasm, SSE on x86-64).

use alloc::vec;
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use crate::compat::F32Ext;

// Micro-kernel tile: MR rows x NR columns of accumulators. aarch64 has 32 vector registers
// (4x16 = 16 accumulators); wasm SIMD / x86-64 SSE have 16 (2x16 = 8). Measured, see README.
const NR: usize = 16; // output columns per panel
#[cfg(target_arch = "aarch64")]
const MR: usize = 4; // output rows per micro-kernel call
#[cfg(not(target_arch = "aarch64"))]
const MR: usize = 2;
const MB: usize = 64; // rows per cache block

/// Right-hand matrix packed into panels of NR output columns, k-major inside a panel, so the
/// micro-kernel reads NR contiguous weights per k step (outer-product form, no reductions).
pub struct Packed {
    pub n: usize,
    pub k: usize,
    data: Vec<f32>, // ceil(n / NR) panels x k x NR, zero-padded
}

impl Packed {
    /// From b (n x k): row j holds the weights of output column j (PyTorch Linear layout).
    pub fn from_rows(b: &[f32], n: usize, k: usize) -> Packed {
        let mut data = vec![0.0; n.div_ceil(NR) * k * NR];
        for j in 0..n {
            let (panel, l) = (j / NR, j % NR);
            for (p, v) in b[j * k..(j + 1) * k].iter().enumerate() {
                data[(panel * k + p) * NR + l] = *v;
            }
        }
        Packed { n, k, data }
    }

    /// The weights back as rows (n x k, PyTorch Linear layout).
    pub fn rows(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.n * self.k];
        for j in 0..self.n {
            let (panel, l) = (j / NR, j % NR);
            for p in 0..self.k {
                out[j * self.k + p] = self.data[(panel * self.k + p) * NR + l];
            }
        }
        out
    }

    fn panel(&self, i: usize) -> &[f32] {
        &self.data[i * self.k * NR..(i + 1) * self.k * NR]
    }
}

/// out (m x n) = a (m x k) * B (+ bias per column).
pub fn gemm(a: &[f32], m: usize, b: &Packed, bias: Option<&[f32]>) -> Vec<f32> {
    let (n, k) = (b.n, b.k);
    assert!(a.len() >= m * k);
    let mut out = vec![0.0; m * n];
    gemm_rows(&a[..m * k], b, bias, &mut out);
    out
}

/// out (m x n) = a (m x k) * B, on the calling thread (for use inside parallel loops).
pub fn gemm_into(a: &[f32], b: &Packed, out: &mut [f32]) {
    blocked(a, b, None, out);
}

fn gemm_rows(a: &[f32], b: &Packed, bias: Option<&[f32]>, out: &mut [f32]) {
    par_rows(a, b.k, b.n, out, &|a, out| blocked(a, b, bias, out));
}

/// Rows of the product split across the thread pool when the product is big enough: `f(a rows, out rows)` for
/// disjoint row ranges (a: k per row, out: n per row).
fn par_rows(a: &[f32], k: usize, n: usize, out: &mut [f32], f: &(dyn Fn(&[f32], &mut [f32]) + Sync)) {
    let m = a.len() / k;
    #[cfg(feature = "std")]
    {
        let th = crate::pool::threads();
        if th > 1 && m >= 2 * PAR_MIN_ROWS && m * n * k >= PAR_MIN_WORK {
            let tasks = th.min(m / PAR_MIN_ROWS);
            let per = m.div_ceil(tasks).next_multiple_of(4);
            let parts: Vec<std::sync::Mutex<(&[f32], &mut [f32])>> =
                a.chunks(per * k).zip(out.chunks_mut(per * n)).map(std::sync::Mutex::new).collect();
            crate::pool::par_for(parts.len(), &|i| {
                let mut p = parts[i].lock().unwrap();
                let (a, o) = &mut *p;
                f(a, o)
            });
            return;
        }
    }
    let _ = (m, n);
    f(a, out)
}

/// `f(first row, rows)` over `out` cut into rows of `row` values, split across the thread pool when `work` (about the
/// multiply-adds of the whole loop) is big enough.
pub fn par_chunks(out: &mut [f32], row: usize, work: usize, f: &(dyn Fn(usize, &mut [f32]) + Sync)) {
    #[cfg(feature = "std")]
    {
        let rows = if row == 0 { 0 } else { out.len() / row };
        let th = crate::pool::threads();
        if th > 1 && rows >= 2 && work >= PAR_MIN_WORK {
            let tasks = (th * 2).min(rows);
            let per = rows.div_ceil(tasks);
            let parts: Vec<std::sync::Mutex<&mut [f32]>> = out.chunks_mut(per * row).map(std::sync::Mutex::new).collect();
            crate::pool::par_for(parts.len(), &|i| f(i * per, &mut parts[i].lock().unwrap()));
            return;
        }
    }
    let _ = (row, work);
    f(0, out)
}

/// Row blocks of `out` (rows of `row` values) in parallel with ONE pool synchronization: `f(first row, rows)` on
/// about one block per thread, every pool loop inside `f` serial (see `pool::serial`). For row-wise chains (a
/// product, its activation, the next product, a residual add) that would otherwise synchronize the pool once per
/// step. `work`: about the multiply-adds of the whole loop; below `PAR_MIN_WORK` (or one row) it runs as one block
/// on the calling thread, with the pool free to split the steps inside as before. Same results either way: every
/// step works row by row.
pub fn par_blocks(out: &mut [f32], row: usize, work: usize, f: &(dyn Fn(usize, &mut [f32]) + Sync)) {
    #[cfg(feature = "std")]
    {
        let rows = if row == 0 { 0 } else { out.len() / row };
        let th = crate::pool::threads();
        if th > 1 && rows >= 2 * BLOCK_MIN_ROWS && work >= PAR_MIN_WORK {
            let tasks = th.min(rows / BLOCK_MIN_ROWS);
            let per = rows.div_ceil(tasks).next_multiple_of(4);
            let parts: Vec<std::sync::Mutex<&mut [f32]>> = out.chunks_mut(per * row).map(std::sync::Mutex::new).collect();
            crate::pool::par_for(parts.len(), &|i| crate::pool::serial(|| f(i * per, &mut parts[i].lock().unwrap())));
            return;
        }
    }
    let _ = (row, work);
    f(0, out)
}

/// Fewest rows per `par_blocks` task.
#[cfg(feature = "std")]
const BLOCK_MIN_ROWS: usize = 4;

/// Fewest rows per parallel task, and the least multiply-adds (m n k) worth splitting.
#[cfg(feature = "std")]
const PAR_MIN_ROWS: usize = 8;
/// Few-row int8 products: the least weight rows per task, and the least multiply-adds worth splitting by columns.
#[cfg(feature = "std")]
const PAR_MIN_COLS: usize = 128;
#[cfg(feature = "std")]
const PAR_MIN_COLS_WORK: usize = 1 << 19;
#[cfg(feature = "std")]
const PAR_MIN_WORK: usize = 1 << 21;

/// Weights as panels of NR output columns, k-major inside a panel (see `Packed`).
trait Panels {
    fn n(&self) -> usize;
    fn k(&self) -> usize;
    /// Panel `jp`: stored, or widened into `scratch` (k x NR).
    fn panel<'s>(&'s self, jp: usize, scratch: &'s mut [f32]) -> &'s [f32];
}

impl Panels for Packed {
    fn n(&self) -> usize {
        self.n
    }
    fn k(&self) -> usize {
        self.k
    }
    fn panel<'s>(&'s self, jp: usize, _: &'s mut [f32]) -> &'s [f32] {
        Packed::panel(self, jp)
    }
}

/// The cache-blocked loop around the micro-kernel: blocks of MB rows of `a`, one weight panel at a time.
fn blocked(a: &[f32], b: &impl Panels, bias: Option<&[f32]>, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if vnni() || crate::x86::ymm() {
        return blocked_avx512(a, b, bias, out);
    }
    let (n, k) = (b.n(), b.k());
    let m = a.len() / k;
    let mut tile = [[0f32; NR]; MR];
    let mut scratch = vec![0f32; k * NR];
    for i0 in (0..m).step_by(MB) {
        let i1 = (i0 + MB).min(m);
        for jp in 0..n.div_ceil(NR) {
            let panel = b.panel(jp, &mut scratch);
            let j0 = jp * NR;
            let cols = NR.min(n - j0);
            let mut init = [0f32; NR];
            if let Some(bias) = bias {
                init[..cols].copy_from_slice(&bias[j0..j0 + cols]);
            }
            let mut i = i0;
            while i < i1 {
                let rows = MR.min(i1 - i);
                if rows == MR {
                    kernel::<MR>(&a[i * k..(i + MR) * k], k, panel, &init, &mut tile);
                } else {
                    for r in 0..rows {
                        let mut one = [[0f32; NR]; 1];
                        kernel::<1>(&a[(i + r) * k..(i + r + 1) * k], k, panel, &init, &mut one);
                        tile[r] = one[0];
                    }
                }
                for r in 0..rows {
                    out[(i + r) * n + j0..(i + r) * n + j0 + cols].copy_from_slice(&tile[r][..cols]);
                }
                i += rows;
            }
        }
    }
}

/// `blocked` with an AVX-512 micro-kernel: 8 rows x 16 columns, one zmm accumulator per row (fused multiply-add); on
/// the 256-bit levels the AVX2 one of `x86` (6 rows x 16 columns, the same fused multiply-adds, so the same results).
#[cfg(target_arch = "x86_64")]
fn blocked_avx512(a: &[f32], b: &impl Panels, bias: Option<&[f32]>, out: &mut [f32]) {
    let wide = vnni();
    let (n, k) = (b.n(), b.k());
    let m = a.len() / k;
    assert!(out.len() >= m * n);
    let mut scratch = vec![0f32; k * NR];
    for i0 in (0..m).step_by(MB) {
        let i1 = (i0 + MB).min(m);
        for jp in 0..n.div_ceil(NR) {
            let panel = b.panel(jp, &mut scratch);
            assert!(panel.len() >= k * NR);
            let j0 = jp * NR;
            let cols = NR.min(n - j0);
            let mut init = [0f32; NR];
            if let Some(bias) = bias {
                init[..cols].copy_from_slice(&bias[j0..j0 + cols]);
            }
            let mut i = i0;
            while i < i1 {
                let rows = (i1 - i).min(if wide { 8 } else { crate::x86::F32_ROWS });
                if !wide {
                    let done = crate::x86::kernel_f32(rows, &a[i * k..(i + rows) * k], k, panel, &init, &mut out[i * n + j0..], n, cols);
                    assert!(done);
                    i += rows;
                    continue;
                }
                let args = (&a[i * k..(i + rows) * k], k, panel, &init, &mut out[i * n + j0..], n, cols);
                #[allow(unsafe_code)]
                // SAFETY: `vnni()` checked AVX-512 F; the kernel reads `rows` rows of k, k x 16 panel values and writes
                // `rows` rows of `cols` at stride n, all inside the slices (checked in the kernel).
                unsafe {
                    match rows {
                        8 => kernel_avx512::<8>(args),
                        7 => kernel_avx512::<7>(args),
                        6 => kernel_avx512::<6>(args),
                        5 => kernel_avx512::<5>(args),
                        4 => kernel_avx512::<4>(args),
                        3 => kernel_avx512::<3>(args),
                        2 => kernel_avx512::<2>(args),
                        _ => kernel_avx512::<1>(args),
                    }
                }
                i += rows;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn kernel_avx512<const R: usize>((a, k, panel, init, out, n, cols): (&[f32], usize, &[f32], &[f32; NR], &mut [f32], usize, usize)) {
    use core::arch::x86_64::*;
    assert!(a.len() >= R * k && panel.len() >= k * NR && cols <= NR && out.len() >= (R - 1) * n + cols);
    // SAFETY: reads a[r k + p] (r < R, p < k) and panel[p 16 .. p 16 + 16] (p < k); writes `cols` lanes of R rows.
    unsafe {
        let (ap, bp) = (a.as_ptr(), panel.as_ptr());
        let start = _mm512_loadu_ps(init.as_ptr());
        let mut acc = [start; R];
        for p in 0..k {
            let w = _mm512_loadu_ps(bp.add(p * NR));
            for r in 0..R {
                acc[r] = _mm512_fmadd_ps(_mm512_set1_ps(*ap.add(r * k + p)), w, acc[r]);
            }
        }
        let mask: __mmask16 = if cols == NR { 0xffff } else { (1u16 << cols) - 1 };
        for r in 0..R {
            _mm512_mask_storeu_ps(out.as_mut_ptr().add(r * n), mask, acc[r]);
        }
    }
}

/// out = init + rows(a) x panel, for R rows of `a` (each k long) and one panel of NR columns.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn kernel<const R: usize>(a: &[f32], k: usize, panel: &[f32], init: &[f32; NR], out: &mut [[f32; NR]; R]) {
    let rows: [&[f32]; R] = core::array::from_fn(|r| &a[r * k..(r + 1) * k]);
    let mut acc = [*init; R];
    for (p, b) in panel[..k * NR].chunks_exact(NR).enumerate() {
        for r in 0..R {
            let x = rows[r][p];
            for l in 0..NR {
                acc[r][l] += x * b[l];
            }
        }
    }
    *out = acc;
}

/// NEON version (`arm::kernel_f32`): R x 16 accumulators in q registers, fused multiply-add by element.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn kernel<const R: usize>(a: &[f32], k: usize, panel: &[f32], init: &[f32; NR], out: &mut [[f32; NR]; R]) {
    const { assert!(NR == 16) };
    crate::arm::kernel_f32::<R>(a, k, panel, init, out)
}

/// exp for x <= ~88 via 2^x = 2^i * 2^f with a degree-5 polynomial (rel. err ~2e-7).
/// Branch-free so loops over it vectorize, unlike the libm call.
#[inline(always)]
#[allow(clippy::approx_constant)] // polynomial coefficients, not ln 2
pub fn exp(x: f32) -> f32 {
    let t = (x * core::f32::consts::LOG2_E).clamp(-126.0, 126.0);
    let i = t.floor();
    let f = t - i;
    let p = 1.0 + f * (0.693_147_2 + f * (0.240_226_5 + f * (0.055_504_11 + f * (0.009_618_129 + f * 0.001_333_355))));
    f32::from_bits(((i as i32 + 127) as u32) << 23) * p
}

/// ln for x > 0 (normal): x = m 2^e with m in [sqrt(1/2), sqrt 2), ln m = 2 atanh(s), s = (m - 1) / (m + 1), as an odd
/// series to s^11 (|s| < 0.172: rel. err < 1e-8). Branch-free so loops over it vectorize, unlike the libm call.
#[inline(always)]
pub fn ln(x: f32) -> f32 {
    let b = x.to_bits();
    // exponent relative to sqrt(1/2), so the mantissa lands in [sqrt(1/2), sqrt 2)
    let e = ((b.wrapping_sub(0x3f35_04f3)) as i32) >> 23;
    let m = f32::from_bits(b.wrapping_sub((e as u32) << 23));
    let s = (m - 1.0) / (m + 1.0);
    let z = s * s;
    let p = 2.0 + z * (0.666_666_7 + z * (0.4 + z * (0.285_714_3 + z * (0.222_222_2 + z * 0.181_818_2))));
    e as f32 * core::f32::consts::LN_2 + s * p
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    1.0 - 2.0 / (exp(2.0 * x) + 1.0)
}

/// Integer weights with one f32 scale per `group` consecutive weights of a row (group = k: per row).
pub struct Quant {
    pub n: usize,
    pub k: usize,
    group: usize,
    scale: Vec<f32>, // n * k / group
    q: Vec<i8>,      // n x k
    /// x86-64 VNNI: -128 x the sum of each group's weights (n * k / group), see `tile_vnni`.
    #[cfg(target_arch = "x86_64")]
    wsum: Vec<i32>,
    /// x86-64 int8 levels (AVX-512 VNNI, AVX-VNNI, AVX2 with exact i16 sums), groups of 64: the weights repacked for `pk_run` (then `q` is empty).
    #[cfg(target_arch = "x86_64")]
    pk: Option<PackI8>,
    /// aarch64: the weights repacked for `arm::block` (then `q` is empty).
    #[cfg(target_arch = "aarch64")]
    apk: Option<crate::arm::Pack>,
}

/// Integer weights in panels of 16 output columns for `tile_pk`: per panel, per 4 inputs, the 4 bytes of each of the
/// 16 columns (64 bytes = one `vpdpbusd` operand); scales and weight sums per panel, group and column.
#[cfg(target_arch = "x86_64")]
struct PackI8 {
    w: Vec<i8>,     // panels x k/4 x 16 x 4
    sw: Vec<f32>,   // panels x ng x 16
    ws: Vec<i32>,   // panels x ng x 16: -128 x the group's weight sum
    /// AVX2 level: `x86::i16_steps` of these weights (0 on the VNNI levels).
    steps: usize,
}

#[cfg(target_arch = "x86_64")]
impl PackI8 {
    fn new(n: usize, k: usize, g: usize, scale: &[f32], q: &[i8], steps: usize) -> PackI8 {
        let (np, ng) = (n.div_ceil(16), k / g);
        let mut w = vec![0i8; np * k * 16];
        let mut sw = vec![0f32; np * ng * 16];
        let mut ws = vec![0i32; np * ng * 16];
        for j in 0..n {
            let (p, l) = (j / 16, j % 16);
            for kk in 0..k {
                w[p * k * 16 + (kk / 4) * 64 + l * 4 + kk % 4] = q[j * k + kk];
            }
            for c in 0..ng {
                sw[(p * ng + c) * 16 + l] = scale[j * ng + c];
                ws[(p * ng + c) * 16 + l] = -128 * q[j * k + c * g..j * k + (c + 1) * g].iter().map(|&v| v as i32).sum::<i32>();
            }
        }
        PackI8 { w, sw, ws, steps }
    }
}

impl Quant {
    pub fn new(n: usize, k: usize, group: usize, scale: Vec<f32>, q: Vec<i8>) -> Quant {
        assert!(group > 0 && k % group == 0 && scale.len() == n * k / group && q.len() == n * k);
        #[cfg(target_arch = "x86_64")]
        let pk = {
            // the 256-bit levels pack too (AVX2 only when its i16 sums stay exact for these weights)
            let steps = if crate::x86::ymm() { crate::x86::i16_steps(&q, group) } else { 0 };
            let on = vnni() || (crate::x86::ymm() && crate::x86::pack_usable(steps));
            (on && group % 64 == 0 && k % 64 == 0).then(|| PackI8::new(n, k, group, &scale, &q, steps))
        };
        #[cfg(target_arch = "x86_64")]
        let (q, wsum) = if pk.is_some() {
            (Vec::new(), Vec::new())
        } else {
            let ws = q.chunks_exact(group).map(|g| -128 * g.iter().map(|&w| w as i32).sum::<i32>()).collect();
            (q, ws)
        };
        #[cfg(target_arch = "aarch64")]
        let apk = (int8_dot() && crate::arm::Pack::usable(k, group)).then(|| crate::arm::Pack::new(n, k, group, &scale, &q));
        #[cfg(target_arch = "aarch64")]
        let q = if apk.is_some() { Vec::new() } else { q };
        Quant {
            n,
            k,
            group,
            scale,
            q,
            #[cfg(target_arch = "x86_64")]
            wsum,
            #[cfg(target_arch = "x86_64")]
            pk,
            #[cfg(target_arch = "aarch64")]
            apk,
        }
    }

    /// Whether a `Quant` with this group size multiplies in int8 here (else its weights are widened per call).
    pub fn int8_for(group: usize) -> bool {
        (group % 16 == 0 && int8_dot() && (!cfg!(target_arch = "x86_64") || !vnni() || group % 64 == 0)) || (ymm() && group % 64 == 0)
    }

    /// out (m x n) = a (m x k) * W^T (+ bias).
    ///
    /// x86-64: weights packed at load (`pk`) are multiplied in int8 by `gemm_pk`; AVX-512 VNNI without packed weights
    /// by `gemm_i8`. With the dot-product extension (aarch64 `sdot`) or wasm SIMD128 (`i32x4.dot_i16x8_s`), the
    /// activations are quantized to int8 per row and per weight group and multiplied by the integer weights directly
    /// (`INT8_DOT=0` turns this off on aarch64). Otherwise the weights are widened to f32 one panel of NR rows at a
    /// time and reused for a block of MB rows of `a` by the float micro-kernel.

    fn gemm(&self, a: &[f32], m: usize, bias: Option<&[f32]>) -> Vec<f32> {
        assert!(a.len() >= m * self.k);
        #[cfg(target_arch = "x86_64")]
        if let Some(pk) = &self.pk {
            return self.gemm_pk(pk, &a[..m * self.k], m, bias);
        }
        // 256-bit levels without packed weights (AVX2, weights too large for exact i16 sums): widened per call
        if Quant::int8_for(self.group) && !ymm() {
            return self.gemm_i8(&a[..m * self.k], m, bias);
        }
        let mut out = vec![0.0; m * self.n];
        par_rows(&a[..m * self.k], self.k, self.n, &mut out, &|a, out| blocked(a, self, bias, out));
        out
    }

    fn gemm_i8(&self, a: &[f32], m: usize, bias: Option<&[f32]>) -> Vec<f32> {
        #[cfg(target_arch = "x86_64")]
        if let Some(pk) = &self.pk {
            return self.gemm_pk(pk, a, m, bias);
        }
        let (n, k, g) = (self.n, self.k, self.group);
        let (aq, sa) = quantize_rows(a, k, g);
        let mut out = vec![0.0; m * n];
        #[cfg(feature = "std")]
        {
            let ng = k / g;
            let th = crate::pool::threads();
            if th > 1 && m >= 2 * PAR_MIN_ROWS && m * n * k >= PAR_MIN_WORK {
                let tasks = th.min(m / PAR_MIN_ROWS);
                let per = m.div_ceil(tasks).next_multiple_of(4);
                let parts: Vec<std::sync::Mutex<(&[i8], &[f32], &mut [f32])>> = aq
                    .chunks(per * k)
                    .zip(sa.chunks(per * ng))
                    .zip(out.chunks_mut(per * n))
                    .map(|((a, s), o)| std::sync::Mutex::new((a, s, o)))
                    .collect();
                crate::pool::par_for(parts.len(), &|i| {
                    let mut p = parts[i].lock().unwrap();
                    let (a, s, o) = &mut *p;
                    let rows = a.len() / k;
                    self.cols_i8(a, s, rows, o)
                });
            } else if th > 1 && m * n * k >= PAR_MIN_COLS_WORK && n >= 2 * PAR_MIN_COLS {
                // few rows (decoder steps): split the output columns instead, each task into its own block
                let tasks = th.min(n / PAR_MIN_COLS);
                let per = n.div_ceil(tasks).next_multiple_of(if cfg!(target_arch = "aarch64") { 16 } else { TC });
                let blocks: Vec<std::sync::Mutex<Vec<f32>>> =
                    (0..n.div_ceil(per)).map(|t| std::sync::Mutex::new(vec![0.0; m * (per.min(n - t * per))])).collect();
                crate::pool::par_for(blocks.len(), &|t| {
                    let mut b = blocks[t].lock().unwrap();
                    let (j0, j1) = (t * per, ((t + 1) * per).min(n));
                    self.block_i8(&aq, &sa, 0, m, j0, j1, &mut b, j1 - j0, j0);
                });
                for (t, b) in blocks.iter().enumerate() {
                    let b = b.lock().unwrap();
                    let (j0, w) = (t * per, b.len() / m);
                    for i in 0..m {
                        out[i * n + j0..i * n + j0 + w].copy_from_slice(&b[i * w..(i + 1) * w]);
                    }
                }
            } else {
                self.cols_i8(&aq, &sa, m, &mut out);
            }
        }
        #[cfg(not(feature = "std"))]
        self.cols_i8(&aq, &sa, m, &mut out);
        if let Some(b) = bias {
            for row in out.chunks_exact_mut(n) {
                for (o, b) in row.iter_mut().zip(b) {
                    *o += b;
                }
            }
        }
        out
    }

    /// `gemm_i8` with the repacked weights (the x86-64 int8 levels: AVX-512 VNNI `pk_block`, the 256-bit ones `x86::pk_block`).
    #[cfg(target_arch = "x86_64")]
    fn gemm_pk(&self, pk: &PackI8, a: &[f32], m: usize, bias: Option<&[f32]>) -> Vec<f32> {
        let (n, k, g) = (self.n, self.k, self.group);
        let a = &a[..m * k];
        #[cfg(feature = "std")]
        let (aq, sa) = {
            let ng = k / g;
            let th = crate::pool::threads();
            if th > 1 && m >= 2 * PK_QUANT_ROWS && m * k >= PK_QUANT_WORK {
                let mut q = vec![0u8; m * k];
                let mut sc = vec![0f32; m * ng];
                let per = m.div_ceil(th.min(m / PK_QUANT_ROWS));
                let parts: Vec<std::sync::Mutex<(&[f32], &mut [u8], &mut [f32])>> = a
                    .chunks(per * k)
                    .zip(q.chunks_mut(per * k))
                    .zip(sc.chunks_mut(per * ng))
                    .map(|((a, q), s)| std::sync::Mutex::new((a, q, s)))
                    .collect();
                crate::pool::par_for(parts.len(), &|i| {
                    let mut p = parts[i].lock().unwrap();
                    let (a, q, s) = &mut *p;
                    quantize_u8_into(a, g, q, s)
                });
                drop(parts);
                (q, sc)
            } else {
                quantize_rows_u8(a, g)
            }
        };
        #[cfg(not(feature = "std"))]
        let (aq, sa) = quantize_rows_u8(a, g);
        // every element is written by `pk_block` (all rows, all 16-column panels, masked last panel), so the
        // output needs no zero fill
        let mut out: Vec<f32> = Vec::with_capacity(m * n);

        #[cfg(feature = "std")]
        {
            let th = crate::pool::threads();
            let np = n.div_ceil(16);
            if th > 1 && m * n * k >= if m < 2 * PR { PK_PAR_WORK_FEW } else { PK_PAR_WORK } && m * np >= 4 {
                // blocks of rows x 16-column panels: few rows -> split the columns (each thread streams only its share
                // of the weights), many rows -> also split the rows
                let tasks = th.min((m * n * k) / PK_TASK_WORK).max(2);
                let rg = (m / PK_MIN_ROWS).clamp(1, tasks);
                let cg = (tasks / rg).clamp(1, np);
                let (rper, pper) = (m.div_ceil(rg).next_multiple_of(PR), np.div_ceil(cg));
                let (nr, nc) = (m.div_ceil(rper), np.div_ceil(pper));
                let o = SendPtr(out.as_mut_ptr());
                let (aq, sa) = (&aq[..], &sa[..]);
                crate::pool::par_for(nr * nc, &|t| {
                    let (ri, ci) = (t / nc, t % nc);
                    let (r0, r1) = (ri * rper, ((ri + 1) * rper).min(m));
                    let (p_lo, p_hi) = (ci * pper, ((ci + 1) * pper).min(np));
                    let o = &o;
                    #[allow(unsafe_code)]
                    // SAFETY: `pk` exists only on a level that multiplies it (see `pk_run`); `out` holds m x n; tasks write
                    // disjoint blocks.
                    unsafe {
                        pk_run(pk, aq, sa, r0, r1, p_lo, p_hi, n, k, g, bias, o.0)
                    }
                });
                #[allow(unsafe_code)]
                // SAFETY: the tasks above wrote all m x n values (disjoint blocks covering every row and panel).
                unsafe {
                    out.set_len(m * n)
                };
                return out;
            }
        }
        #[allow(unsafe_code)]
        // SAFETY: `pk` exists only on a level that multiplies it (see `pk_run`); it writes all m x n values of the
        // capacity reserved above.
        unsafe {
            pk_run(pk, &aq, &sa, 0, m, 0, n.div_ceil(16), n, k, g, bias, out.as_mut_ptr());
            out.set_len(m * n);
        }
        out
    }

    /// The int8 product (m x n) into `out`.
    fn cols_i8(&self, aq: &[i8], sa: &[f32], m: usize, out: &mut [f32]) {
        let n = self.n;
        // cache blocks: IB activation rows x JB weight rows stay hot while the tiles sweep them
        for i0 in (0..m).step_by(IB) {
            let i1 = (i0 + IB).min(m);
            for j0 in (0..n).step_by(JB) {
                let j1 = (j0 + JB).min(n);
                self.block_i8(aq, sa, i0, i1, j0, j1, out, n, 0);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// Rows i0..m x weight rows j0..n_end into `out` (row stride `stride`, column j at j - col0).
    fn block_i8(&self, aq: &[i8], sa: &[f32], i0: usize, m: usize, j0: usize, n_end: usize, out: &mut [f32], stride: usize, col0: usize) {
        #[cfg(target_arch = "aarch64")]
        if let Some(pk) = &self.apk {
            return crate::arm::block(pk, aq, sa, i0, m, j0, n_end, out, stride, col0);
        }
        let (k, g, n) = (self.k, self.group, stride);
        let ng = k / g;
        let mut j = j0;
        while j < n_end {
            let c = if n_end - j >= TC { TC } else { 1 };
            let (wq, sw) = (&self.q[j * k..(j + c) * k], &self.scale[j * ng..(j + c) * ng]);
            #[cfg(target_arch = "x86_64")]
            let ws: &[i32] = &self.wsum[j * ng..(j + c) * ng];
            #[cfg(not(target_arch = "x86_64"))]
            let ws: &[i32] = &[];
            let mut i = i0;
            while i < m {
                let r = if m - i >= TR { TR } else { 1 };
                let (aq, sa) = (&aq[i * k..(i + r) * k], &sa[i * ng..(i + r) * ng]);
                let o = &mut out[i * n + j - col0..];
                match (r == TR, c == TC) {
                    (true, true) => tile_i8::<TR, TC>(aq, sa, wq, sw, ws, k, g, o, n),
                    (false, true) => tile_i8::<1, TC>(aq, sa, wq, sw, ws, k, g, o, n),
                    (true, false) => tile_i8::<TR, 1>(aq, sa, wq, sw, ws, k, g, o, n),
                    (false, false) => tile_i8::<1, 1>(aq, sa, wq, sw, ws, k, g, o, n),
                }
                i += r;
            }
            j += c;
        }
    }
}

// int8 tile: TR activation rows x TC weight rows per micro-kernel call (2x4: shapes up to 4x4 and 2x8 measured the
// same on an A76; x86-64 AVX-512: 4x4)
#[cfg(not(target_arch = "x86_64"))]
const TR: usize = 2;
#[cfg(target_arch = "x86_64")]
const TR: usize = 4;
const TC: usize = 4;
// cache blocks of the int8 product: activation rows x weight rows
const IB: usize = 64;
const JB: usize = 64;

/// Whether `Quant` multiplies in int8 (aarch64 unless INT8_DOT=0, see `arm`; wasm SIMD128).
fn int8_dot() -> bool {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return true;
    // aarch64: smmla, sdot or plain NEON (`arm::isa`), so always (INT8_DOT=0 turns it off)
    #[cfg(all(target_arch = "aarch64", feature = "std"))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("INT8_DOT").map_or(true, |v| v != "0"))
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "std")))]
    {
        true
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        // the same integer arithmetic, portable (to check accuracy off-device): TINYASR_INT8=1
        if vnni() {
            return true;
        }
        #[cfg(feature = "std")]
        {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var("TINYASR_INT8").is_ok_and(|v| v == "1"))
        }
        #[cfg(not(feature = "std"))]
        {
            false
        }
    }
}

/// Rows of `a` (k each) -> int8 with one scale per `g` values: (values, scales).
fn quantize_rows(a: &[f32], k: usize, g: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; a.len()];
    let mut s = vec![0f32; a.len() / g];
    for ((x, y), s) in a.chunks_exact(g).zip(q.chunks_exact_mut(g)).zip(s.iter_mut()) {
        let max = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        *s = max / 127.0;
        let inv = if max > 0.0 { 127.0 / max } else { 0.0 };
        for (v, o) in x.iter().zip(y.iter_mut()) {
            let r = v * inv;
            *o = (r + if r >= 0.0 { 0.5 } else { -0.5 }) as i8; // round half away from zero
        }
    }
    let _ = k;
    (q, s)
}

/// out[r * stride + c] = sum over groups of (int8 dot of activation row r and weight row c) x both scales,
/// for R activation rows and C weight rows.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tile_i8<const R: usize, const C: usize>(aq: &[i8], sa: &[f32], wq: &[i8], sw: &[f32], ws: &[i32], k: usize, g: usize, out: &mut [f32], stride: usize) {
    let _ = ws;
    assert!(aq.len() >= R * k && wq.len() >= C * k && sa.len() >= R * (k / g) && sw.len() >= C * (k / g));
    assert!(out.len() > (R - 1) * stride + C - 1);
    let mut res = [[0f32; C]; R];
    // aarch64 reaches this only for groups `arm::Pack` cannot take
    #[cfg(target_arch = "aarch64")]
    assert!(crate::arm::isa() >= crate::arm::Isa::Dotprod, "int8 tile without dotprod");
    #[cfg(target_arch = "aarch64")]
    #[allow(unsafe_code)]
    // SAFETY: the dot-product extension was checked just above; all reads are inside the slices asserted above.
    unsafe {
        tile_sdot::<R, C>(aq.as_ptr(), sa.as_ptr(), wq.as_ptr(), sw.as_ptr(), k, g, &mut res)
    };
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[allow(unsafe_code)]
    // SAFETY: all reads are inside the slices asserted above.
    unsafe {
        tile_wasm::<R, C>(aq.as_ptr(), sa.as_ptr(), wq.as_ptr(), sw.as_ptr(), k, g, &mut res)
    };
    #[cfg(target_arch = "x86_64")]
    let portable = !vnni();
    #[cfg(target_arch = "x86_64")]
    if !portable {
        assert!(g % 64 == 0 && ws.len() >= C * (k / g));
        #[allow(unsafe_code)]
        // SAFETY: `vnni()` checked AVX-512 F/BW/VNNI; all reads are inside the slices asserted above.
        unsafe {
            tile_vnni::<R, C>(aq.as_ptr(), sa.as_ptr(), wq.as_ptr(), sw.as_ptr(), ws.as_ptr(), k, g, &mut res)
        };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    let portable = true;
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    if portable {
    for (r, res) in res.iter_mut().enumerate() {
        for (c, res) in res.iter_mut().enumerate() {
            let (a, w) = (&aq[r * k..(r + 1) * k], &wq[c * k..(c + 1) * k]);
            for (p, (a, w)) in a.chunks_exact(g).zip(w.chunks_exact(g)).enumerate() {
                let d: i32 = a.iter().zip(w).map(|(a, w)| *a as i32 * *w as i32).sum();
                *res += d as f32 * (sa[r * (k / g) + p] * sw[c * (k / g) + p]);
            }
        }
    }
    }
    for (r, res) in res.iter().enumerate() {
        out[r * stride..r * stride + C].copy_from_slice(res);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(unsafe_code)]
unsafe fn tile_sdot<const R: usize, const C: usize>(aq: *const i8, sa: *const f32, wq: *const i8, sw: *const f32, k: usize, g: usize, res: &mut [[f32; C]; R]) {
    use core::arch::aarch64::*;
    let ng = k / g;
    // SAFETY: the caller guarantees R rows of k activations / C rows of k weights and their ng scales each.
    unsafe {
        let mut accf = [[vdupq_n_f32(0.0); C]; R];
        for c in 0..ng {
            let mut acci = [[vdupq_n_s32(0); C]; R];
            let mut p = c * g;
            while p < (c + 1) * g {
                let a: [int8x16_t; R] = core::array::from_fn(|r| vld1q_s8(aq.add(r * k + p)));
                let w: [int8x16_t; C] = core::array::from_fn(|j| vld1q_s8(wq.add(j * k + p)));
                for r in 0..R {
                    for j in 0..C {
                        // sdot (vdotq_s32 is still unstable in core::arch)
                        core::arch::asm!("sdot {acc:v}.4s, {a:v}.16b, {w:v}.16b", acc = inout(vreg) acci[r][j], a = in(vreg) a[r], w = in(vreg) w[j], options(pure, nomem, nostack));
                    }
                }
                p += 16;
            }
            for r in 0..R {
                for j in 0..C {
                    accf[r][j] = vfmaq_n_f32(accf[r][j], vcvtq_f32_s32(acci[r][j]), *sa.add(r * ng + c) * *sw.add(j * ng + c));
                }
            }
        }
        for r in 0..R {
            for j in 0..C {
                res[r][j] = vaddvq_f32(accf[r][j]);
            }
        }
    }
}

/// wasm SIMD128 version of `tile_sdot`: int8 widened to int16, `i32x4.dot_i16x8_s` (16 products per 2 dots).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
unsafe fn tile_wasm<const R: usize, const C: usize>(aq: *const i8, sa: *const f32, wq: *const i8, sw: *const f32, k: usize, g: usize, res: &mut [[f32; C]; R]) {
    use core::arch::wasm32::*;
    let ng = k / g;
    // SAFETY: the caller guarantees R rows of k activations / C rows of k weights and their ng scales each.
    unsafe {
        let mut accf = [[f32x4_splat(0.0); C]; R];
        for c in 0..ng {
            let mut acci = [[i32x4_splat(0); C]; R];
            let mut p = c * g;
            while p < (c + 1) * g {
                let a: [(v128, v128); R] = core::array::from_fn(|r| {
                    let v = v128_load(aq.add(r * k + p) as *const v128);
                    (i16x8_extend_low_i8x16(v), i16x8_extend_high_i8x16(v))
                });
                for j in 0..C {
                    let v = v128_load(wq.add(j * k + p) as *const v128);
                    let (lo, hi) = (i16x8_extend_low_i8x16(v), i16x8_extend_high_i8x16(v));
                    for r in 0..R {
                        acci[r][j] = i32x4_add(acci[r][j], i32x4_add(i32x4_dot_i16x8(a[r].0, lo), i32x4_dot_i16x8(a[r].1, hi)));
                    }
                }
                p += 16;
            }
            for r in 0..R {
                for j in 0..C {
                    let s = f32x4_splat(*sa.add(r * ng + c) * *sw.add(j * ng + c));
                    accf[r][j] = f32x4_add(accf[r][j], f32x4_mul(f32x4_convert_i32x4(acci[r][j]), s));
                }
            }
        }
        for r in 0..R {
            for j in 0..C {
                let v = accf[r][j];
                res[r][j] = f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<1>(v) + f32x4_extract_lane::<2>(v) + f32x4_extract_lane::<3>(v);
            }
        }
    }
}

impl Panels for Quant {
    fn n(&self) -> usize {
        self.n
    }
    fn k(&self) -> usize {
        self.k
    }
    fn panel<'s>(&'s self, jp: usize, scratch: &'s mut [f32]) -> &'s [f32] {
        let (n, k, g) = (self.n, self.k, self.group);
        let j0 = jp * NR;
        let cols = NR.min(n - j0);
        if cols < NR {
            scratch.fill(0.0);
        }
        for l in 0..cols {
            let j = j0 + l;
            let (row, sc) = (&self.q[j * k..(j + 1) * k], &self.scale[j * k / g..(j + 1) * k / g]);
            for (c, (q, s)) in row.chunks_exact(g).zip(sc).enumerate() {
                for (p, q) in q.iter().enumerate() {
                    scratch[(c * g + p) * NR + l] = *q as f32 * s;
                }
            }
        }
        scratch
    }
}

pub enum Weight {
    F32(Packed),
    Int(Quant),
}

/// y = x W^T + b, with W in PyTorch Linear layout (n x k).
pub struct Linear {
    pub w: Weight,
    pub b: Option<Vec<f32>>,
}

impl Linear {
    pub fn n(&self) -> usize {
        match &self.w {
            Weight::F32(p) => p.n,
            Weight::Int(q) => q.n,
        }
    }

    /// Input values per row (the weights' row length).
    pub fn k(&self) -> usize {
        match &self.w {
            Weight::F32(p) => p.k,
            Weight::Int(q) => q.k,
        }
    }

    /// x: m rows of k.
    pub fn apply(&self, x: &[f32], m: usize) -> Vec<f32> {
        match &self.w {
            Weight::F32(p) => gemm(x, m, p, self.b.as_deref()),
            Weight::Int(q) => q.gemm(x, m, self.b.as_deref()),
        }
    }
}

/// x86-64 on the AVX-512 VNNI level (`x86::isa`: AVX-512 F, BW and VNNI, not lowered by TINYASR_ISA or TINYASR_INT8=0):
/// `Quant` multiplies in int8 with the kernels of this file.
#[cfg(target_arch = "x86_64")]
#[inline]
fn vnni() -> bool {
    crate::x86::isa() == crate::x86::Isa::Avx512Vnni
}
/// x86-64 on a 256-bit level (AVX2 + FMA or AVX-VNNI, `x86::ymm`); false elsewhere.
#[inline]
fn ymm() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        crate::x86::ymm()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
#[cfg(not(target_arch = "x86_64"))]
fn vnni() -> bool {
    false
}

/// x86-64 AVX-512 VNNI version of `tile_sdot`. `vpdpbusd` multiplies unsigned by signed bytes, so the activations
/// are shifted to unsigned (x + 128, by flipping the top bit) and each group's accumulator starts at -128 x the sum of
/// its weights (`ws`), which makes the integer dot product exact.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn tile_vnni<const R: usize, const C: usize>(aq: *const i8, sa: *const f32, wq: *const i8, sw: *const f32, ws: *const i32, k: usize, g: usize, res: &mut [[f32; C]; R]) {
    use core::arch::x86_64::*;
    let ng = k / g;
    // SAFETY: the caller guarantees R rows of k activations / C rows of k weights, their ng scales and weight sums.
    unsafe {
        let flip = _mm512_set1_epi8(-128);
        let mut accf = [[_mm512_setzero_ps(); C]; R];
        for c in 0..ng {
            let mut init = [_mm512_setzero_si512(); C];
            for j in 0..C {
                init[j] = _mm512_maskz_set1_epi32(1, *ws.add(j * ng + c));
            }
            let mut acci = [init; R];
            let mut p = c * g;
            while p < (c + 1) * g {
                let mut a = [_mm512_setzero_si512(); R];
                for r in 0..R {
                    a[r] = _mm512_xor_si512(_mm512_loadu_si512(aq.add(r * k + p).cast()), flip);
                }
                let mut w = [_mm512_setzero_si512(); C];
                for j in 0..C {
                    w[j] = _mm512_loadu_si512(wq.add(j * k + p).cast());
                }
                for r in 0..R {
                    for j in 0..C {
                        acci[r][j] = _mm512_dpbusd_epi32(acci[r][j], a[r], w[j]);
                    }
                }
                p += 64;
            }
            for r in 0..R {
                for j in 0..C {
                    let s = _mm512_set1_ps(*sa.add(r * ng + c) * *sw.add(j * ng + c));
                    accf[r][j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acci[r][j]), s, accf[r][j]);
                }
            }
        }
        for r in 0..R {
            for j in 0..C {
                res[r][j] = _mm512_reduce_add_ps(accf[r][j]);
            }
        }
    }
}

/// Rows (k each) -> unsigned bytes x + 128 (x: int8 with one scale per `g` values, as `quantize_rows`) and scales.
#[cfg(target_arch = "x86_64")]
fn quantize_rows_u8(a: &[f32], g: usize) -> (Vec<u8>, Vec<f32>) {
    let mut q = vec![0u8; a.len()];
    let mut s = vec![0f32; a.len() / g];
    quantize_u8_into(a, g, &mut q, &mut s);
    (q, s)
}

/// |x| max of a group, lane-wise so it vectorizes (max is exact: same result as a sequential fold).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn absmax(x: &[f32]) -> f32 {
    let mut l = [0f32; 16];
    let c = x.chunks_exact(16);
    let rest = c.remainder();
    for v in c {
        for j in 0..16 {
            l[j] = l[j].max(v[j].abs());
        }
    }
    let mut m = rest.iter().fold(0f32, |m, v| m.max(v.abs()));
    for v in l {
        m = m.max(v);
    }
    m
}

#[cfg(target_arch = "x86_64")]
/// `quantize_rows_u8` into given buffers (a.len() bytes, a.len() / g scales).
fn quantize_u8_into(a: &[f32], g: usize, q: &mut [u8], s: &mut [f32]) {
    if g % 16 == 0 && vnni() {
        #[allow(unsafe_code)]
        // SAFETY: `vnni()` checked AVX-512 F/BW; q has a.len() bytes, s a.len() / g values.
        unsafe {
            quantize_u8_avx512(a, g, q, s)
        };
        return;
    }
    if crate::x86::quantize_u8(a, g, q, s) {
        return;
    }
    for ((x, y), s) in a.chunks_exact(g).zip(q.chunks_exact_mut(g)).zip(s.iter_mut()) {
        let max = absmax(x);
        *s = max / 127.0;
        let inv = if max > 0.0 { 127.0 / max } else { 0.0 };
        for (v, o) in x.iter().zip(y.iter_mut()) {
            let r = v * inv;
            *o = ((r + 0.5f32.copysign(r)) as i32 + 128) as u8; // round half away from zero, branch-free
        }
    }
}

/// A raw output pointer shared by tasks that write disjoint blocks.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
struct SendPtr(*mut f32);
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[allow(unsafe_code)]
// SAFETY: only used to hand disjoint output blocks to pool tasks (see gemm_pk).
unsafe impl Sync for SendPtr {}

/// Packed product: least multiply-adds worth splitting, about the work per task, fewest rows per row block.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
const PK_PAR_WORK: usize = 1 << 19;
/// The same for products of fewer than 2 x PR rows (decoder steps): below this the pool's dispatch costs more than it
/// saves (measured: LSTM, projection and joiner products of 1-4 rows x 512 x 512..2048 are faster on one thread).
#[cfg(all(target_arch = "x86_64", feature = "std"))]
const PK_PAR_WORK_FEW: usize = 1 << 22;
#[cfg(all(target_arch = "x86_64", feature = "std"))]
const PK_TASK_WORK: usize = 1 << 18;
#[cfg(all(target_arch = "x86_64", feature = "std"))]
const PK_MIN_ROWS: usize = 64;
#[cfg(all(target_arch = "x86_64", feature = "std"))]
/// Activation quantization: fewest rows per task, least values worth splitting.
const PK_QUANT_ROWS: usize = 16;
#[cfg(all(target_arch = "x86_64", feature = "std"))]
const PK_QUANT_WORK: usize = 1 << 16;

/// `pk_block` on the level of this run: AVX-512 VNNI here, the 256-bit levels in `x86` (the same layout and results).
/// Safety: as `pk_block`.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn pk_run(pk: &PackI8, aq: &[u8], sa: &[f32], r0: usize, r1: usize, p_lo: usize, p_hi: usize, n: usize, k: usize, g: usize, bias: Option<&[f32]>, out: *mut f32) {
    // SAFETY: `vnni()` checked AVX-512 F/BW/VNNI for `pk_block`; `x86::pk_block` checks its own level and sizes; the
    // output contract is the caller's.
    unsafe {
        if vnni() {
            pk_block(pk, aq, sa, r0, r1, p_lo, p_hi, n, k, g, bias, out)
        } else {
            crate::x86::pk_block(&pk.w, &pk.sw, &pk.ws, pk.steps, aq, sa, r0, r1, p_lo, p_hi, n, k, g, bias, out)
        }
    }
}

/// Activation rows of the packed product per micro-kernel call, and rows per cache block.
#[cfg(target_arch = "x86_64")]
const PR: usize = 4;
#[cfg(target_arch = "x86_64")]
const PB: usize = 64;

/// out (rows x n) = activations (rows x k, unsigned, see `quantize_rows_u8`) x packed weights (+ bias): rows r0..r1 and
/// 16-column panels p_lo..p_hi only, into `out` (the whole rows x n product, row stride
/// n) through a raw pointer, so several threads can fill disjoint blocks of one output.
/// Safety: `out` points at rows x n floats (rows >= r1) and no other thread writes this block meanwhile.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn pk_block(pk: &PackI8, aq: &[u8], sa: &[f32], r0: usize, r1: usize, p_lo: usize, p_hi: usize, n: usize, k: usize, g: usize, bias: Option<&[f32]>, out: *mut f32) {
    let (np, ng) = (n.div_ceil(16), k / g);
    let rows = r1;
    assert!(r0 <= r1 && p_lo <= p_hi && p_hi <= np);
    assert!(aq.len() >= rows * k && sa.len() >= rows * ng);
    assert!(pk.w.len() >= np * k * 16 && pk.sw.len() >= np * ng * 16 && pk.ws.len() >= np * ng * 16 && g % 4 == 0);
    if let Some(b) = bias {
        assert!(b.len() >= n);
    }
    for i0 in (r0..rows).step_by(PB) {
        let i1 = (i0 + PB).min(rows);
        let mut p = p_lo;
        while p < p_hi {
            let pp = if p_hi - p >= 2 { 2 } else { 1 };
            let mut i = i0;
            while i < i1 {
                let r = (i1 - i).min(PR);
                // SAFETY: rows i..i+r < rows, panels p..p+pp < np, all inside the slices asserted above.
                unsafe {
                    let args = (aq.as_ptr().add(i * k), sa.as_ptr().add(i * ng), pk, p, k, g, out.add(i * n), n, bias);
                    match (r, pp) {
                        (4, 2) => tile_pk::<4, 2>(args),
                        (4, 1) => tile_pk::<4, 1>(args),
                        (3, 2) => tile_pk::<3, 2>(args),
                        (3, 1) => tile_pk::<3, 1>(args),
                        (2, 2) => tile_pk::<2, 2>(args),
                        (2, 1) => tile_pk::<2, 1>(args),
                        (1, 2) => tile_pk::<1, 2>(args),
                        _ => tile_pk::<1, 1>(args),
                    }
                }
                i += r;
            }
            p += pp;
        }
    }
}

#[cfg(target_arch = "x86_64")]
type PkArgs<'a> = (*const u8, *const f32, &'a PackI8, usize, usize, usize, *mut f32, usize, Option<&'a [f32]>);

/// R activation rows x P panels of 16 columns: per 4 inputs one broadcast of each row's 4 bytes and one `vpdpbusd` per
/// row and panel; per group of g inputs the integer sums (started at -128 x the weight sums, so exact) are scaled by
/// both scales into float accumulators.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
#[inline]
#[allow(unsafe_code)]
unsafe fn tile_pk<const R: usize, const P: usize>((a, sa, pk, p0, k, g, out, n, bias): PkArgs) {
    use core::arch::x86_64::*;
    let ng = k / g;
    // SAFETY: the caller guarantees R rows of k bytes / ng scales at a / sa, panels p0..p0+P of `pk`, R rows of n at out.
    unsafe {
        let w = pk.w.as_ptr().add(p0 * k * 16);
        let sw = pk.sw.as_ptr().add(p0 * ng * 16);
        let ws = pk.ws.as_ptr().add(p0 * ng * 16);
        let mut accf = [[_mm512_setzero_ps(); P]; R];
        for c in 0..ng {
            let mut acci = [[_mm512_setzero_si512(); P]; R];
            for q in 0..P {
                let init = _mm512_loadu_si512(ws.add((q * ng + c) * 16).cast());
                for r in 0..R {
                    acci[r][q] = init;
                }
            }
            let mut kk = c * g;
            while kk < (c + 1) * g {
                let mut wv = [_mm512_setzero_si512(); P];
                for q in 0..P {
                    wv[q] = _mm512_loadu_si512(w.add(q * k * 16 + kk * 16).cast());
                }
                for r in 0..R {
                    let av = _mm512_set1_epi32(core::ptr::read_unaligned(a.add(r * k + kk) as *const i32));
                    for q in 0..P {
                        acci[r][q] = _mm512_dpbusd_epi32(acci[r][q], av, wv[q]);
                    }
                }
                kk += 4;
            }
            for q in 0..P {
                let s = _mm512_loadu_ps(sw.add((q * ng + c) * 16));
                for r in 0..R {
                    let sc = _mm512_mul_ps(s, _mm512_set1_ps(*sa.add(r * ng + c)));
                    accf[r][q] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acci[r][q]), sc, accf[r][q]);
                }
            }
        }
        for q in 0..P {
            let j0 = (p0 + q) * 16;
            let cols = 16.min(n - j0);
            let mask: __mmask16 = if cols == 16 { 0xffff } else { (1u16 << cols) - 1 };
            let b = match bias {
                Some(b) => _mm512_maskz_loadu_ps(mask, b.as_ptr().add(j0)),
                None => _mm512_setzero_ps(),
            };
            for r in 0..R {
                _mm512_mask_storeu_ps(out.add(r * n + j0), mask, _mm512_add_ps(accf[r][q], b));
            }
        }
    }
}

/// `quantize_rows_u8` with AVX-512: the same arithmetic (max |x| per group, x * 127 / max rounded half away from
/// zero, + 128), 16 values at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(unsafe_code)]
unsafe fn quantize_u8_avx512(a: &[f32], g: usize, q: &mut [u8], s: &mut [f32]) {
    use core::arch::x86_64::*;
    assert!(g % 16 == 0 && q.len() >= a.len() && s.len() >= a.len() / g);
    // SAFETY: every access is within a / q (len a.len()) and s (a.len() / g), checked above.
    unsafe {
        let half = _mm512_set1_ps(0.5);
        let signm = _mm512_set1_epi32(i32::MIN);
        let off = _mm512_set1_epi32(128);
        for (gi, x) in a.chunks_exact(g).enumerate() {
            let p = x.as_ptr();
            let mut mx = _mm512_setzero_ps();
            let mut i = 0;
            while i < g {
                // |x| with AVX-512 F (`_mm512_and_ps` is AVX-512 DQ, which `vnni()` does not check)
                mx = _mm512_max_ps(mx, _mm512_abs_ps(_mm512_loadu_ps(p.add(i))));
                i += 16;
            }
            let max = _mm512_reduce_max_ps(mx);
            *s.get_unchecked_mut(gi) = max / 127.0;
            let inv = _mm512_set1_ps(if max > 0.0 { 127.0 / max } else { 0.0 });
            let o = q.as_mut_ptr().add(gi * g);
            let mut i = 0;
            while i < g {
                let r = _mm512_mul_ps(_mm512_loadu_ps(p.add(i)), inv);
                // r + copysign(0.5, r), truncated toward zero
                let h = _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(half), _mm512_and_si512(_mm512_castps_si512(r), signm)));
                let v = _mm512_add_epi32(_mm512_cvttps_epi32(_mm512_add_ps(r, h)), off);
                _mm_storeu_si128(o.add(i).cast(), _mm512_cvtepi32_epi8(v));
                i += 16;
            }
        }
    }
}

/// x = softplus(x - shift) - 0.08 x - offset, in place (the Zipformer SwooshL / SwooshR activations), with the same
/// exp / ln approximations as `exp` and `ln`; AVX-512 when present (the same operations, so the same results).
pub fn swoosh(x: &mut [f32], shift: f32, offset: f32) {
    #[cfg(target_arch = "x86_64")]
    if vnni() {
        #[allow(unsafe_code)]
        // SAFETY: `vnni()` checked AVX-512 F.
        unsafe {
            swoosh_avx512(x, shift, offset)
        };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    let d = crate::x86::swoosh(x, shift, offset);
    #[cfg(target_arch = "aarch64")]
    let d = crate::arm::swoosh(x, shift, offset);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let d = 0;
    x[d..].iter_mut().for_each(|v| *v = softplus(*v - shift) - 0.08 * *v - offset);
}

#[inline(always)]
fn softplus(x: f32) -> f32 {
    x.max(0.0) + ln(1.0 + exp(-x.abs()))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, clippy::approx_constant)]
unsafe fn swoosh_avx512(x: &mut [f32], shift: f32, offset: f32) {
    use core::arch::x86_64::*;
    let n16 = x.len() / 16 * 16;
    // SAFETY: loads and stores at i < n16 <= x.len(), 16 at a time.
    unsafe {
        let p = x.as_mut_ptr();
        let (sh, of, k08) = (_mm512_set1_ps(shift), _mm512_set1_ps(offset), _mm512_set1_ps(0.08));
        let (one, two, zero) = (_mm512_set1_ps(1.0), _mm512_set1_ps(2.0), _mm512_setzero_ps());
        let (lo, hi) = (_mm512_set1_ps(-126.0), _mm512_set1_ps(126.0));
        let log2e = _mm512_set1_ps(core::f32::consts::LOG2_E);
        let ln2 = _mm512_set1_ps(core::f32::consts::LN_2);
        let c = [0.693_147_2f32, 0.240_226_5, 0.055_504_11, 0.009_618_129, 0.001_333_355].map(|v| _mm512_set1_ps(v));
        let d = [0.666_666_7f32, 0.4, 0.285_714_3, 0.222_222_2, 0.181_818_2].map(|v| _mm512_set1_ps(v));
        let bias127 = _mm512_set1_epi32(127);
        let sqh = _mm512_set1_epi32(0x3f35_04f3);
        let mut i = 0;
        while i < n16 {
            let v = _mm512_loadu_ps(p.add(i));
            let y = _mm512_sub_ps(v, sh);
            // exp(-|y|)
            let ny = _mm512_sub_ps(zero, _mm512_abs_ps(y)); // AVX-512 F (`_mm512_and_ps` would need DQ)
            let t = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(ny, log2e), lo), hi);
            let fi = _mm512_roundscale_ps::<0x09>(t); // floor
            let f = _mm512_sub_ps(t, fi);
            let mut q = _mm512_mul_ps(f, c[4]);
            q = _mm512_mul_ps(f, _mm512_add_ps(c[3], q));
            q = _mm512_mul_ps(f, _mm512_add_ps(c[2], q));
            q = _mm512_mul_ps(f, _mm512_add_ps(c[1], q));
            q = _mm512_mul_ps(f, _mm512_add_ps(c[0], q));
            q = _mm512_add_ps(one, q);
            let pw = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(_mm512_cvttps_epi32(fi), bias127)));
            let e = _mm512_mul_ps(pw, q);
            // ln(1 + e)
            let xl = _mm512_add_ps(one, e);
            let b = _mm512_castps_si512(xl);
            let ex = _mm512_srai_epi32::<23>(_mm512_sub_epi32(b, sqh));
            let m = _mm512_castsi512_ps(_mm512_sub_epi32(b, _mm512_slli_epi32::<23>(ex)));
            let s = _mm512_div_ps(_mm512_sub_ps(m, one), _mm512_add_ps(m, one));
            let z = _mm512_mul_ps(s, s);
            let mut r = _mm512_mul_ps(z, d[4]);
            r = _mm512_mul_ps(z, _mm512_add_ps(d[3], r));
            r = _mm512_mul_ps(z, _mm512_add_ps(d[2], r));
            r = _mm512_mul_ps(z, _mm512_add_ps(d[1], r));
            r = _mm512_mul_ps(z, _mm512_add_ps(d[0], r));
            r = _mm512_add_ps(two, r);
            let lnv = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(ex), ln2), _mm512_mul_ps(s, r));
            let sp = _mm512_add_ps(_mm512_max_ps(y, zero), lnv);
            let out = _mm512_sub_ps(_mm512_sub_ps(sp, _mm512_mul_ps(k08, v)), of);
            _mm512_storeu_ps(p.add(i), out);
            i += 16;
        }
    }
    for v in &mut x[n16..] {
        *v = softplus(*v - shift) - 0.08 * *v - offset;
    }
}

/// exp of 16 lanes, the same arithmetic as `exp` (clamp, 2^floor, degree-5 polynomial, no fused multiply-add).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
#[allow(unsafe_code)]
unsafe fn exp16(x: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512 {
    use core::arch::x86_64::*;
    let t = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(x, _mm512_set1_ps(core::f32::consts::LOG2_E)), _mm512_set1_ps(-126.0)), _mm512_set1_ps(126.0));
    let fi = _mm512_roundscale_ps::<0x09>(t);
    let f = _mm512_sub_ps(t, fi);
    let mut q = _mm512_mul_ps(f, _mm512_set1_ps(0.001_333_355));
    q = _mm512_mul_ps(f, _mm512_add_ps(_mm512_set1_ps(0.009_618_129), q));
    q = _mm512_mul_ps(f, _mm512_add_ps(_mm512_set1_ps(0.055_504_11), q));
    q = _mm512_mul_ps(f, _mm512_add_ps(_mm512_set1_ps(0.240_226_5), q));
    q = _mm512_mul_ps(f, _mm512_add_ps(_mm512_set1_ps(0.693_147_2), q));
    q = _mm512_add_ps(_mm512_set1_ps(1.0), q);
    let pw = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(_mm512_cvttps_epi32(fi), _mm512_set1_epi32(127))));
    _mm512_mul_ps(pw, q)
}

/// tanh of 16 lanes as `tanh`: 1 - 2 / (exp(2x) + 1).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
#[allow(unsafe_code)]
unsafe fn tanh16(x: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512 {
    use core::arch::x86_64::*;
    // SAFETY: register-only arithmetic.
    unsafe {
        let e = exp16(_mm512_mul_ps(_mm512_set1_ps(2.0), x));
        _mm512_sub_ps(_mm512_set1_ps(1.0), _mm512_div_ps(_mm512_set1_ps(2.0), _mm512_add_ps(e, _mm512_set1_ps(1.0))))
    }
}

/// sigmoid of 16 lanes as `sigmoid`: 1 / (1 + exp(-x)).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
#[allow(unsafe_code)]
unsafe fn sigmoid16(x: core::arch::x86_64::__m512) -> core::arch::x86_64::__m512 {
    use core::arch::x86_64::*;
    // SAFETY: register-only arithmetic.
    unsafe {
        let e = exp16(_mm512_sub_ps(_mm512_setzero_ps(), x));
        _mm512_div_ps(_mm512_set1_ps(1.0), _mm512_add_ps(_mm512_set1_ps(1.0), e))
    }
}

/// ln(sum exp(v)) (stable: max first). AVX-512 when present (summation order differs from the scalar loop).
pub fn logsumexp(v: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if vnni() && v.len() >= 16 {
        #[allow(unsafe_code)]
        // SAFETY: `vnni()` checked AVX-512 F.
        return unsafe { logsumexp_avx512(v) };
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(r) = crate::x86::logsumexp(v) {
        return r;
    }
    #[cfg(target_arch = "aarch64")]
    if v.len() >= 4 {
        return crate::arm::logsumexp(v);
    }
    let m = v.iter().cloned().fold(f32::MIN, f32::max);
    m + v.iter().map(|x| exp(x - m)).sum::<f32>().ln()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn logsumexp_avx512(v: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let n16 = v.len() / 16 * 16;
    // SAFETY: loads at i < n16 <= v.len().
    unsafe {
        let p = v.as_ptr();
        let mut mx = _mm512_set1_ps(f32::MIN);
        let mut i = 0;
        while i < n16 {
            mx = _mm512_max_ps(mx, _mm512_loadu_ps(p.add(i)));
            i += 16;
        }
        let mut m = _mm512_reduce_max_ps(mx);
        for &x in &v[n16..] {
            m = m.max(x);
        }
        let mv = _mm512_set1_ps(m);
        let mut s = _mm512_setzero_ps();
        let mut i = 0;
        while i < n16 {
            s = _mm512_add_ps(s, exp16(_mm512_sub_ps(_mm512_loadu_ps(p.add(i)), mv)));
            i += 16;
        }
        let mut sum = _mm512_reduce_add_ps(s);
        for &x in &v[n16..] {
            sum += exp(x - m);
        }
        m + sum.ln()
    }
}

/// out[k] = tanh(a[k] + b[k]).
pub fn tanh_sum(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    #[cfg_attr(target_arch = "x86_64", allow(unused_assignments))]
    let mut i0 = 0;
    #[cfg(target_arch = "x86_64")]
    if vnni() {
        i0 = n / 16 * 16;
        #[allow(unsafe_code)]
        // SAFETY: `vnni()` checked AVX-512 F; loads / stores at i < i0 <= n.
        unsafe { tanh_sum_avx512(a, b, out, i0) }
    } else {
        i0 = crate::x86::tanh_sum(a, b, out);
    }
    #[cfg(target_arch = "aarch64")]
    {
        i0 += crate::arm::tanh_sum(&a[..n], &b[..n], &mut out[..n]);
    }
    for i in i0..n {
        out[i] = tanh(a[i] + b[i]);
    }
}

/// `tanh_sum` for i < i0 (the caller checked AVX-512 F and i0 <= all lengths). A `target_feature` function, so the
/// intrinsics and `tanh16` inline into the loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn tanh_sum_avx512(a: &[f32], b: &[f32], out: &mut [f32], i0: usize) {
    use core::arch::x86_64::*;
    assert!(i0 <= a.len() && i0 <= b.len() && i0 <= out.len());
    let mut i = 0;
    // SAFETY: i + 16 <= i0 (i0 is a multiple of 16) <= every length (asserted).
    unsafe {
        while i < i0 {
            let x = _mm512_add_ps(_mm512_loadu_ps(a.as_ptr().add(i)), _mm512_loadu_ps(b.as_ptr().add(i)));
            _mm512_storeu_ps(out.as_mut_ptr().add(i), tanh16(x));
            i += 16;
        }
    }
}

/// LSTM cell update from the gate pre-activations g (i, f, g, o; 4 x n): c = f c + i g, h = o tanh(c).
pub fn lstm_cell(g: &[f32], c: &mut [f32], h: &mut [f32]) {
    let n = c.len();
    assert!(g.len() >= 4 * n && h.len() >= n);
    #[cfg_attr(target_arch = "x86_64", allow(unused_assignments))]
    let mut j0 = 0;
    #[cfg(target_arch = "x86_64")]
    if vnni() {
        j0 = n / 16 * 16;
        #[allow(unsafe_code)]
        // SAFETY: `vnni()` checked AVX-512 F; all accesses at j < j0 <= n (g at up to 3 n + j), asserted above.
        unsafe { lstm_cell_avx512(g, c, h, j0) }
    } else {
        j0 = crate::x86::lstm_cell(g, c, h);
    }
    #[cfg(target_arch = "aarch64")]
    {
        j0 += crate::arm::lstm_cell(g, c, h);
    }
    for j in j0..n {
        let (i, f, gg, o) = (sigmoid(g[j]), sigmoid(g[n + j]), tanh(g[2 * n + j]), sigmoid(g[3 * n + j]));
        c[j] = f * c[j] + i * gg;
        h[j] = o * tanh(c[j]);
    }
}

/// `lstm_cell` for j < j0 (the caller checked AVX-512 F; j0 <= n = c.len(), g.len() >= 4 n, h.len() >= n).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn lstm_cell_avx512(g: &[f32], c: &mut [f32], h: &mut [f32], j0: usize) {
    use core::arch::x86_64::*;
    let n = c.len();
    assert!(j0 <= n && g.len() >= 4 * n && h.len() >= n);
    let gp = g.as_ptr();
    let mut j = 0;
    // SAFETY: j + 16 <= j0 <= n; g read at up to 3 n + j + 16 <= 4 n (asserted).
    unsafe {
        while j < j0 {
            let i = sigmoid16(_mm512_loadu_ps(gp.add(j)));
            let f = sigmoid16(_mm512_loadu_ps(gp.add(n + j)));
            let gg = tanh16(_mm512_loadu_ps(gp.add(2 * n + j)));
            let o = sigmoid16(_mm512_loadu_ps(gp.add(3 * n + j)));
            let cv = _mm512_add_ps(_mm512_mul_ps(f, _mm512_loadu_ps(c.as_ptr().add(j))), _mm512_mul_ps(i, gg));
            _mm512_storeu_ps(c.as_mut_ptr().add(j), cv);
            _mm512_storeu_ps(h.as_mut_ptr().add(j), _mm512_mul_ps(o, tanh16(cv)));
            j += 16;
        }
    }
}

/// Softmax of one row in place (max subtracted first). AVX-512 when present (summation order differs from scalar).
pub fn softmax_row(row: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if vnni() && row.len() >= 16 {
        #[allow(unsafe_code)]
        // SAFETY: `vnni()` checked AVX-512 F.
        unsafe { softmax_row_avx512(row) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if crate::x86::softmax_row(row) {
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if row.len() >= 4 {
        return crate::arm::softmax_row(row);
    }
    let m = row.iter().cloned().fold(f32::MIN, f32::max);
    let mut s = 0.0;
    row.iter_mut().for_each(|v| {
        *v = exp(*v - m);
        s += *v;
    });
    let inv = 1.0 / s;
    row.iter_mut().for_each(|v| *v *= inv);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn softmax_row_avx512(row: &mut [f32]) {
    use core::arch::x86_64::*;
    let n = row.len();
    let n16 = n / 16 * 16;
    // SAFETY: accesses at i < n16 <= n, tail by index.
    unsafe {
        let p = row.as_mut_ptr();
        let mut mx = _mm512_set1_ps(f32::MIN);
        let mut i = 0;
        while i < n16 {
            mx = _mm512_max_ps(mx, _mm512_loadu_ps(p.add(i)));
            i += 16;
        }
        let mut m = _mm512_reduce_max_ps(mx);
        for j in n16..n {
            m = m.max(*p.add(j));
        }
        let mv = _mm512_set1_ps(m);
        let mut s = _mm512_setzero_ps();
        let mut i = 0;
        while i < n16 {
            let e = exp16(_mm512_sub_ps(_mm512_loadu_ps(p.add(i)), mv));
            _mm512_storeu_ps(p.add(i), e);
            s = _mm512_add_ps(s, e);
            i += 16;
        }
        let mut sum = _mm512_reduce_add_ps(s);
        for j in n16..n {
            let e = exp(*p.add(j) - m);
            *p.add(j) = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        let iv = _mm512_set1_ps(inv);
        let mut i = 0;
        while i < n16 {
            _mm512_storeu_ps(p.add(i), _mm512_mul_ps(_mm512_loadu_ps(p.add(i)), iv));
            i += 16;
        }
        for j in n16..n {
            *p.add(j) *= inv;
        }
    }
}

/// Depthwise taps: out[c] = start[c] + sum over u < taps of w[u d + c] x[u d + c], d = out.len(); `start` is `init`, or
/// `out` itself (accumulate) when None. AVX-512 (fused multiply-add, accumulators in registers) when present.
pub fn dw_taps(out: &mut [f32], init: Option<&[f32]>, w: &[f32], x: &[f32], taps: usize) {
    let d = out.len();
    assert!(w.len() >= taps * d && x.len() >= taps * d && init.is_none_or(|v| v.len() >= d));
    #[cfg(target_arch = "x86_64")]
    if vnni() && d % 16 == 0 {
        #[allow(unsafe_code)]
        // SAFETY: `vnni()` checked AVX-512 F; sizes checked above, d a multiple of 16.
        unsafe { dw_taps_avx512(out, init, w, x, taps) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if crate::x86::dw_taps(out, init, w, x, taps) {
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if crate::arm::dw_taps(out, init, w, x, taps) {
        return;
    }
    if let Some(v) = init {
        out.copy_from_slice(&v[..d]);
    }
    for u in 0..taps {
        for ((o, w), x) in out.iter_mut().zip(&w[u * d..(u + 1) * d]).zip(&x[u * d..(u + 1) * d]) {
            *o += w * x;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn dw_taps_avx512(out: &mut [f32], init: Option<&[f32]>, w: &[f32], x: &[f32], taps: usize) {
    use core::arch::x86_64::*;
    let d = out.len();
    // SAFETY: the caller checked w, x >= taps d, init >= d, d % 16 == 0; every access is below those bounds.
    unsafe {
        let (op, wp, xp) = (out.as_mut_ptr(), w.as_ptr(), x.as_ptr());
        let sp = init.map_or(op as *const f32, |v| v.as_ptr());
        let mut c = 0;
        while c + 64 <= d {
            let mut acc = [_mm512_loadu_ps(sp.add(c)), _mm512_loadu_ps(sp.add(c + 16)), _mm512_loadu_ps(sp.add(c + 32)), _mm512_loadu_ps(sp.add(c + 48))];
            for u in 0..taps {
                let (wr, xr) = (wp.add(u * d + c), xp.add(u * d + c));
                for q in 0..4 {
                    acc[q] = _mm512_fmadd_ps(_mm512_loadu_ps(wr.add(16 * q)), _mm512_loadu_ps(xr.add(16 * q)), acc[q]);
                }
            }
            for q in 0..4 {
                _mm512_storeu_ps(op.add(c + 16 * q), acc[q]);
            }
            c += 64;
        }
        while c < d {
            let mut acc = _mm512_loadu_ps(sp.add(c));
            for u in 0..taps {
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(wp.add(u * d + c)), _mm512_loadu_ps(xp.add(u * d + c)), acc);
            }
            _mm512_storeu_ps(op.add(c), acc);
            c += 16;
        }
    }
}

/// out[c] = a[c] * sigmoid(b[c]) (gated linear unit), `sigmoid` arithmetic; AVX-512 when present.
pub fn glu(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = out.len();
    assert!(a.len() >= n && b.len() >= n);
    #[cfg_attr(target_arch = "x86_64", allow(unused_assignments))]
    let mut i0 = 0;
    #[cfg(target_arch = "x86_64")]
    if vnni() {
        i0 = n / 16 * 16;
        #[allow(unsafe_code)]
        // SAFETY: `vnni()` checked AVX-512 F; accesses at i < i0 <= n (lengths asserted above).
        unsafe { glu_avx512(a, b, out, i0) }
    } else {
        i0 = crate::x86::glu(a, b, out);
    }
    #[cfg(target_arch = "aarch64")]
    {
        i0 += crate::arm::glu(a, b, out);
    }
    for i in i0..n {
        out[i] = a[i] * sigmoid(b[i]);
    }
}

/// `glu` for i < i0 (the caller checked AVX-512 F and i0 <= all lengths).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn glu_avx512(a: &[f32], b: &[f32], out: &mut [f32], i0: usize) {
    use core::arch::x86_64::*;
    assert!(i0 <= a.len() && i0 <= b.len() && i0 <= out.len());
    let mut i = 0;
    // SAFETY: i + 16 <= i0 <= every length (asserted).
    unsafe {
        while i < i0 {
            let s = sigmoid16(_mm512_loadu_ps(b.as_ptr().add(i)));
            _mm512_storeu_ps(out.as_mut_ptr().add(i), _mm512_mul_ps(_mm512_loadu_ps(a.as_ptr().add(i)), s));
            i += 16;
        }
    }
}

/// Whether `conv0_rows` runs (AVX-512 or the 256-bit levels; the arithmetic of the f32 product it replaces).
pub fn conv0_direct_ok() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        vnni() || crate::x86::ymm()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The embed's first convolution (1 -> 8 channels, 3x3, frequency padding 1) for output time steps i0 .. i0 + nt of
/// features x (rows of f): out (nt x f x 8) = bias + sum over taps a*3 + b (in that order, one fused multiply-add
/// each) of w[co][a*3 + b] x[i + a][j + b - 1] (zero outside the rows): the same operations, in the same order, as
/// the f32 product of the im2col rows (`blocked_avx512`), so the same results. Needs `conv0_direct_ok()`.
pub fn conv0_rows(x: &[f32], f: usize, i0: usize, nt: usize, w: &[f32], bias: &[f32], out: &mut [f32]) {
    assert!(w.len() >= 72 && bias.len() >= 8 && out.len() >= nt * f * 8 && x.len() >= (i0 + nt + 2) * f && f >= 1);
    #[cfg(target_arch = "x86_64")]
    {
        assert!(conv0_direct_ok());
        if !vnni() {
            let done = crate::x86::conv0_rows(x, f, i0, nt, w, bias, out);
            assert!(done);
            return;
        }
        #[allow(unsafe_code)]
        // SAFETY: `vnni()` checked AVX-512 F; sizes checked above (rows i0 .. i0 + nt + 2 of x, nt f 8 outputs).
        unsafe {
            conv0_avx512(x, f, i0, nt, w, bias, out)
        };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn conv0_avx512(x: &[f32], f: usize, i0: usize, nt: usize, w: &[f32], bias: &[f32], out: &mut [f32]) {
    use core::arch::x86_64::*;
    // lanes: (output column j, channel 0..8) and (j + 1, channel 0..8)
    // SAFETY: x reads at rows i0 + ii + a < i0 + nt + 2 and columns 0 .. f (checked by the caller); out writes at
    // (ii f + j) 8 .. + 16 for j + 1 < f, or 8 lanes (masked) for the last odd column: inside nt f 8.
    unsafe {
        let mut wv = [_mm512_setzero_ps(); 9];
        for (p, v) in wv.iter_mut().enumerate() {
            let mut l = [0f32; 16];
            for co in 0..8 {
                l[co] = w[co * 9 + p];
                l[8 + co] = w[co * 9 + p];
            }
            *v = _mm512_loadu_ps(l.as_ptr());
        }
        let mut bl = [0f32; 16];
        bl[..8].copy_from_slice(&bias[..8]);
        bl[8..].copy_from_slice(&bias[..8]);
        let bv = _mm512_loadu_ps(bl.as_ptr());
        let xp = x.as_ptr();
        let at = |r: usize, c: isize| -> f32 { if c >= 0 && (c as usize) < f { *xp.add(r * f + c as usize) } else { 0.0 } };
        for ii in 0..nt {
            let i = i0 + ii;
            let mut j = 0;
            while j < f {
                let mut acc = bv;
                for a in 0..3 {
                    for b in 0..3 {
                        let c = j as isize + b as isize - 1;
                        let xv = _mm512_mask_broadcastss_ps(_mm512_set1_ps(at(i + a, c)), 0xff00, _mm_set_ss(at(i + a, c + 1)));
                        acc = _mm512_fmadd_ps(xv, wv[a * 3 + b], acc);
                    }
                }
                let o = out.as_mut_ptr().add((ii * f + j) * 8);
                if j + 1 < f {
                    _mm512_storeu_ps(o, acc);
                } else {
                    _mm512_mask_storeu_ps(o, 0x00ff, acc);
                }
                j += 2;
            }
        }
    }
}
