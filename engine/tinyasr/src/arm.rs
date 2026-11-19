//! aarch64 kernels picked at run time.
//!
//! int8 products on weights repacked once at load time in panels of 16 output columns (per 4 inputs, the 4 bytes of
//! each of the 16 columns: the x86 `PackI8` layout, minus its weight sums since `sdot` multiplies signed by signed):
//! - i8mm: `smmla` (2 x 8 by 8 x 2 per instruction, 32 multiply-adds), weight pairs made with two `zip`s;
//! - dotprod: `sdot` by element (16 multiply-adds, one activation quad broadcast from a register);
//! - NEON (ARMv8.0): `smull` + `sadalp` (8 multiply-adds per instruction pair).
//!
//! All three add the same integer group sums with the same f32 arithmetic, so they give the same results.
//! `TINYASR_ISA=i8mm|dotprod|neon` caps the level (never above what the CPU reports); `TINYASR_ISA_LOG=1` prints it.
//! The f32 element-wise kernels below repeat the scalar arithmetic of `kernels` (no fused multiply-add) 4 lanes at a
//! time, so their results equal the scalar ones; only the sums inside `softmax_row` and `logsumexp` change order.
#![allow(unsafe_code)]

use alloc::vec;
use alloc::vec::Vec;
use core::arch::aarch64::*;

#[cfg(not(feature = "std"))]
use crate::compat::F32Ext;
use crate::kernels::exp;

/// The int8 instruction level in use.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Isa {
    Neon,
    Dotprod,
    I8mm,
}

/// The level the CPU has, capped by `TINYASR_ISA` (std builds), detected once.
pub fn isa() -> Isa {
    #[cfg(feature = "std")]
    {
        static ON: std::sync::OnceLock<Isa> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            let hw = if !std::arch::is_aarch64_feature_detected!("dotprod") {
                Isa::Neon
            } else if std::arch::is_aarch64_feature_detected!("i8mm") {
                Isa::I8mm
            } else {
                Isa::Dotprod
            };
            let want = match std::env::var("TINYASR_ISA").as_deref() {
                Ok("neon") => Isa::Neon,
                Ok("dotprod") => Isa::Dotprod,
                _ => Isa::I8mm,
            };
            let isa = hw.min(want);
            if std::env::var_os("TINYASR_ISA_LOG").is_some() {
                std::eprintln!("tinyasr aarch64 int8: {isa:?} (cpu: {hw:?})");
            }
            isa
        })
    }
    #[cfg(all(not(feature = "std"), any(target_os = "linux", target_os = "android")))]
    {
        // what the kernel reports (HWCAP_ASIMDDP, HWCAP2_I8MM), read once: 0 not known yet, else Isa + 1
        use core::sync::atomic::{AtomicU8, Ordering};
        unsafe extern "C" {
            fn getauxval(t: core::ffi::c_ulong) -> core::ffi::c_ulong;
        }
        static ON: AtomicU8 = AtomicU8::new(0);
        let mut on = ON.load(Ordering::Relaxed);
        if on == 0 {
            // SAFETY: getauxval only reads the auxiliary vector.
            let (hw, hw2) = unsafe { (getauxval(16), getauxval(26)) };
            let dp = cfg!(target_feature = "dotprod") || hw & (1 << 20) != 0;
            let mm = cfg!(target_feature = "i8mm") || hw2 & (1 << 13) != 0;
            on = if dp && mm { 3 } else if dp { 2 } else { 1 };
            ON.store(on, Ordering::Relaxed);
        }
        [Isa::Neon, Isa::Dotprod, Isa::I8mm][on as usize - 1]
    }
    #[cfg(all(not(feature = "std"), not(any(target_os = "linux", target_os = "android"))))]
    {
        if cfg!(all(target_feature = "dotprod", target_feature = "i8mm")) {
            Isa::I8mm
        } else if cfg!(target_feature = "dotprod") {
            Isa::Dotprod
        } else {
            Isa::Neon
        }
    }
}

/// Integer weights in panels of 16 output columns: per panel, per 4 inputs, the 4 bytes of each of the 16 columns
/// (zero columns pad the last panel); scales per panel, group and column.
pub struct Pack {
    w: Vec<i8>,   // panels x k/4 x 16 x 4
    sw: Vec<f32>, // panels x ng x 16
    n: usize,
    k: usize,
    g: usize,
}

impl Pack {
    /// Whether weights of row length k with groups of g can be packed (16-byte steps never cross a group).
    pub fn usable(k: usize, g: usize) -> bool {
        g > 0 && g % 16 == 0 && k % g == 0
    }

    pub fn new(n: usize, k: usize, g: usize, scale: &[f32], q: &[i8]) -> Pack {
        assert!(Pack::usable(k, g) && q.len() == n * k && scale.len() == n * (k / g));
        let (np, ng) = (n.div_ceil(16), k / g);
        let mut w = vec![0i8; np * k * 16];
        let mut sw = vec![0f32; np * ng * 16];
        for j in 0..n {
            let (p, l) = (j / 16, j % 16);
            for kk in 0..k {
                w[p * k * 16 + (kk / 4) * 64 + l * 4 + kk % 4] = q[j * k + kk];
            }
            for c in 0..ng {
                sw[(p * ng + c) * 16 + l] = scale[j * ng + c];
            }
        }
        Pack { w, sw, n, k, g }
    }
}

/// Activation rows i0..m (int8 `aq`, m x k, scales `sa`, m x ng) times weight columns j0..n_end into `out`
/// (row stride `stride`, column j at j - col0), as `Quant::block_i8`.
#[allow(clippy::too_many_arguments)]
pub fn block(pk: &Pack, aq: &[i8], sa: &[f32], i0: usize, m: usize, j0: usize, n_end: usize, out: &mut [f32], stride: usize, col0: usize) {
    if i0 >= m || j0 >= n_end {
        return;
    }
    let (k, g) = (pk.k, pk.g);
    let ng = k / g;
    assert!(n_end <= pk.n && j0 >= col0 && aq.len() >= m * k && sa.len() >= m * ng);
    assert!(out.len() >= (m - 1) * stride + n_end - col0);
    let isa = isa();
    let rmax = if isa == Isa::Neon { 2 } else { 4 };
    for p in j0 / 16..n_end.div_ceil(16) {
        let (c0, c1) = (j0.max(16 * p), n_end.min(16 * p + 16));
        let (w, sw) = (&pk.w[p * k * 16..(p + 1) * k * 16], &pk.sw[p * ng * 16..(p + 1) * ng * 16]);
        let (wp, swp) = (w.as_ptr(), sw.as_ptr());
        let mut i = i0;
        while i < m {
            let r = (m - i).min(rmax);
            // rows past the last one repeat it (computed, never stored)
            let ar = |t: usize| aq[(i + t.min(r - 1)) * k..(i + t.min(r - 1) + 1) * k].as_ptr();
            let sr = |t: usize| sa[(i + t.min(r - 1)) * ng..(i + t.min(r - 1) + 1) * ng].as_ptr();
            let mut res = [[0f32; 16]; 4];
            // SAFETY: each row pointer has k activations and ng scales, the panel k x 16 weights and ng x 16 scales
            // (sliced above); g % 16 == 0 and k % g == 0 (`Pack::usable`), so every 16-byte step stays inside them.
            // The instruction level was checked by `isa()`.
            unsafe {
                match (isa, r) {
                    (Isa::I8mm, 3..) => tile_smmla::<2>([[ar(0), ar(1)], [ar(2), ar(3)]], [[sr(0), sr(1)], [sr(2), sr(3)]], wp, swp, k, g, &mut res),
                    (Isa::I8mm, 2) => tile_smmla::<1>([[ar(0), ar(1)]], [[sr(0), sr(1)]], wp, swp, k, g, &mut res),
                    (Isa::I8mm | Isa::Dotprod, ..=1) => tile_sdot::<1>([ar(0)], [sr(0)], wp, swp, k, g, &mut res),
                    (Isa::Dotprod, 2) => tile_sdot::<2>([ar(0), ar(1)], [sr(0), sr(1)], wp, swp, k, g, &mut res),
                    (Isa::Dotprod, 3) => tile_sdot::<3>([ar(0), ar(1), ar(2)], [sr(0), sr(1), sr(2)], wp, swp, k, g, &mut res),
                    (Isa::Dotprod, _) => tile_sdot::<4>([ar(0), ar(1), ar(2), ar(3)], [sr(0), sr(1), sr(2), sr(3)], wp, swp, k, g, &mut res),
                    (Isa::Neon, ..=1) => tile_neon::<1>([ar(0)], [sr(0)], wp, swp, k, g, &mut res),
                    (Isa::Neon, _) => tile_neon::<2>([ar(0), ar(1)], [sr(0), sr(1)], wp, swp, k, g, &mut res),
                }
            }
            for (t, res) in res.iter().enumerate().take(r) {
                let o = (i + t) * stride;
                out[o + c0 - col0..o + c1 - col0].copy_from_slice(&res[c0 - 16 * p..c1 - 16 * p]);
            }
            i += r;
        }
    }
}

/// res[r][4t..4t+4] += float(acc[t]) x (sw[4t..4t+4] x sa_r): the group step every int8 tile ends with.
#[inline(always)]
unsafe fn add_group(res: &mut [f32; 16], acc: &[int32x4_t; 4], sw: *const f32, sa: f32) {
    // SAFETY: the caller guarantees 16 scales at sw.
    unsafe {
        for (t, acc) in acc.iter().enumerate() {
            let o = res.as_mut_ptr().add(4 * t);
            vst1q_f32(o, vfmaq_f32(vld1q_f32(o), vcvtq_f32_s32(*acc), vmulq_n_f32(vld1q_f32(sw.add(4 * t)), sa)));
        }
    }
}

/// acc[t] += 4-byte dot products of the 4 columns in w[t] with activation quad L of `a`.
#[inline(always)]
#[allow(unused_unsafe)]
unsafe fn sdot_lane<const L: u32>(acc: &mut [int32x4_t; 4], w: &[int8x16_t; 4], a: int8x16_t) {
    for t in 0..4 {
        // SAFETY: register-only; the caller runs with the dot-product extension. (vdotq_laneq_s32 is unstable.)
        unsafe {
            match L {
                0 => core::arch::asm!("sdot {c:v}.4s, {w:v}.16b, {a:v}.4b[0]", c = inout(vreg) acc[t], w = in(vreg) w[t], a = in(vreg) a, options(pure, nomem, nostack)),
                1 => core::arch::asm!("sdot {c:v}.4s, {w:v}.16b, {a:v}.4b[1]", c = inout(vreg) acc[t], w = in(vreg) w[t], a = in(vreg) a, options(pure, nomem, nostack)),
                2 => core::arch::asm!("sdot {c:v}.4s, {w:v}.16b, {a:v}.4b[2]", c = inout(vreg) acc[t], w = in(vreg) w[t], a = in(vreg) a, options(pure, nomem, nostack)),
                _ => core::arch::asm!("sdot {c:v}.4s, {w:v}.16b, {a:v}.4b[3]", c = inout(vreg) acc[t], w = in(vreg) w[t], a = in(vreg) a, options(pure, nomem, nostack)),
            }
        }
    }
}

/// R rows x 16 columns with `sdot` by element: per 16 inputs one load per row, 16 weight loads, 16 R `sdot`.
#[target_feature(enable = "dotprod")]
unsafe fn tile_sdot<const R: usize>(a: [*const i8; R], sa: [*const f32; R], w: *const i8, sw: *const f32, k: usize, g: usize, res: &mut [[f32; 16]; 4]) {
    let ng = k / g;
    // SAFETY: the caller guarantees k activations and ng scales per row, k x 16 weights and ng x 16 scales.
    unsafe {
        for c in 0..ng {
            let mut acc = [[vdupq_n_s32(0); 4]; R];
            let mut p = c * g;
            while p < (c + 1) * g {
                let av: [int8x16_t; R] = core::array::from_fn(|r| vld1q_s8(a[r].add(p)));
                let wp = w.add(p * 16);
                let wv = |q: usize| -> [int8x16_t; 4] { core::array::from_fn(|t| vld1q_s8(wp.add(64 * q + 16 * t))) };
                let (w0, w1, w2, w3) = (wv(0), wv(1), wv(2), wv(3));
                for r in 0..R {
                    sdot_lane::<0>(&mut acc[r], &w0, av[r]);
                    sdot_lane::<1>(&mut acc[r], &w1, av[r]);
                    sdot_lane::<2>(&mut acc[r], &w2, av[r]);
                    sdot_lane::<3>(&mut acc[r], &w3, av[r]);
                }
                p += 16;
            }
            for r in 0..R {
                add_group(&mut res[r], &acc[r], sw.add(c * 16), *sa[r].add(c));
            }
        }
    }
}

/// P row pairs x 16 columns with `smmla`: per 8 inputs, 8 weight column pairs (zip of two 4-input blocks) times
/// each row pair; lanes of one accumulator: (r0 c0, r0 c1, r1 c0, r1 c1).
#[target_feature(enable = "i8mm")]
unsafe fn tile_smmla<const P: usize>(a: [[*const i8; 2]; P], sa: [[*const f32; 2]; P], w: *const i8, sw: *const f32, k: usize, g: usize, res: &mut [[f32; 16]; 4]) {
    let ng = k / g;
    // SAFETY: the caller guarantees k activations and ng scales per row, k x 16 weights and ng x 16 scales.
    unsafe {
        for c in 0..ng {
            let mut acc = [[vdupq_n_s32(0); 8]; P];
            let mut p = c * g;
            while p < (c + 1) * g {
                // [r0 inputs 0..8, r1 inputs 0..8] and [r0 inputs 8..16, r1 inputs 8..16]
                let ap: [[int8x16_t; 2]; P] = core::array::from_fn(|i| {
                    let (x0, x1) = (vreinterpretq_s64_s8(vld1q_s8(a[i][0].add(p))), vreinterpretq_s64_s8(vld1q_s8(a[i][1].add(p))));
                    [vreinterpretq_s8_s64(vzip1q_s64(x0, x1)), vreinterpretq_s8_s64(vzip2q_s64(x0, x1))]
                });
                for h in 0..2 {
                    let wp = w.add((p + 8 * h) * 16);
                    for t in 0..4 {
                        let (lo, hi) = (vreinterpretq_s32_s8(vld1q_s8(wp.add(16 * t))), vreinterpretq_s32_s8(vld1q_s8(wp.add(64 + 16 * t))));
                        let b = [vreinterpretq_s8_s32(vzip1q_s32(lo, hi)), vreinterpretq_s8_s32(vzip2q_s32(lo, hi))];
                        for i in 0..P {
                            for (u, b) in b.iter().enumerate() {
                                // (vmmlaq_s32 is unstable)
                                core::arch::asm!("smmla {c:v}.4s, {a:v}.16b, {b:v}.16b", c = inout(vreg) acc[i][2 * t + u], a = in(vreg) ap[i][h], b = in(vreg) *b, options(pure, nomem, nostack));
                            }
                        }
                    }
                }
                p += 16;
            }
            for i in 0..P {
                let sav = vcombine_f32(vdup_n_f32(*sa[i][0].add(c)), vdup_n_f32(*sa[i][1].add(c)));
                for (cp, acc) in acc[i].iter().enumerate() {
                    let s = vld1_f32(sw.add(c * 16 + 2 * cp));
                    let (o0, o1) = (res[2 * i].as_mut_ptr().add(2 * cp), res[2 * i + 1].as_mut_ptr().add(2 * cp));
                    let v = vfmaq_f32(vcombine_f32(vld1_f32(o0), vld1_f32(o1)), vcvtq_f32_s32(*acc), vmulq_f32(vcombine_f32(s, s), sav));
                    vst1_f32(o0, vget_low_f32(v));
                    vst1_f32(o1, vget_high_f32(v));
                }
            }
        }
    }
}

/// One activation quad (lane L of `a`) times the 16 columns: `smull` / `smull2` to 16 bits, `sadalp` into pairs.
#[inline(always)]
#[allow(unused_unsafe)]
unsafe fn neon_lane<const L: i32>(acc: &mut [int32x4_t; 8], w: &[int8x16_t; 4], a: int32x4_t) {
    // SAFETY: register-only NEON (part of aarch64). |a| <= 127 and |w| <= 128, so each product fits 16 bits.
    unsafe {
        let a4 = vreinterpretq_s8_s32(vdupq_laneq_s32::<L>(a));
        for t in 0..4 {
            acc[2 * t] = vpadalq_s16(acc[2 * t], vmull_s8(vget_low_s8(w[t]), vget_low_s8(a4)));
            acc[2 * t + 1] = vpadalq_s16(acc[2 * t + 1], vmull_high_s8(w[t], a4));
        }
    }
}

/// R rows x 16 columns in plain NEON (ARMv8.0): accumulators hold pair sums (c0 01, c0 23, c1 01, c1 23).
unsafe fn tile_neon<const R: usize>(a: [*const i8; R], sa: [*const f32; R], w: *const i8, sw: *const f32, k: usize, g: usize, res: &mut [[f32; 16]; 4]) {
    let ng = k / g;
    // SAFETY: the caller guarantees k activations and ng scales per row, k x 16 weights and ng x 16 scales.
    unsafe {
        for c in 0..ng {
            let mut acc = [[vdupq_n_s32(0); 8]; R];
            let mut p = c * g;
            while p < (c + 1) * g {
                let av: [int32x4_t; R] = core::array::from_fn(|r| vreinterpretq_s32_s8(vld1q_s8(a[r].add(p))));
                let wp = w.add(p * 16);
                let wv = |q: usize| -> [int8x16_t; 4] { core::array::from_fn(|t| vld1q_s8(wp.add(64 * q + 16 * t))) };
                let (w0, w1, w2, w3) = (wv(0), wv(1), wv(2), wv(3));
                for r in 0..R {
                    neon_lane::<0>(&mut acc[r], &w0, av[r]);
                    neon_lane::<1>(&mut acc[r], &w1, av[r]);
                    neon_lane::<2>(&mut acc[r], &w2, av[r]);
                    neon_lane::<3>(&mut acc[r], &w3, av[r]);
                }
                p += 16;
            }
            for r in 0..R {
                let sums: [int32x4_t; 4] = core::array::from_fn(|t| vpaddq_s32(acc[r][2 * t], acc[r][2 * t + 1]));
                add_group(&mut res[r], &sums, sw.add(c * 16), *sa[r].add(c));
            }
        }
    }
}

/// f32 micro-kernel as `kernels::kernel` (same fused multiply-adds in the same order, so the same results), with the
/// activations loaded 4 at a time and used by lane (`fmla v.4s, v.4s, v.s[l]`) instead of one broadcast per value.
#[inline(always)]
pub fn kernel_f32<const R: usize>(a: &[f32], k: usize, panel: &[f32], init: &[f32; 16], out: &mut [[f32; 16]; R]) {
    assert!(a.len() >= R * k && panel.len() >= k * 16);
    #[inline(always)]
    #[allow(unused_unsafe)]
    unsafe fn step<const R: usize, const L: i32>(acc: &mut [[float32x4_t; 4]; R], av: &[float32x4_t; R], b: *const f32) {
        // SAFETY: the caller guarantees 16 values at b.
        unsafe {
            let bv = [vld1q_f32(b), vld1q_f32(b.add(4)), vld1q_f32(b.add(8)), vld1q_f32(b.add(12))];
            for r in 0..R {
                for q in 0..4 {
                    acc[r][q] = vfmaq_laneq_f32::<L>(acc[r][q], bv[q], av[r]);
                }
            }
        }
    }
    // SAFETY: NEON is part of aarch64; row r reads a[r k + p], p < k; the panel reads panel[16 p .. 16 p + 16], p < k
    // (both asserted above).
    unsafe {
        let ip = init.as_ptr();
        let start = [vld1q_f32(ip), vld1q_f32(ip.add(4)), vld1q_f32(ip.add(8)), vld1q_f32(ip.add(12))];
        let mut acc = [start; R];
        let (ap, bp) = (a.as_ptr(), panel.as_ptr());
        let mut p = 0;
        while p + 4 <= k {
            let av: [float32x4_t; R] = core::array::from_fn(|r| vld1q_f32(ap.add(r * k + p)));
            step::<R, 0>(&mut acc, &av, bp.add(p * 16));
            step::<R, 1>(&mut acc, &av, bp.add((p + 1) * 16));
            step::<R, 2>(&mut acc, &av, bp.add((p + 2) * 16));
            step::<R, 3>(&mut acc, &av, bp.add((p + 3) * 16));
            p += 4;
        }
        while p < k {
            let av: [float32x4_t; R] = core::array::from_fn(|r| vdupq_n_f32(*ap.add(r * k + p)));
            step::<R, 0>(&mut acc, &av, bp.add(p * 16));
            p += 1;
        }
        for (o, acc) in out.iter_mut().zip(&acc) {
            for (q, v) in acc.iter().enumerate() {
                vst1q_f32(o.as_mut_ptr().add(4 * q), *v);
            }
        }
    }
}

/// `kernels::exp`, 4 lanes, the same operations.
#[inline(always)]
#[allow(unused_unsafe, clippy::approx_constant)]
fn exp4(x: float32x4_t) -> float32x4_t {
    // SAFETY: register-only NEON (part of aarch64).
    unsafe {
        let t = vminq_f32(vmaxq_f32(vmulq_f32(x, vdupq_n_f32(core::f32::consts::LOG2_E)), vdupq_n_f32(-126.0)), vdupq_n_f32(126.0));
        let fi = vrndmq_f32(t);
        let f = vsubq_f32(t, fi);
        let mut q = vmulq_f32(f, vdupq_n_f32(0.001_333_355));
        q = vmulq_f32(f, vaddq_f32(vdupq_n_f32(0.009_618_129), q));
        q = vmulq_f32(f, vaddq_f32(vdupq_n_f32(0.055_504_11), q));
        q = vmulq_f32(f, vaddq_f32(vdupq_n_f32(0.240_226_5), q));
        q = vmulq_f32(f, vaddq_f32(vdupq_n_f32(0.693_147_2), q));
        q = vaddq_f32(vdupq_n_f32(1.0), q);
        let pw = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(fi), vdupq_n_s32(127))));
        vmulq_f32(pw, q)
    }
}

/// `kernels::ln`, 4 lanes, the same operations.
#[inline(always)]
#[allow(unused_unsafe)]
fn ln4(x: float32x4_t) -> float32x4_t {
    // SAFETY: register-only NEON (part of aarch64).
    unsafe {
        let one = vdupq_n_f32(1.0);
        let b = vreinterpretq_s32_f32(x);
        let e = vshrq_n_s32::<23>(vsubq_s32(b, vdupq_n_s32(0x3f35_04f3)));
        let m = vreinterpretq_f32_s32(vsubq_s32(b, vshlq_n_s32::<23>(e)));
        let s = vdivq_f32(vsubq_f32(m, one), vaddq_f32(m, one));
        let z = vmulq_f32(s, s);
        let mut r = vmulq_f32(z, vdupq_n_f32(0.181_818_2));
        r = vmulq_f32(z, vaddq_f32(vdupq_n_f32(0.222_222_2), r));
        r = vmulq_f32(z, vaddq_f32(vdupq_n_f32(0.285_714_3), r));
        r = vmulq_f32(z, vaddq_f32(vdupq_n_f32(0.4), r));
        r = vmulq_f32(z, vaddq_f32(vdupq_n_f32(0.666_666_7), r));
        r = vaddq_f32(vdupq_n_f32(2.0), r);
        vaddq_f32(vmulq_f32(vcvtq_f32_s32(e), vdupq_n_f32(core::f32::consts::LN_2)), vmulq_f32(s, r))
    }
}

/// `kernels::sigmoid`, 4 lanes: 1 / (1 + exp(-x)).
#[inline(always)]
#[allow(unused_unsafe)]
fn sigmoid4(x: float32x4_t) -> float32x4_t {
    // SAFETY: register-only NEON (part of aarch64).
    unsafe { vdivq_f32(vdupq_n_f32(1.0), vaddq_f32(vdupq_n_f32(1.0), exp4(vnegq_f32(x)))) }
}

/// `kernels::tanh`, 4 lanes: 1 - 2 / (exp(2x) + 1).
#[inline(always)]
#[allow(unused_unsafe)]
fn tanh4(x: float32x4_t) -> float32x4_t {
    // SAFETY: register-only NEON (part of aarch64).
    unsafe {
        let two = vdupq_n_f32(2.0);
        vsubq_f32(vdupq_n_f32(1.0), vdivq_f32(two, vaddq_f32(exp4(vmulq_f32(two, x)), vdupq_n_f32(1.0))))
    }
}

/// `kernels::swoosh` on the first len/4*4 values; returns how many it did.
pub fn swoosh(x: &mut [f32], shift: f32, offset: f32) -> usize {
    let n4 = x.len() / 4 * 4;
    let p = x.as_mut_ptr();
    // SAFETY: NEON is part of aarch64; loads and stores at i < n4 <= x.len().
    unsafe {
        let (sh, of, k08, zero, one) = (vdupq_n_f32(shift), vdupq_n_f32(offset), vdupq_n_f32(0.08), vdupq_n_f32(0.0), vdupq_n_f32(1.0));
        let mut i = 0;
        while i < n4 {
            let v = vld1q_f32(p.add(i));
            let y = vsubq_f32(v, sh);
            let sp = vaddq_f32(vmaxq_f32(y, zero), ln4(vaddq_f32(one, exp4(vnegq_f32(vabsq_f32(y))))));
            vst1q_f32(p.add(i), vsubq_f32(vsubq_f32(sp, vmulq_f32(k08, v)), of));
            i += 4;
        }
    }
    n4
}

/// out[i] = f(a[i], b[i]) for i < n/4*4 (n = the shortest length); returns how many it did.
#[inline(always)]
fn map2(a: &[f32], b: &[f32], out: &mut [f32], f: impl Fn(float32x4_t, float32x4_t) -> float32x4_t) -> usize {
    let n4 = a.len().min(b.len()).min(out.len()) / 4 * 4;
    // SAFETY: NEON is part of aarch64; accesses at i < n4 <= every length.
    unsafe {
        let mut i = 0;
        while i < n4 {
            vst1q_f32(out.as_mut_ptr().add(i), f(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i))));
            i += 4;
        }
    }
    n4
}

/// `kernels::tanh_sum` (tanh(a + b)) on the first n/4*4 values; returns how many it did.
#[allow(unused_unsafe)]
pub fn tanh_sum(a: &[f32], b: &[f32], out: &mut [f32]) -> usize {
    // SAFETY: register-only NEON.
    map2(a, b, out, |a, b| tanh4(unsafe { vaddq_f32(a, b) }))
}

/// `kernels::glu` (a sigmoid(b)) on the first n/4*4 values; returns how many it did.
#[allow(unused_unsafe)]
pub fn glu(a: &[f32], b: &[f32], out: &mut [f32]) -> usize {
    // SAFETY: register-only NEON.
    map2(a, b, out, |a, b| unsafe { vmulq_f32(a, sigmoid4(b)) })
}

/// `kernels::lstm_cell` for j < n/4*4 (n = c.len(); g >= 4 n, h >= n); returns how many it did.
pub fn lstm_cell(g: &[f32], c: &mut [f32], h: &mut [f32]) -> usize {
    let n = c.len();
    assert!(g.len() >= 4 * n && h.len() >= n);
    let n4 = n / 4 * 4;
    // SAFETY: NEON is part of aarch64; accesses at j < n4 <= n, g at up to 3 n + j (asserted above).
    unsafe {
        let (gp, cp, hp) = (g.as_ptr(), c.as_mut_ptr(), h.as_mut_ptr());
        let mut j = 0;
        while j < n4 {
            let (i, f) = (sigmoid4(vld1q_f32(gp.add(j))), sigmoid4(vld1q_f32(gp.add(n + j))));
            let (gg, o) = (tanh4(vld1q_f32(gp.add(2 * n + j))), sigmoid4(vld1q_f32(gp.add(3 * n + j))));
            let cv = vaddq_f32(vmulq_f32(f, vld1q_f32(cp.add(j))), vmulq_f32(i, gg));
            vst1q_f32(cp.add(j), cv);
            vst1q_f32(hp.add(j), vmulq_f32(o, tanh4(cv)));
            j += 4;
        }
    }
    n4
}

/// `kernels::softmax_row` (the whole row; the sum in another order).
pub fn softmax_row(row: &mut [f32]) {
    let n = row.len();
    let n4 = n / 4 * 4;
    // SAFETY: NEON is part of aarch64; vector accesses at i < n4 <= n, the tail by index below n.
    unsafe {
        let p = row.as_mut_ptr();
        let mut mx = vdupq_n_f32(f32::MIN);
        for i in (0..n4).step_by(4) {
            mx = vmaxq_f32(mx, vld1q_f32(p.add(i)));
        }
        let mut m = vmaxvq_f32(mx);
        for j in n4..n {
            m = m.max(*p.add(j));
        }
        let (mv, mut s) = (vdupq_n_f32(m), vdupq_n_f32(0.0));
        for i in (0..n4).step_by(4) {
            let e = exp4(vsubq_f32(vld1q_f32(p.add(i)), mv));
            vst1q_f32(p.add(i), e);
            s = vaddq_f32(s, e);
        }
        let mut sum = vaddvq_f32(s);
        for j in n4..n {
            let e = exp(*p.add(j) - m);
            *p.add(j) = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for i in (0..n4).step_by(4) {
            vst1q_f32(p.add(i), vmulq_n_f32(vld1q_f32(p.add(i)), inv));
        }
        for j in n4..n {
            *p.add(j) *= inv;
        }
    }
}

/// `kernels::logsumexp` (the sum in another order).
pub fn logsumexp(v: &[f32]) -> f32 {
    let n4 = v.len() / 4 * 4;
    // SAFETY: NEON is part of aarch64; loads at i < n4 <= v.len().
    unsafe {
        let p = v.as_ptr();
        let mut mx = vdupq_n_f32(f32::MIN);
        for i in (0..n4).step_by(4) {
            mx = vmaxq_f32(mx, vld1q_f32(p.add(i)));
        }
        let m = v[n4..].iter().fold(vmaxvq_f32(mx), |m, &x| m.max(x));
        let (mv, mut s) = (vdupq_n_f32(m), vdupq_n_f32(0.0));
        for i in (0..n4).step_by(4) {
            s = vaddq_f32(s, exp4(vsubq_f32(vld1q_f32(p.add(i)), mv)));
        }
        let sum = v[n4..].iter().fold(vaddvq_f32(s), |s, &x| s + exp(x - m));
        m + sum.ln()
    }
}

/// `kernels::dw_taps` when d = out.len() is a multiple of 4 (same multiplies and adds in the same order, accumulators
/// in registers); false (nothing done) otherwise.
pub fn dw_taps(out: &mut [f32], init: Option<&[f32]>, w: &[f32], x: &[f32], taps: usize) -> bool {
    let d = out.len();
    if d % 4 != 0 {
        return false;
    }
    assert!(w.len() >= taps * d && x.len() >= taps * d && init.is_none_or(|v| v.len() >= d));
    // SAFETY: NEON is part of aarch64; every access is below taps d (w, x) or d (out, init), checked above.
    unsafe {
        let (op, wp, xp) = (out.as_mut_ptr(), w.as_ptr(), x.as_ptr());
        let sp = init.map_or(op as *const f32, |v| v.as_ptr());
        let mut c = 0;
        while c + 16 <= d {
            let mut acc: [float32x4_t; 4] = core::array::from_fn(|q| vld1q_f32(sp.add(c + 4 * q)));
            for u in 0..taps {
                let (wr, xr) = (wp.add(u * d + c), xp.add(u * d + c));
                for (q, acc) in acc.iter_mut().enumerate() {
                    *acc = vaddq_f32(*acc, vmulq_f32(vld1q_f32(wr.add(4 * q)), vld1q_f32(xr.add(4 * q))));
                }
            }
            for (q, acc) in acc.iter().enumerate() {
                vst1q_f32(op.add(c + 4 * q), *acc);
            }
            c += 16;
        }
        while c < d {
            let mut acc = vld1q_f32(sp.add(c));
            for u in 0..taps {
                acc = vaddq_f32(acc, vmulq_f32(vld1q_f32(wp.add(u * d + c)), vld1q_f32(xp.add(u * d + c))));
            }
            vst1q_f32(op.add(c), acc);
            c += 4;
        }
    }
    true
}
