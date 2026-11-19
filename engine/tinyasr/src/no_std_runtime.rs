//! The lang items a `no_std` static/shared library has to carry.
//!
//! A Rust library normally leaves both to the final binary, but a C or C++ host cannot define Rust
//! lang items, so the build without `std` provides them itself. `std` builds don't have
//! this module. The allocator forwards to the C runtime the host already links; panics abort.

#![allow(unsafe_code)]

use core::alloc::{GlobalAlloc, Layout};

unsafe extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
    fn abort() -> !;
}

/// Forwards to the host C allocator. Only pointer alignment is assumed from `malloc`; stricter
/// alignments are over-allocated and aligned by hand, with the original
/// pointer stored in the word just below the returned address.
struct HostAlloc;

const MALLOC_ALIGN: usize = core::mem::align_of::<usize>();

// SAFETY: `alloc` returns null or a block of at least `layout.size()` bytes aligned to `layout.align()`, and
// `dealloc` frees exactly the `malloc` block `alloc` got for the same layout.
unsafe impl GlobalAlloc for HostAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.align() <= MALLOC_ALIGN {
            // SAFETY: C malloc takes any size; its blocks are aligned for any scalar, so for `MALLOC_ALIGN`.
            return unsafe { malloc(layout.size()) };
        }
        let Some(padded) = layout.size().checked_add(layout.align() + core::mem::size_of::<usize>()) else {
            return core::ptr::null_mut();
        };
        // SAFETY: as above; the size has room for the stored pointer and the alignment gap.
        let raw = unsafe { malloc(padded) };
        if raw.is_null() {
            return raw;
        }
        // SAFETY: one word into a block of `padded` > one word bytes stays inside it.
        let reserved = unsafe { raw.add(core::mem::size_of::<*mut u8>()) };
        // SAFETY: the offset is below `layout.align()`, and `padded` leaves `layout.align()` bytes after `reserved`
        // besides `layout.size()`, so `aligned` and the `size` bytes after it stay inside the block.
        let aligned = unsafe { reserved.add(reserved.align_offset(layout.align())) };
        // SAFETY: `aligned` is at least one word above `raw` and inside the padded block; it is aligned to more than
        // a word, so the word below it is aligned for a pointer.
        unsafe { (aligned as *mut *mut u8).sub(1).write(raw) };
        aligned
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.align() <= MALLOC_ALIGN {
            // SAFETY: `ptr` came from `malloc` in `alloc` for this same layout and is freed once.
            unsafe { free(ptr) };
        } else {
            // SAFETY: alloc stored the original `malloc` pointer just below the address it returned.
            unsafe { free((ptr as *mut *mut u8).sub(1).read()) };
        }
    }
}

#[global_allocator]
static ALLOCATOR: HostAlloc = HostAlloc;

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // No message: formatting would pull in core::fmt, which a no_std build tries to leave out.
    // SAFETY: C abort takes no arguments and never returns.
    unsafe { abort() }
}

/// Builds abort on panic, but Android's precompiled `core`/`alloc` still reference this symbol.
#[unsafe(no_mangle)]
pub extern "C" fn rust_eh_personality() {}
