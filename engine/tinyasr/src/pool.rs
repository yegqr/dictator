//! A small persistent thread pool for data-parallel loops (std only, no dependencies).
//!
//! `par_for(n, f)` runs `f(0) .. f(n - 1)` on the calling thread and the pool's workers and returns when all calls
//! have finished. Workers are started on first use: `TINYASR_THREADS` of them in total (default: the number of logical
//! CPUs; 1 turns the pool off).
//!
//! Loops may nest and run from several threads at once: every running loop is a job on one shared list, the caller
//! works through its own job's tasks, and idle workers take tasks from the newest job first (the innermost, finest
//! loop), so threads left over by an outer loop with few tasks (e.g. two pieces of a recording on eight threads)
//! help inside the pieces.

use core::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

type Task = &'static (dyn Fn(usize) + Sync);

/// One running loop; lives on its caller's stack until all its tasks are done.
struct Job {
    f: Task,
    n: usize,
    next: AtomicUsize, // next task to hand out (claimed under the pool lock)
    done: AtomicUsize, // tasks finished
}

struct State {
    jobs: Vec<usize>, // &Job as addresses, oldest first
    sleepers: usize,
}

struct Pool {
    threads: usize,
    st: Mutex<State>,
    work: Condvar, // workers: a task is ready
    idle: Condvar, // callers: a job finished
}

thread_local! {
    static SERIAL: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Runs `f` with every `par_for` on this thread (also nested ones) serial: for tasks that are already one block of a
/// bigger parallel loop and should not pay the pool's synchronization again for each small step inside.
pub fn serial<R>(f: impl FnOnce() -> R) -> R {
    let was = SERIAL.with(|c| c.replace(true));
    let r = f();
    SERIAL.with(|c| c.set(was));
    r
}

fn pool() -> &'static Pool {
    static P: OnceLock<&'static Pool> = OnceLock::new();
    P.get_or_init(|| {
        let threads = std::env::var("TINYASR_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
            .clamp(1, 64);
        let p: &'static Pool = Box::leak(Box::new(Pool {
            threads,
            st: Mutex::new(State { jobs: Vec::new(), sleepers: 0 }),
            work: Condvar::new(),
            idle: Condvar::new(),
        }));
        for _ in 1..threads {
            // a host that cannot start more threads (process limit) still works: the caller runs every task no
            // worker takes, so fewer workers only cost speed
            if std::thread::Builder::new().name("tinyasr".into()).spawn(move || worker(p)).is_err() {
                break;
            }
        }
        p
    })
}

/// Total threads a `par_for` uses (the caller included).
pub fn threads() -> usize {
    pool().threads
}

#[allow(unsafe_code)]
fn job_ref(a: usize) -> &'static Job {
    // SAFETY: addresses on the list point at live `Job`s: a caller removes its job (under the lock) only after all of
    // its tasks are done, and tasks are only claimed under the same lock while the job is listed.
    unsafe { &*(a as *const Job) }
}

/// Claims a task under the lock: from `only` if given, else from the newest job that has one.
fn claim(st: &mut State, only: Option<usize>) -> Option<(&'static Job, usize)> {
    let pick = |a: usize| {
        let j = job_ref(a);
        let i = j.next.load(Ordering::Relaxed);
        (i < j.n).then(|| {
            j.next.store(i + 1, Ordering::Relaxed);
            (j, i)
        })
    };
    match only {
        Some(a) => pick(a),
        None => st.jobs.iter().rev().find_map(|&a| pick(a)),
    }
}

/// Aborts the process when dropped, i.e. when a panic unwinds through it (it is forgotten on the normal path). The
/// pool's soundness needs every claimed task to finish and every `par_for` to wait for its tasks, whatever the
/// panic strategy of the final binary (a dependency's `panic = "abort"` profile does not apply to it): a panicking
/// task must neither unwind out of `par_for` while workers still run the closure on its stack frame, nor end a worker
/// without marking its task done (the caller would wait forever).
struct AbortOnUnwind;
impl Drop for AbortOnUnwind {
    fn drop(&mut self) {
        std::process::abort();
    }
}

/// Runs one claimed task and marks it done; wakes the job's caller if it was the last one.
fn run(p: &Pool, j: &'static Job, i: usize) {
    let guard = AbortOnUnwind;
    (j.f)(i);
    core::mem::forget(guard);
    let n = j.n;
    // after this increment the job may be gone: touch nothing of it any more
    if j.done.fetch_add(1, Ordering::AcqRel) + 1 == n {
        let _g = p.st.lock().unwrap();
        p.idle.notify_all();
    }
}

fn worker(p: &'static Pool) {
    loop {
        let mut st = p.st.lock().unwrap();
        loop {
            if let Some((j, i)) = claim(&mut st, None) {
                drop(st);
                run(p, j, i);
                break;
            }
            st.sleepers += 1;
            st = p.work.wait(st).unwrap();
            st.sleepers -= 1;
        }
    }
}

/// Runs `f(i)` for every i in 0..n, in parallel; returns when all are done.
pub fn par_for(n: usize, f: &(dyn Fn(usize) + Sync)) {
    let p = pool();
    if n <= 1 || p.threads == 1 || SERIAL.with(|c| c.get()) {
        return (0..n).for_each(f);
    }
    // SAFETY: the reference is only reachable through the job list while this call runs: we remove the job (under the
    // lock) only after every task has finished, and tasks are claimed under that lock. A panic in between aborts
    // (`AbortOnUnwind`, independent of the profile's panic strategy), so no unwinding skips the wait.
    #[allow(unsafe_code)]
    let task: Task = unsafe { core::mem::transmute::<&(dyn Fn(usize) + Sync), Task>(f) };
    let job = Job { f: task, n, next: AtomicUsize::new(0), done: AtomicUsize::new(0) };
    let addr = &job as *const Job as usize;
    // from here until the job is off the list, a panic must not unwind past this frame (see `AbortOnUnwind`)
    let guard = AbortOnUnwind;
    {
        let mut st = p.st.lock().unwrap();
        st.jobs.push(addr);
        if st.sleepers > 0 {
            p.work.notify_all();
        }
    }
    // our own tasks first
    loop {
        let c = {
            let mut st = p.st.lock().unwrap();
            claim(&mut st, Some(addr))
        };
        match c {
            Some((j, i)) => run(p, j, i),
            None => break,
        }
    }
    // then wait for the ones the workers took
    let mut st = p.st.lock().unwrap();
    while job.done.load(Ordering::Acquire) < n {
        st = p.idle.wait(st).unwrap();
    }
    st.jobs.retain(|&a| a != addr);
    drop(st);
    core::mem::forget(guard);
}
