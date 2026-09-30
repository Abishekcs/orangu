// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Keeping a tree's cores at full speed while a request is running.
//!
//! A node of a tree works in bursts: its layers of a token, then nothing
//! while the nodes below run theirs. The kernel's frequency governor
//! (`schedutil`) sees a core busy half the time and clocks it down — on
//! the CIX P1 from 2.5 GHz to 1.4–1.7 GHz — and every layer ran about 60%
//! slower than on a server busy all the time: a two-node tree decoded at
//! 160 ms a token where one server took 99. With the cores held at their
//! clock, the same tree took 103.
//!
//! What holds them is one spinning thread per core this process may run
//! on, at `SCHED_IDLE` — below every other task, so it yields to any real
//! work at once, and to other processes too — which keeps each core looking
//! busy to the governor. They spin only while a forward has run in the
//! last [`LINGER`], and sleep otherwise: an idle node clocks down as ever.
//! The cost is power while requests run, what a busy single server draws.
//!
//! Linux's utilization clamp (`uclamp.min`) looked like the tidier tool and
//! was measured first: on big.LITTLE it also steers the energy-aware
//! scheduler, which packed the threads onto the fastest cores, and the tree
//! got slower, not faster.
//!
//! **Opt-in, `ORANGU_WORKERS_CLOCK=1`.** It helps a node whose cores are its
//! own — one node to a machine, or nodes held to separate cores — and it
//! hurts nodes that share cores: another node's spinners then sit on the
//! cores this node computes on. Three nodes of Llama-3.2-3B on one board,
//! every node using every core (the default), decoded at 118 ms a token
//! without it and 237 with it; the same three held to four threads each
//! went from 172 to 130 with it. A node cannot tell which case it is in.
//! On a machine that is a tree's alone, the `performance` CPU governor does
//! the same job without spinning, where root is at hand.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// How long the cores are held after the last forward: longer than the gap
/// between one token's forwards at a node, shorter than anyone waits
/// between requests.
pub const LINGER: Duration = Duration::from_millis(300);

/// When the process started, as the zero `ACTIVE_UNTIL` counts from.
static EPOCH: OnceLock<Instant> = OnceLock::new();

/// Until when (milliseconds past `EPOCH`) the spinners hold the cores.
static ACTIVE_UNTIL: AtomicU64 = AtomicU64::new(0);

/// Whether the keeper is on: `ORANGU_WORKERS_CLOCK=1` turns it on.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::engine::env::flag_on("ORANGU_WORKERS_CLOCK"))
}

fn now_ms() -> u64 {
    EPOCH.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// Starts the spinners, idle until the first [`hold`]. Once per process;
/// a no-op when the keeper is off. Returns how many cores it keeps.
pub fn start() -> usize {
    static STARTED: OnceLock<usize> = OnceLock::new();
    *STARTED.get_or_init(|| {
        if !enabled() {
            return 0;
        }
        let _ = now_ms();
        let cores = allowed_cores();
        for &core in &cores {
            let _ = std::thread::Builder::new()
                .name(format!("orangu-clock-{core}"))
                .spawn(move || spin(core));
        }
        cores.len()
    })
}

/// Keeps the cores clocked for the next [`LINGER`]: called around every
/// forward a node runs.
pub fn hold() {
    if enabled() {
        let until = now_ms() + LINGER.as_millis() as u64;
        ACTIVE_UNTIL.fetch_max(until, Ordering::Relaxed);
    }
}

fn spin(core: usize) {
    lower_to_idle(core);
    loop {
        if now_ms() < ACTIVE_UNTIL.load(Ordering::Relaxed) {
            for _ in 0..10_000 {
                std::hint::spin_loop();
            }
        } else {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

/// Pins the calling thread to `core` and moves it to `SCHED_IDLE`.
#[cfg(target_os = "linux")]
fn lower_to_idle(core: usize) {
    // Safety: both calls change only the calling thread's own affinity and
    // scheduling class, with a zeroed and then filled `cpu_set_t` and a
    // zeroed `sched_param` — what `SCHED_IDLE` takes.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(core, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        let param: libc::sched_param = std::mem::zeroed();
        libc::sched_setscheduler(0, libc::SCHED_IDLE, &param);
    }
}

#[cfg(not(target_os = "linux"))]
fn lower_to_idle(_core: usize) {}

/// The cores this process may run on.
#[cfg(target_os = "linux")]
fn allowed_cores() -> Vec<usize> {
    // Safety: `sched_getaffinity` fills a zeroed `cpu_set_t` of the size
    // given; `CPU_ISSET` only reads it.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Vec::new();
        }
        (0..libc::CPU_SETSIZE as usize)
            .filter(|&core| libc::CPU_ISSET(core, &set))
            .collect()
    }
}

#[cfg(not(target_os = "linux"))]
fn allowed_cores() -> Vec<usize> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hold_lasts_for_the_linger() {
        let before = now_ms();
        hold();
        let until = ACTIVE_UNTIL.load(Ordering::Relaxed);
        if enabled() {
            assert!(until >= before + LINGER.as_millis() as u64);
        }
    }

    #[test]
    fn this_process_may_run_somewhere() {
        if cfg!(target_os = "linux") {
            assert!(!allowed_cores().is_empty());
        }
    }
}
