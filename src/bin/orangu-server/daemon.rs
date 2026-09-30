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

//! `--daemon`, in two steps around the start-up.
//!
//! A fork keeps only the thread that called it. The server used to fork
//! once the model was loaded and the listeners bound — and by then loading
//! had built thread pools (rayon's, the CPU backend's) and a `[workers]`
//! node had started its listener: the daemon woke up without any of them,
//! answered `/health` and waited forever on the first request that needed a
//! pool.
//!
//! So [`begin`] forks first, before any thread exists, and the original
//! process only waits: on a pipe, for the child to say it is ready. The
//! child loads and binds still attached to the terminal, so a bad config, an
//! unresolvable model or an address in use is printed there as before, and
//! then [`ready`] detaches it — a new session, `/`, `/dev/null` for stdio —
//! and tells the waiting process, which exits 0. A child that fails first
//! closes the pipe without a word, and the waiting process exits with the
//! child's own status.
//!
//! The pipe is inherited across `exec`, on purpose: a start that fails over
//! to a fallback model re-executes the child in the foreground, as an
//! attached start would, and the waiting process then waits on that.

use anyhow::Result;

#[cfg(unix)]
mod imp {
    use anyhow::{Context, Result, anyhow, bail};
    use std::sync::Mutex;

    /// The write end of the pipe, in the child between [`begin`] and
    /// [`ready`].
    static WRITER: Mutex<Option<i32>> = Mutex::new(None);

    const READY: &[u8] = b"ok";

    pub fn begin() -> Result<()> {
        let mut fds = [0i32; 2];
        // Safety: `pipe` fills the two descriptors of a two-element array.
        if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
            bail!("pipe: {}", std::io::Error::last_os_error());
        }
        let [reader, writer] = fds;
        // Safety: called before any thread exists (see `main`), so the
        // child is a whole copy of this process.
        match unsafe { libc::fork() } {
            -1 => bail!("fork: {}", std::io::Error::last_os_error()),
            0 => {
                // Safety: closing this process's own copy of the read end.
                unsafe { libc::close(reader) };
                *WRITER.lock().unwrap() = Some(writer);
                Ok(())
            }
            child => {
                // Safety: closing the write end, so the read below ends when
                // the child's copy closes.
                unsafe { libc::close(writer) };
                std::process::exit(wait_for(reader, child));
            }
        }
    }

    /// The waiting process: 0 when the child said it is ready, else the
    /// child's exit status.
    fn wait_for(reader: i32, child: libc::pid_t) -> i32 {
        let mut said = Vec::new();
        let mut buf = [0u8; 16];
        loop {
            // Safety: reads into a buffer of the length given.
            let n = unsafe { libc::read(reader, buf.as_mut_ptr().cast(), buf.len()) };
            if n > 0 {
                said.extend_from_slice(&buf[..n as usize]);
            } else if n == 0
                || std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted
            {
                break;
            }
        }
        if said == READY {
            return 0;
        }
        let mut status = 0;
        // Safety: waits for this process's own child.
        if unsafe { libc::waitpid(child, &mut status, 0) } == child && libc::WIFEXITED(status) {
            libc::WEXITSTATUS(status).max(1)
        } else {
            1
        }
    }

    pub fn ready() -> Result<()> {
        let Some(writer) = WRITER.lock().unwrap().take() else {
            return Err(anyhow!("--daemon: not started through daemon::begin"));
        };
        // Safety: each call changes only this process's own session,
        // directory, mask and descriptors; `/dev/null` is opened read-write
        // and duplicated onto 0, 1 and 2.
        unsafe {
            if libc::setsid() == -1 {
                bail!("setsid: {}", std::io::Error::last_os_error());
            }
            libc::umask(0o027);
            std::env::set_current_dir("/").context("chdir /")?;
            let null = libc::open(c"/dev/null".as_ptr(), libc::O_RDWR);
            if null < 0 {
                bail!("/dev/null: {}", std::io::Error::last_os_error());
            }
            for fd in 0..3 {
                libc::dup2(null, fd);
            }
            if null > 2 {
                libc::close(null);
            }
            libc::write(writer, READY.as_ptr().cast(), READY.len());
            libc::close(writer);
        }
        Ok(())
    }
}

/// Forks; only the child returns. The parent waits for [`ready`], or for
/// the child to fail, and exits. Must be called before any thread exists.
#[cfg(unix)]
pub fn begin() -> Result<()> {
    imp::begin()
}

/// Detaches the child from the terminal and releases the waiting parent.
#[cfg(unix)]
pub fn ready() -> Result<()> {
    imp::ready()
}

#[cfg(not(unix))]
pub fn begin() -> Result<()> {
    Err(anyhow::anyhow!(
        "--daemon is only supported on Unix-like platforms"
    ))
}

#[cfg(not(unix))]
pub fn ready() -> Result<()> {
    Ok(())
}
