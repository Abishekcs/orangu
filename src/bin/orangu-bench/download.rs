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

//! `--download-probe`: how fast `orangu-server download` fetches from the
//! Hub on this link, against how many connections it uses.
//!
//! # Why connections, and why the real download path
//!
//! A CDN caps what one connection carries, and that cap — not the link — is
//! what limits a single stream on a fast line; on a slow one, a single stream
//! already fills the link and more only add overhead. Which of the two a
//! machine is on is not knowable without measuring, and the answer is what
//! `ORANGU_DOWNLOAD_CONNECTIONS` is set from. So the probe runs the server's
//! own download code — the same pieces, retries and writes to disk — at each
//! connection count, for a fixed window, rather than timing `curl`.
//!
//! # Both directions
//!
//! A shared link changes under a probe that takes minutes: someone else
//! starts a transfer, a CDN node warms up. As in `--storage-probe`, every
//! count is measured twice, once ascending and once descending, and the mean
//! reported; when the two passes disagree the link changed during the probe,
//! and the table says so.

use orangu::model_download::{ProbeTarget, probe_download};
use std::path::Path;
use std::time::Duration;

/// One connection count's result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub connections: u32,
    /// Ascending-pass rate, MB/s (10^6 bytes, as link speeds are quoted).
    pub up: f64,
    /// Descending-pass rate, MB/s.
    pub down: f64,
}

impl Point {
    pub fn mean(&self) -> f64 {
        (self.up + self.down) / 2.0
    }

    /// How far apart the two passes are, as a fraction of the faster.
    pub fn spread(&self) -> f64 {
        let hi = self.up.max(self.down);
        if hi <= 0.0 {
            0.0
        } else {
            (self.up - self.down).abs() / hi
        }
    }
}

/// One timed download, in MB/s.
fn rate(
    target: &ProbeTarget,
    dir: &Path,
    connections: u32,
    window: Duration,
) -> anyhow::Result<f64> {
    let (bytes, elapsed) = probe_download(target, dir, connections as usize, window)?;
    Ok(bytes as f64 / 1e6 / elapsed.as_secs_f64().max(1e-9))
}

/// Every count ascending, then every count descending. `progress` hears of
/// each measurement as it lands — the sweep takes minutes.
pub fn sweep(
    target: &ProbeTarget,
    dir: &Path,
    counts: &[u32],
    window: Duration,
    mut progress: impl FnMut(&str, u32, f64),
) -> anyhow::Result<Vec<Point>> {
    let mut ascending = Vec::with_capacity(counts.len());
    for &n in counts {
        let mb_s = rate(target, dir, n, window)?;
        progress("up", n, mb_s);
        ascending.push(mb_s);
    }
    let mut descending = vec![0.0; counts.len()];
    for (i, &n) in counts.iter().enumerate().rev() {
        let mb_s = rate(target, dir, n, window)?;
        progress("down", n, mb_s);
        descending[i] = mb_s;
    }
    Ok(counts
        .iter()
        .zip(ascending)
        .zip(descending)
        .map(|((&connections, up), down)| Point {
            connections,
            up,
            down,
        })
        .collect())
}

/// The probe's table, as the report prints it.
pub fn table(points: &[Point]) -> String {
    let mut out = String::from(
        "  conns |     MB/s |       up |     down |  spread\n\
         --------------------------------------------------\n",
    );
    for p in points {
        out.push_str(&format!(
            "  {:>5} | {:8.1} | {:8.1} | {:8.1} | {:6.1}%\n",
            p.connections,
            p.mean(),
            p.up,
            p.down,
            100.0 * p.spread(),
        ));
    }
    if let Some(worst) = points
        .iter()
        .max_by(|a, b| a.spread().total_cmp(&b.spread()))
        && worst.spread() > 0.25
    {
        out.push_str(&format!(
            "  the {}-connection point differs by {:.0}% between the ascending and descending \
             passes —\n  the link changed during the probe, so read the two columns, not the mean\n",
            worst.connections,
            100.0 * worst.spread(),
        ));
    }
    out
}

/// The probe's history rows: one per connection count, `n` the count.
pub fn records(points: &[Point], label: &str) -> Vec<super::history::Record> {
    let date = super::history::today();
    points
        .iter()
        .map(|p| super::history::Record {
            date: date.clone(),
            label: label.to_string(),
            mode: "download_mb_s".to_string(),
            n: p.connections,
            best: p.up.max(p.down),
            mean: p.mean(),
            sd: (p.up - p.down).abs() / 2.0,
            sd_sample: None,
            device: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(connections: u32, up: f64, down: f64) -> Point {
        Point {
            connections,
            up,
            down,
        }
    }

    #[test]
    fn the_reported_rate_is_the_mean_of_both_passes() {
        assert!((point(8, 60.0, 40.0).mean() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn a_changing_link_is_called_out() {
        assert!(!table(&[point(1, 5.0, 5.0)]).contains("changed"));
        assert!(table(&[point(1, 5.0, 5.0), point(8, 50.0, 10.0)]).contains("8-connection"));
    }

    #[test]
    fn records_carry_the_count_and_both_passes() {
        let rows = records(&[point(4, 30.0, 20.0)], "link");
        assert_eq!(rows[0].mode, "download_mb_s");
        assert_eq!(rows[0].n, 4);
        assert_eq!(rows[0].best, 30.0);
        assert_eq!(rows[0].mean, 25.0);
    }
}
