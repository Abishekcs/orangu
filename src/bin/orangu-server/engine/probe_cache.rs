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

//! Start-up probe answers, kept across starts.
//!
//! Where a token is decoded is measured at every start
//! (`engine::decode_backend`): the model is built on the cores as well and a
//! few decode steps are timed on each — seconds of start-up for a model of a
//! few billion parameters. The answer depends on the device and its driver,
//! the model file, this build's kernels and the thread count, so it is kept
//! under exactly those in `~/.orangu/server/tuning/probes.json` and used
//! again while it is younger than [`MAX_AGE`]; past that it is measured
//! afresh, so a machine whose conditions drift is re-checked.
//!
//! Start-up choices between kernel families made by timing them (whether
//! the decode kernels take the integer dot) are kept the same way, under
//! the same key.
//!
//! `ORANGU_PROBE_CACHE=0` measures every time.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// How long a kept answer is trusted.
pub const MAX_AGE: Duration = Duration::from_secs(7 * 24 * 3600);

/// One decode-placement measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeProbe {
    /// Seconds per one-token decode step on the device.
    pub device_s: f64,
    /// The same on the cores.
    pub cpu_s: f64,
}

fn enabled() -> bool {
    crate::engine::env::flag_on_unless_disabled("ORANGU_PROBE_CACHE")
}

fn file() -> Option<PathBuf> {
    crate::engine::backend::tuning_profile::dir().map(|d| d.join("probes.json"))
}

/// What the answer for `model` on `device` depends on, as one string — or
/// `None` when the file cannot be described.
pub fn key(device: &str, model: &Path) -> Option<String> {
    let meta = std::fs::metadata(model).ok()?;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_secs();
    // The binary's own time as well as its version: two builds of one commit
    // can carry different kernels.
    let exe = std::env::current_exe()
        .and_then(std::fs::metadata)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    Some(format!(
        "{device}|{}|{}|{mtime}|{} {exe}|{}",
        model.display(),
        meta.len(),
        orangu::build_info::id(),
        rayon::current_num_threads()
    ))
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn read(path: &Path) -> serde_json::Map<String, serde_json::Value> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| match v {
            serde_json::Value::Object(m) => Some(m),
            _ => None,
        })
        .unwrap_or_default()
}

fn get_in(path: &Path, key: &str, at: u64) -> Option<DecodeProbe> {
    let map = read(path);
    let entry = map.get(key)?;
    let when = entry["when"].as_u64()?;
    if at.saturating_sub(when) > MAX_AGE.as_secs() {
        return None;
    }
    Some(DecodeProbe {
        device_s: entry["device_s"].as_f64()?,
        cpu_s: entry["cpu_s"].as_f64()?,
    })
}

fn put_in(path: &Path, key: &str, probe: DecodeProbe, at: u64) -> std::io::Result<()> {
    put_value_in(
        path,
        key,
        serde_json::json!({"device_s": probe.device_s, "cpu_s": probe.cpu_s, "when": at}),
        at,
    )
}

fn put_value_in(path: &Path, key: &str, value: serde_json::Value, at: u64) -> std::io::Result<()> {
    let mut map = read(path);
    // Entries past their age only take room.
    map.retain(|_, v| {
        v["when"]
            .as_u64()
            .is_some_and(|w| at.saturating_sub(w) <= MAX_AGE.as_secs())
    });
    map.insert(key.to_string(), value);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::Value::Object(map).to_string())?;
    std::fs::rename(tmp, path)
}

/// The key a kept on/off choice named `what` is stored under, beside the
/// decode probe of the same `key`.
fn choice_key(what: &str, key: &str) -> String {
    format!("{what}|{key}")
}

fn get_choice_in(path: &Path, key: &str, at: u64) -> Option<bool> {
    let map = read(path);
    let entry = map.get(key)?;
    let when = entry["when"].as_u64()?;
    if at.saturating_sub(when) > MAX_AGE.as_secs() {
        return None;
    }
    entry["on"].as_bool()
}

/// A kept on/off choice `what` for `key` — whether a kernel family won its
/// start-up timing, say — if there is a young enough one.
pub fn choice(what: &str, key: &str) -> Option<bool> {
    if !enabled() {
        return None;
    }
    get_choice_in(&file()?, &choice_key(what, key), now())
}

/// Keeps the choice `what` for `key`, with the two timings that made it.
pub fn keep_choice(what: &str, key: &str, on: bool, on_s: f64, off_s: f64) {
    if !enabled() {
        return;
    }
    let Some(path) = file() else {
        return;
    };
    let at = now();
    let value = serde_json::json!({"on": on, "on_s": on_s, "off_s": off_s, "when": at});
    if let Err(err) = put_value_in(&path, &choice_key(what, key), value, at) {
        log::warn!(
            "orangu-server: could not keep the {what} choice in {}: {err}",
            path.display()
        );
    }
}

/// A kept count `what` for `key` — the row count a kernel won its
/// start-up timing at, say — if there is a young enough one.
pub fn count(what: &str, key: &str) -> Option<usize> {
    if !enabled() {
        return None;
    }
    get_count_in(&file()?, &choice_key(what, key), now())
}

fn get_count_in(path: &Path, key: &str, at: u64) -> Option<usize> {
    let map = read(path);
    let entry = map.get(key)?;
    let when = entry["when"].as_u64()?;
    if at.saturating_sub(when) > MAX_AGE.as_secs() {
        return None;
    }
    usize::try_from(entry["count"].as_u64()?).ok()
}

/// Keeps the count `what` for `key`, with each candidate's timing.
pub fn keep_count(what: &str, key: &str, count: usize, timings: &[(usize, f64)]) {
    if !enabled() {
        return;
    }
    let Some(path) = file() else {
        return;
    };
    let at = now();
    let seconds: serde_json::Map<String, serde_json::Value> = timings
        .iter()
        .map(|&(c, s)| (c.to_string(), serde_json::json!(s)))
        .collect();
    let value = serde_json::json!({"count": count, "seconds": seconds, "when": at});
    if let Err(err) = put_value_in(&path, &choice_key(what, key), value, at) {
        log::warn!(
            "orangu-server: could not keep the {what} count in {}: {err}",
            path.display()
        );
    }
}

/// The kept decode probe for `key`, if there is a young enough one.
pub fn decode_probe(key: &str) -> Option<DecodeProbe> {
    if !enabled() {
        return None;
    }
    get_in(&file()?, key, now())
}

/// Keeps `probe` for `key`. A failure to write costs the next start a
/// measurement and nothing else, so it is logged and dropped.
pub fn keep_decode_probe(key: &str, probe: DecodeProbe) {
    if !enabled() {
        return;
    }
    let Some(path) = file() else {
        return;
    };
    if let Err(err) = put_in(&path, key, probe, now()) {
        log::warn!(
            "orangu-server: could not keep the decode probe in {}: {err}",
            path.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kept_probe_is_read_back_until_it_ages_out() {
        let dir = std::env::temp_dir().join(format!("orangu-probes-{}", std::process::id()));
        let path = dir.join("probes.json");
        let p = DecodeProbe {
            device_s: 0.0153,
            cpu_s: 0.0641,
        };
        put_in(&path, "k", p, 1000).unwrap();
        assert_eq!(get_in(&path, "k", 1000 + 60), Some(p));
        assert_eq!(get_in(&path, "other", 1000), None);
        assert_eq!(get_in(&path, "k", 1000 + MAX_AGE.as_secs() + 1), None);
        // A later write drops the aged entry and keeps the new one.
        put_in(&path, "k2", p, 1000 + MAX_AGE.as_secs() + 1).unwrap();
        assert_eq!(read(&path).len(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_kept_choice_sits_beside_the_probe_of_its_key() {
        let dir = std::env::temp_dir().join(format!("orangu-choices-{}", std::process::id()));
        let path = dir.join("probes.json");
        let p = DecodeProbe {
            device_s: 0.02,
            cpu_s: 0.05,
        };
        put_in(&path, "k", p, 1000).unwrap();
        let value = serde_json::json!({"on": false, "on_s": 0.03, "off_s": 0.02, "when": 1000});
        put_value_in(&path, &choice_key("i8", "k"), value, 1000).unwrap();
        assert_eq!(get_in(&path, "k", 1000), Some(p));
        assert_eq!(
            get_choice_in(&path, &choice_key("i8", "k"), 1000),
            Some(false)
        );
        assert_eq!(get_choice_in(&path, &choice_key("i8", "other"), 1000), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_kept_count_reads_back_and_a_choice_is_not_one() {
        let dir = std::env::temp_dir().join(format!("orangu-counts-{}", std::process::id()));
        let path = dir.join("probes.json");
        let value =
            serde_json::json!({"count": 4, "seconds": {"0": 0.03, "4": 0.02}, "when": 1000});
        put_value_in(&path, &choice_key("q4k-short", "k"), value, 1000).unwrap();
        let value = serde_json::json!({"on": true, "on_s": 0.02, "off_s": 0.03, "when": 1000});
        put_value_in(&path, &choice_key("mmq", "k"), value, 1000).unwrap();
        assert_eq!(
            get_count_in(&path, &choice_key("q4k-short", "k"), 1000),
            Some(4)
        );
        assert_eq!(get_count_in(&path, &choice_key("mmq", "k"), 1000), None);
        assert_eq!(
            get_count_in(
                &path,
                &choice_key("q4k-short", "k"),
                1000 + MAX_AGE.as_secs() + 1
            ),
            None
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_key_names_the_file_and_the_build() {
        let me = std::env::current_exe().unwrap();
        let k = key("1002-7340-radv", &me).unwrap();
        assert!(k.starts_with("1002-7340-radv|"));
        assert!(k.contains(orangu::build_info::id()));
        assert!(key("d", Path::new("/no/such/file")).is_none());
    }
}
