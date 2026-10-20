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

//! A tuning profile per device and driver.
//!
//! The kernels' parameters are compiled-in defaults that an environment
//! variable can override ([`super::env_tuning_value`]). A profile is the same
//! overrides kept on disk for one device: what a sweep found best on this
//! card, applied on every start without anyone exporting anything.
//!
//! Precedence is the environment, then the profile, then the default — so a
//! sweep that sets a variable still measures the value it set.
//!
//! The file is `~/.orangu/server/tuning/<vendor>-<device>-<driver>.json`:
//!
//! ```json
//! { "values": { "ORANGU_REDUCE_N_ROWS": "2", "ORANGU_COOP_MIN_N_TOKENS": "32" } }
//! ```
//!
//! The driver is part of the name because a driver update changes which
//! choice is fastest; a profile from the previous driver is not applied to
//! the next one. `ORANGU_TUNING_PROFILE=0` ignores the profile.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

/// What was loaded, once per process.
struct Profile {
    path: PathBuf,
    values: HashMap<String, String>,
}

static PROFILE: OnceLock<Option<Profile>> = OnceLock::new();

/// The device's identity as [`file_name`] spells it, without the extension —
/// what anything kept per device is keyed by.
static DEVICE_KEY: OnceLock<String> = OnceLock::new();

#[derive(serde::Deserialize)]
struct File {
    #[serde(default)]
    values: HashMap<String, serde_json::Value>,
}

/// The profile's file name for a device: vendor and device ids, and the
/// driver's name and version reduced to a file-name-safe slug.
pub fn file_name(vendor: u32, device: u32, driver: &str, driver_info: &str) -> String {
    let slug: String = format!("{driver}-{driver_info}")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    let slug = slug.trim_matches('_');
    format!("{vendor:04x}-{device:04x}-{slug}.json")
}

/// Where profiles live.
pub fn dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".orangu/server/tuning"))
}

/// Loads the profile for this adapter. Only the first call counts: knobs are
/// read once, and a second device's profile arriving later could not change
/// them consistently.
pub fn install(info: &wgpu::AdapterInfo) {
    DEVICE_KEY.get_or_init(|| {
        file_name(info.vendor, info.device, &info.driver, &info.driver_info)
            .trim_end_matches(".json")
            .to_string()
    });
    PROFILE.get_or_init(|| {
        if std::env::var("ORANGU_TUNING_PROFILE").is_ok_and(|v| v.trim() == "0") {
            return None;
        }
        let path = dir()?.join(file_name(
            info.vendor,
            info.device,
            &info.driver,
            &info.driver_info,
        ));
        load(path)
    });
}

fn load(path: PathBuf) -> Option<Profile> {
    let text = std::fs::read_to_string(&path).ok()?;
    let file: File = match serde_json::from_str(&text) {
        Ok(file) => file,
        Err(err) => {
            log::warn!(
                "orangu-server: tuning profile {} is not valid ({err}); ignoring it",
                path.display()
            );
            return None;
        }
    };
    let values: HashMap<String, String> = file
        .values
        .into_iter()
        .map(|(k, v)| {
            let v = match v {
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            };
            (k, v)
        })
        .collect();
    log::info!(
        "orangu-server: tuning profile {} — {} value(s)",
        path.display(),
        values.len()
    );
    Some(Profile { path, values })
}

/// The first device's key — see [`DEVICE_KEY`].
pub fn device_key() -> Option<&'static str> {
    DEVICE_KEY.get().map(String::as_str)
}

/// `var` from the environment, else from the profile.
pub fn var(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .or_else(|| value(name).map(str::to_string))
}

/// `name` from the profile alone.
pub fn value(name: &str) -> Option<&'static str> {
    PROFILE
        .get()?
        .as_ref()?
        .values
        .get(name)
        .map(String::as_str)
}

/// The loaded profile's path, for the startup report and `/props`.
pub fn loaded_path() -> Option<&'static std::path::Path> {
    Some(PROFILE.get()?.as_ref()?.path.as_path())
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_file_name_carries_the_driver_version() {
        let a = super::file_name(0x1002, 0x7340, "radv", "Mesa 25.2.1");
        let b = super::file_name(0x1002, 0x7340, "radv", "Mesa 25.3.0");
        assert_eq!(a, "1002-7340-radv_mesa_25.2.1.json");
        assert_ne!(a, b, "a driver update must not reuse the old profile");
    }

    #[test]
    fn a_profile_reads_strings_and_numbers() {
        let dir = std::env::temp_dir().join(format!("orangu-tuning-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.json");
        std::fs::write(&path, r#"{"values":{"A":"2","B":32}}"#).unwrap();
        let p = super::load(path).unwrap();
        assert_eq!(p.values["A"], "2");
        assert_eq!(p.values["B"], "32");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
