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

//! `--workers on|off|compare` (W-84): one orangu-server `[workers]` tree
//! measured through its tree and with its top-level node alone, switched
//! through `POST /props`.

use crate::history::Record;

/// The `workers` object of `GET /props`: `None` from a server without a
/// `[workers]` section, or one that is not orangu-server.
pub fn state(
    client: &reqwest::blocking::Client,
    url: &str,
) -> anyhow::Result<Option<serde_json::Value>> {
    let props: serde_json::Value = client.get(format!("{url}/props")).send()?.json()?;
    Ok(props.get("workers").filter(|w| !w.is_null()).cloned())
}

/// Serves through the tree (`true`) or alone. The server refuses while a
/// request runs, and on a node that cannot switch; its reason is the error.
pub fn switch(client: &reqwest::blocking::Client, url: &str, enabled: bool) -> anyhow::Result<()> {
    // A slot is freed a moment after its answer ends: a refusal for running
    // requests is asked again for a few seconds.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let resp = loop {
        let resp = client
            .post(format!("{url}/props"))
            .json(&serde_json::json!({"workers": {"enabled": enabled}}))
            .send()?;
        if resp.status() != reqwest::StatusCode::CONFLICT || std::time::Instant::now() >= deadline {
            break resp;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    };
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        anyhow::bail!(
            "the server would not switch its workers {} ({status}): {}",
            if enabled { "on" } else { "off" },
            body.trim()
        );
    }
    Ok(())
}

/// The modes whose rows are rates in tokens a second: what is compared.
/// The rest a run records beside them (bytes read a token, time to the
/// first token) are in each arm's own report.
const RATES: [&str; 5] = ["pp", "tg", "pg", "curve", "embed"];

/// One arm of a comparison, as the table prints it.
fn key(r: &Record) -> (String, u32) {
    (r.mode.clone(), r.n)
}

/// The rows of `off` and `on` side by side, in `off`'s order, with the tree's
/// rate over the node's alone.
pub fn table(off: &[Record], on: &[Record]) -> Vec<String> {
    let mut lines = vec![
        "    mode |       n |  alone best |  alone mean |   tree best |   tree mean |  tree/alone"
            .to_string(),
        "-".repeat(86),
    ];
    for a in off.iter().filter(|r| RATES.contains(&r.mode.as_str())) {
        let Some(b) = on.iter().find(|b| key(b) == key(a)) else {
            continue;
        };
        let ratio = if a.mean > 0.0 {
            format!("{:>10.2}x", b.mean / a.mean)
        } else {
            format!("{:>11}", "-")
        };
        lines.push(format!(
            "{:>8} | {:>7} | {:>11.2} | {:>11.2} | {:>11.2} | {:>11.2} | {ratio}",
            a.mode, a.n, a.best, a.mean, b.best, b.mean
        ));
    }
    lines
}

/// The same, as JSON.
pub fn json(off: &[Record], on: &[Record]) -> serde_json::Value {
    let rows: Vec<_> = off
        .iter()
        .filter(|r| RATES.contains(&r.mode.as_str()))
        .filter_map(|a| {
            let b = on.iter().find(|b| key(b) == key(a))?;
            Some(serde_json::json!({
                "mode": a.mode,
                "n": a.n,
                "alone": {"best": a.best, "mean": a.mean, "sd": a.sd},
                "tree": {"best": b.best, "mean": b.mean, "sd": b.sd},
            }))
        })
        .collect();
    serde_json::json!({ "workers_compare": rows })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(mode: &str, n: u32, mean: f64) -> Record {
        Record {
            date: "2026-09-29".into(),
            label: "x".into(),
            mode: mode.into(),
            n,
            best: mean + 1.0,
            mean,
            sd: 0.5,
            sd_sample: None,
            device: None,
        }
    }

    #[test]
    fn the_two_ways_are_set_side_by_side() {
        let off = [
            record("pp", 512, 90.0),
            record("tg", 0, 5.0),
            record("io_mb_per_token", 0, 0.0),
        ];
        let on = [
            record("tg", 0, 6.0),
            record("pp", 512, 45.0),
            record("io_mb_per_token", 0, 0.0),
        ];
        let lines = table(&off, &on);
        assert_eq!(lines.len(), 4, "{lines:#?}");
        assert!(
            lines[2].contains("512") && lines[2].ends_with("0.50x"),
            "{}",
            lines[2]
        );
        assert!(lines[3].ends_with("1.20x"), "{}", lines[3]);
        let value = json(&off, &on);
        assert_eq!(value["workers_compare"][1]["tree"]["mean"], 6.0);
        assert_eq!(
            value["workers_compare"].as_array().unwrap().len(),
            2,
            "rates only"
        );
        // A row only one way ran is left out.
        assert_eq!(table(&off[..1], &on[..1]).len(), 2);
    }
}
