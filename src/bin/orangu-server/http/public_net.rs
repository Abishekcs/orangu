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

//! The guard on URLs a request makes the server fetch — a picture or a
//! document named by URL, a `web_fetch` call. Unless
//! `[orangu-server].fetch_private` is on, such a fetch reaches public
//! addresses only: a name is resolved and its loopback, private, link-local
//! and reserved addresses are dropped, a literal address must be public
//! itself, and every redirect is checked the same way. A client with the
//! API key cannot use the server to reach the machine it runs on or the
//! network behind it.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// The most redirects one fetch follows.
const MAX_REDIRECTS: usize = 10;

static PRIVATE_ALLOWED: AtomicBool = AtomicBool::new(false);

/// Sets `[orangu-server].fetch_private`, once at start-up.
pub fn set_private_allowed(allowed: bool) {
    PRIVATE_ALLOWED.store(allowed, Ordering::Relaxed);
}

fn private_allowed() -> bool {
    PRIVATE_ALLOWED.load(Ordering::Relaxed)
}

/// Whether `ip` is an address on the public internet.
pub fn is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_public_v4(v4),
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_public_v4(v4);
            }
            let s = v6.segments();
            !(v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || (s[0] & 0xfe00) == 0xfc00 // unique local
                || (s[0] & 0xffc0) == 0xfe80 // link-local
                || (s[0] == 0x2001 && s[1] == 0x0db8) // documentation
                || (s[0] == 0x0064 && s[1] == 0xff9b && !is_public_v4(embedded_v4(v6)))
                || (s[0] == 0 && s[1] == 0 && s[2] == 0 && s[3] == 0 && s[4] == 0 && s[5] == 0))
        }
    }
}

fn embedded_v4(v6: Ipv6Addr) -> Ipv4Addr {
    let o = v6.octets();
    Ipv4Addr::new(o[12], o[13], o[14], o[15])
}

fn is_public_v4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_unspecified()
        || ip.is_broadcast()
        || ip.is_documentation()
        || ip.is_multicast()
        || a == 0
        || (a == 100 && (b & 0xc0) == 64) // shared address space
        || (a == 192 && b == 0 && c == 0) // protocol assignments
        || (a == 198 && (b & 0xfe) == 18) // benchmarking
        || a >= 240)
}

/// Refuses `url` unless it is `http(s)` and, where its host is a literal
/// address, that address is public. A named host is checked when it is
/// resolved.
pub fn check(url: &reqwest::Url) -> Result<(), String> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err(format!("{url} is not an http(s) URL"));
    }
    if private_allowed() {
        return Ok(());
    }
    let Some(host) = url.host_str() else {
        return Err(format!("{url} names no host"));
    };
    let literal = host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .parse::<IpAddr>()
        .ok();
    match literal {
        Some(ip) if !is_public(ip) => Err(format!(
            "{url} is not a public address, and this server fetches only those"
        )),
        _ => Ok(()),
    }
}

/// Resolves names to their public addresses only.
struct PublicOnly;

impl reqwest::dns::Resolve for PublicOnly {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let host = name.as_str().to_string();
        Box::pin(async move {
            let all: Vec<SocketAddr> = tokio::net::lookup_host((host.as_str(), 0)).await?.collect();
            let public: Vec<SocketAddr> = all
                .into_iter()
                .filter(|addr| private_allowed() || is_public(addr.ip()))
                .collect();
            if public.is_empty() {
                return Err(format!(
                    "{host} has no public address, and this server fetches only those"
                )
                .into());
            }
            Ok(Box::new(public.into_iter()) as reqwest::dns::Addrs)
        })
    }
}

fn redirects() -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::custom(|attempt| {
        if attempt.previous().len() >= MAX_REDIRECTS {
            return attempt.error("too many redirects");
        }
        match check(attempt.url()) {
            Ok(()) => attempt.follow(),
            Err(err) => attempt.error(err),
        }
    })
}

/// `builder` with the guard: names resolve to public addresses only and
/// redirects are checked. Each URL the client is handed still goes through
/// [`check`] first.
pub fn guarded(builder: reqwest::ClientBuilder) -> reqwest::ClientBuilder {
    builder
        .dns_resolver(Arc::new(PublicOnly))
        .redirect(redirects())
}

/// [`guarded`] for a blocking client.
pub fn guarded_blocking(
    builder: reqwest::blocking::ClientBuilder,
) -> reqwest::blocking::ClientBuilder {
    builder
        .dns_resolver(Arc::new(PublicOnly))
        .redirect(redirects())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Loopback, private, link-local, shared, reserved and mapped forms of
    /// them are not public; ordinary internet addresses are.
    #[test]
    fn only_internet_addresses_are_public() {
        for private in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "100.64.0.1",
            "0.0.0.0",
            "255.255.255.255",
            "240.0.0.1",
            "::1",
            "::",
            "fd00::1",
            "fe80::1",
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.1",
            "64:ff9b::a00:1",
            "::7f00:1",
        ] {
            assert!(!is_public(private.parse().unwrap()), "{private}");
        }
        for public in [
            "93.184.216.34",
            "1.1.1.1",
            "2606:4700::1111",
            "::ffff:8.8.8.8",
        ] {
            assert!(is_public(public.parse().unwrap()), "{public}");
        }
    }

    /// A literal private address is refused before any connection; a name
    /// passes to the resolver; only http(s) is fetched.
    #[test]
    fn a_literal_private_address_is_refused() {
        let url = |s: &str| reqwest::Url::parse(s).unwrap();
        assert!(check(&url("http://127.0.0.1:8080/x")).is_err());
        assert!(check(&url("http://[::1]/x")).is_err());
        assert!(check(&url("http://169.254.169.254/latest")).is_err());
        assert!(check(&url("https://example.com/a.png")).is_ok());
        assert!(check(&url("https://1.1.1.1/")).is_ok());
        assert!(check(&url("file:///etc/passwd")).is_err());
    }

    /// `localhost` resolves to nothing public, so a guarded client refuses it.
    #[test]
    fn a_name_for_this_machine_is_refused() {
        let client = guarded_blocking(reqwest::blocking::Client::builder())
            .build()
            .unwrap();
        let err = client.get("http://localhost:9/").send().unwrap_err();
        assert!(format!("{err:?}").contains("no public address"), "{err:?}");
    }
}
