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

//! Whether a speculative step is worth taking.
//!
//! A verify pass yields the tokens of the draft it accepts plus one; a plain
//! decode step yields one. Which is faster depends on the machine — a verify
//! over five positions costs one decode step where the weights do not fit
//! the device and several where they do — and on the text: a reply that
//! echoes its context accepts drafts deep, prose rarely. So both costs are
//! measured as decoding goes, and a step verifies only while
//! `tokens per verify / verify time` beats `rows / step time`.
//!
//! `rows` is how many requests share the step (`engine::decode_batch`): a
//! batched step yields a token for each of them. A verify the model takes
//! into the batch (`ModelForward::batches_verify`) widens that step: it
//! yields its tokens plus one for every other row, in the verify's time, so
//! it pays while `(tokens per verify + rows − 1) / verify time` beats
//! `rows / step time`. A verify that runs outside the batch stops it, and
//! has to beat `rows / step time` on its own tokens: a verify that pays for
//! one stream can cost a batch of four, and is taken there only on a draft
//! that is accepted deep.
//!
//! The two times are a property of the machine and model and are kept for
//! the process; the yield is a property of the text and is kept per
//! request. While verifying loses, one chance in [`PROBE_EVERY`] verifies
//! anyway, so a stretch of echoed text is noticed when it starts.
//!
//! A reply that echoes its context is verified whatever the gate says (the
//! drafter knows: its match runs far back), and the control picks the
//! width of those verifies instead: the drafter's own, or its wider echo
//! length. Each is timed for the process and its yield on echoes kept for
//! the request, and the one with more tokens a millisecond is taken, the
//! other once in [`PROBE_EVERY`] echoes so it stays measured. A wider verify
//! costs more rows; whether the rows are cheap enough is the machine's and
//! the model's, and is measured rather than assumed.
//!
//! `ORANGU_SPEC_ADAPTIVE=0` verifies every draft, an echo at its wider
//! length.

use std::sync::Mutex;
use std::time::Duration;

/// While verifying loses, verify anyway once in this many chances.
pub const PROBE_EVERY: u32 = 16;
/// Tokens per verify assumed before any was measured: optimistic, so the
/// first drafts are tried rather than written off unmeasured.
const PRIOR_YIELD: f64 = 2.0;
/// Weight of the newest measurement in each average.
const ALPHA: f64 = 0.1;
/// Verifies after which, with no plain step timed yet, one plain step is
/// taken to time it.
const VERIFIES_BEFORE_A_TIMED_STEP: u32 = 4;

#[derive(Clone, Copy, Default)]
struct Costs {
    step_ms: Option<f64>,
    verify_ms: Option<f64>,
    /// A verify at the drafter's echo width.
    wide_verify_ms: Option<f64>,
}

static COSTS: Mutex<Costs> = Mutex::new(Costs {
    step_ms: None,
    verify_ms: None,
    wide_verify_ms: None,
});

fn average(old: Option<f64>, new: f64) -> f64 {
    old.map_or(new, |old| old + ALPHA * (new - old))
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// One request's view: its yield, and whether to verify the next draft.
pub struct DraftControl {
    adaptive: bool,
    yield_per_verify: f64,
    skipped: u32,
    verifies: u32,
    /// Tokens per verify of an echo, at the drafter's own width and at its
    /// echo width.
    echo_yield: [Option<f64>; 2],
    echoes: u32,
}

impl DraftControl {
    pub fn new() -> Self {
        Self::with(
            cfg!(not(test)) && crate::engine::env::flag_on_unless_disabled("ORANGU_SPEC_ADAPTIVE"),
        )
    }

    fn with(adaptive: bool) -> Self {
        Self {
            adaptive,
            yield_per_verify: PRIOR_YIELD,
            skipped: 0,
            verifies: 0,
            echo_yield: [None, None],
            echoes: 0,
        }
    }

    /// Whether the next step drafts and verifies rather than decodes, with
    /// `rows` requests sharing a decode step and the verify riding in it
    /// when `in_batch`.
    pub fn should_verify(&mut self, rows: usize, in_batch: bool) -> bool {
        if !self.adaptive {
            return true;
        }
        let costs = *COSTS.lock().unwrap_or_else(|p| p.into_inner());
        self.decide(costs, rows, in_batch)
    }

    fn decide(&mut self, costs: Costs, rows: usize, in_batch: bool) -> bool {
        let (Some(step), Some(verify)) = (costs.step_ms, costs.verify_ms) else {
            // A missing time is measured by doing that thing.
            return costs.step_ms.is_some() || self.verifies < VERIFIES_BEFORE_A_TIMED_STEP;
        };
        let rows = rows.max(1) as f64;
        let others = if in_batch { rows - 1.0 } else { 0.0 };
        if (self.yield_per_verify + others) / verify >= rows / step {
            self.skipped = 0;
            return true;
        }
        self.skipped += 1;
        if self.skipped >= PROBE_EVERY {
            self.skipped = 0;
            return true;
        }
        false
    }

    /// Whether the next echo is verified at the drafter's echo width.
    pub fn echo_wide(&mut self) -> bool {
        if !self.adaptive {
            return true;
        }
        let costs = *COSTS.lock().unwrap_or_else(|p| p.into_inner());
        self.choose_wide(costs)
    }

    fn choose_wide(&mut self, costs: Costs) -> bool {
        self.echoes += 1;
        // An unmeasured width is measured by taking it, the wide one first.
        let (Some(wide_ms), Some(wide_yield)) = (costs.wide_verify_ms, self.echo_yield[1]) else {
            return true;
        };
        let (Some(own_ms), Some(own_yield)) = (costs.verify_ms, self.echo_yield[0]) else {
            return false;
        };
        let wide = wide_yield / wide_ms >= own_yield / own_ms;
        if self.echoes.is_multiple_of(PROBE_EVERY) {
            !wide
        } else {
            wide
        }
    }

    /// A verify pass took `took` and yielded `tokens` (accepted drafts plus
    /// the model's own next token); `echo` is the width of a verify taken
    /// for an echo — `Some(true)` the echo width — and `None` for any other.
    pub fn verified(&mut self, tokens: usize, took: Duration, echo: Option<bool>) {
        self.verifies += 1;
        let tokens = tokens as f64;
        let wide = echo == Some(true);
        // The gate weighs the drafter's own width; an echo verified wide
        // says nothing about that.
        if !wide {
            self.yield_per_verify =
                self.yield_per_verify + ALPHA * (tokens - self.yield_per_verify);
        }
        if let Some(w) = echo {
            let slot = &mut self.echo_yield[usize::from(w)];
            *slot = Some(average(*slot, tokens));
        }
        let mut costs = COSTS.lock().unwrap_or_else(|p| p.into_inner());
        if wide {
            costs.wide_verify_ms = Some(average(costs.wide_verify_ms, ms(took)));
        } else {
            costs.verify_ms = Some(average(costs.verify_ms, ms(took)));
        }
    }

    /// A plain decode step took `took`.
    pub fn stepped(&self, took: Duration) {
        let mut costs = COSTS.lock().unwrap_or_else(|p| p.into_inner());
        costs.step_ms = Some(average(costs.step_ms, ms(took)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn costs(step: f64, verify: f64) -> Costs {
        Costs {
            step_ms: Some(step),
            verify_ms: Some(verify),
            ..Costs::default()
        }
    }

    /// The echo width is tried first, then whichever width yields more
    /// tokens a millisecond is taken, the other once in [`PROBE_EVERY`].
    #[test]
    fn an_echo_takes_the_width_with_more_tokens_a_millisecond() {
        let mut c = DraftControl::with(true);
        let unmeasured = costs(16.0, 40.0);
        assert!(c.choose_wide(unmeasured));
        c.echo_yield[1] = Some(17.0);
        let wide = Costs {
            wide_verify_ms: Some(100.0),
            ..unmeasured
        };
        // The wide width measured, the own one not yet for echoes.
        assert!(!c.choose_wide(wide));
        // 17 / 100 against 5 / 40: wide.
        c.echo_yield[0] = Some(5.0);
        c.echoes = 0;
        let taken = (0..PROBE_EVERY).filter(|_| c.choose_wide(wide)).count();
        assert_eq!(taken as u32, PROBE_EVERY - 1);
        // 17 / 200 against 5 / 40: the own width.
        let dear = Costs {
            wide_verify_ms: Some(200.0),
            ..unmeasured
        };
        c.echoes = 0;
        let taken = (0..PROBE_EVERY).filter(|_| c.choose_wide(dear)).count();
        assert_eq!(taken, 1);
    }

    #[test]
    fn an_echo_verified_wide_leaves_the_gates_yield_alone() {
        let mut c = DraftControl::with(true);
        c.verified(17, Duration::from_millis(1), Some(true));
        assert_eq!(c.yield_per_verify, PRIOR_YIELD);
        assert_eq!(c.echo_yield[1], Some(17.0));
        c.verified(5, Duration::from_millis(1), Some(false));
        assert!(c.yield_per_verify > PRIOR_YIELD);
        assert_eq!(c.echo_yield[0], Some(5.0));
    }

    #[test]
    fn a_verify_is_taken_only_while_it_beats_the_step() {
        let mut c = DraftControl::with(true);
        // Two tokens a verify at 30 ms against 16 ms a step: 0.067 > 0.0625.
        assert!(c.decide(costs(16.0, 30.0), 1, false));
        // At 45 ms it loses.
        assert!(!c.decide(costs(16.0, 45.0), 1, false));
    }

    /// A batched step yields a token a row, so a verify that beats one
    /// stream's step loses to four streams' — unless its draft goes deep.
    #[test]
    fn a_verify_must_beat_every_row_of_a_batched_step() {
        let mut c = DraftControl::with(true);
        assert!(c.decide(costs(30.0, 30.0), 1, false));
        assert!(!c.decide(costs(30.0, 30.0), 4, false));
        c.yield_per_verify = 4.0;
        assert!(c.decide(costs(30.0, 30.0), 4, false));
    }

    /// A verify inside the batch costs the other rows only the time it adds
    /// to their step: at 4 rows it pays on a yield that would lose outside.
    #[test]
    fn a_verify_in_the_batch_is_charged_only_what_it_adds() {
        let mut c = DraftControl::with(true);
        c.yield_per_verify = 2.0;
        // Outside: 2 / 36 < 4 / 30.
        assert!(!c.decide(costs(30.0, 36.0), 4, false));
        // Inside: (2 + 3) / 36 ≥ 4 / 30.
        assert!(c.decide(costs(30.0, 36.0), 4, true));
        // A verify that doubles the step does not pay either way.
        assert!(!c.decide(costs(30.0, 60.0), 4, true));
    }

    #[test]
    fn a_losing_verify_is_still_probed_now_and_then() {
        let mut c = DraftControl::with(true);
        let taken = (0..PROBE_EVERY * 3)
            .filter(|_| c.decide(costs(16.0, 45.0), 1, false))
            .count();
        assert_eq!(taken, 3);
    }

    #[test]
    fn deep_acceptance_turns_verifying_back_on() {
        let mut c = DraftControl::with(true);
        c.yield_per_verify = 1.2;
        assert!(!c.decide(costs(16.0, 45.0), 1, false));
        c.yield_per_verify = 4.0;
        assert!(c.decide(costs(16.0, 45.0), 1, false));
    }

    #[test]
    fn an_unmeasured_step_is_timed_after_a_few_verifies() {
        let mut c = DraftControl::with(true);
        let unmeasured = Costs {
            step_ms: None,
            verify_ms: Some(40.0),
            ..Costs::default()
        };
        assert!(c.decide(unmeasured, 1, false));
        c.verifies = VERIFIES_BEFORE_A_TIMED_STEP;
        assert!(!c.decide(unmeasured, 1, false));
    }

    #[test]
    fn without_adaptation_every_draft_is_verified() {
        let mut c = DraftControl::with(false);
        assert!(c.should_verify(4, false));
    }
}
