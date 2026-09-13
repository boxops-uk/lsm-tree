// Copyright (c) 2024-present, fjall-rs
// This source code is licensed under both the Apache 2.0 and MIT License
// (found in the LICENSE-* files in the repository)

//! **How far a reader may see, and the rule that keeps it honest.**

use crate::SeqNo;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering::AcqRel, Ordering::Acquire};
use std::sync::{Arc, Mutex};

#[derive(Default, Debug)]
struct InFlight {
    /// Sequence numbers taken and not yet finished.
    taken: BTreeSet<SeqNo>,
    /// The highest that has finished, publishable or not.
    done: SeqNo,
}

/// **The visible sequence number: one watermark, and who is allowed to move it.**
///
/// A reader snapshots by taking this value and is then promised every write below it.
/// That promise only holds if a write advances the watermark *after* its rows are
/// readable and *only once every lower-numbered write has landed too* — otherwise a
/// snapshot has a hole in it, and the reader is never told.
///
/// Ordering used to be a side effect of geography: writers applied their rows while
/// holding one global lock, so they could not overtake one another. Anything that
/// shortens that lock — and the write path wants to, since the lock covers work
/// proportional to the rows in a batch — takes the ordering with it.
///
/// So the rule lives here instead, on the number itself. A writer [`begin`]s, which
/// hands back a [`Pending`] carrying its sequence number, and the watermark cannot reach
/// that number until the `Pending` is finished. There is no way to move this counter
/// that skips the queue, which is the point: the previous arrangement had five callers
/// in one crate and several more in another, and being right depended on every one of
/// them remembering.
///
/// [`begin`]: VisibleSeqno::begin
#[derive(Clone, Default, Debug)]
pub struct VisibleSeqno(Arc<Inner>);

#[derive(Default, Debug)]
struct Inner {
    visible: AtomicU64,
    in_flight: Mutex<InFlight>,
}

impl VisibleSeqno {
    /// A watermark recovered at `prev`.
    #[must_use]
    pub fn new(prev: SeqNo) -> Self {
        let this = Self::default();
        this.0.visible.store(prev, std::sync::atomic::Ordering::Release);
        this
    }

    /// How far a reader may currently see.
    #[must_use]
    pub fn get(&self) -> SeqNo {
        self.0.visible.load(Acquire)
    }

    /// Take `seqno` out of circulation until the returned [`Pending`] is finished.
    ///
    /// Call this **before** anything can observe the write missing — before the rows go
    /// anywhere — or there is a window in which a later writer can publish past it.
    #[must_use = "the watermark cannot pass this sequence number until it is finished, \
                  so dropping it immediately is the same as not calling begin at all"]
    pub fn begin(&self, seqno: SeqNo) -> Pending {
        self.begin_range(seqno, 1)
    }

    /// Take the whole run `[base, base + count)` out of circulation at once.
    ///
    /// **For a writer that hands out sequence numbers in order.** A group commit assigns
    /// a contiguous run to the writes it batched, and they land in whatever order those
    /// writers get scheduled — but a reader may only see the run once *all* of it has
    /// landed, so the run is one unit here rather than `count` of them. That collapses
    /// the bookkeeping from a set entry per write to a single entry per group, which is
    /// the difference between a lock on the write path and a lock on the batch path.
    #[must_use = "the watermark cannot pass this run until it is finished"]
    pub fn begin_range(&self, base: SeqNo, count: u64) -> Pending {
        #[expect(clippy::expect_used)]
        let mut in_flight = self.0.in_flight.lock().expect("lock is poisoned");
        in_flight.taken.insert(base);
        drop(in_flight);

        Pending {
            gate: self.clone(),
            seqno: base,
            through: base + count.saturating_sub(1),
        }
    }

    /// **Recovery only.** Set the watermark with nothing in flight.
    pub fn restore(&self, seqno: SeqNo) {
        self.0.visible.fetch_max(seqno, AcqRel);
    }

    fn finish(&self, seqno: SeqNo, through: SeqNo) {
        #[expect(clippy::expect_used)]
        let mut in_flight = self.0.in_flight.lock().expect("lock is poisoned");

        in_flight.taken.remove(&seqno);
        in_flight.done = in_flight.done.max(through);

        // Everything below the lowest number still being applied has landed, so that is
        // how far a reader may see. With nothing in flight, everything finished has.
        let visible = in_flight
            .taken
            .first()
            .map_or(in_flight.done, |lowest| lowest.saturating_sub(1));

        // Under the same lock that decided the bound: two writers finishing at once must
        // not interleave into a watermark neither of them computed.
        self.0.visible.fetch_max(visible + 1, AcqRel);
    }
}

/// A sequence number taken and not yet finished — see [`VisibleSeqno`].
///
/// **Finishing on drop is deliberate.** A write that fails still has to give its number
/// back: one taken and never returned holds the watermark exactly where it is, forever,
/// and the database silently stops making new writes visible. Publishing a number
/// nothing was written at is harmless by comparison — there is nothing there to see —
/// so the safe default is the automatic one, and an error path cannot get it wrong by
/// returning early.
#[derive(Debug)]
pub struct Pending {
    gate: VisibleSeqno,
    seqno: SeqNo,
    /// The last sequence number of the run — equal to `seqno` for a lone write.
    through: SeqNo,
}

impl Pending {
    /// The first sequence number of the run this is holding.
    #[must_use]
    pub fn seqno(&self) -> SeqNo {
        self.seqno
    }

    /// The last sequence number of the run.
    #[must_use]
    pub fn through(&self) -> SeqNo {
        self.through
    }

    /// The rows are readable; let the watermark reach them when the queue allows.
    pub fn publish(self) {
        drop(self);
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        self.gate.finish(self.seqno, self.through);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_log::test;

    #[test]
    fn a_lone_write_publishes_as_soon_as_it_finishes() {
        let gate = VisibleSeqno::default();

        let write = gate.begin(5);
        assert_eq!(0, gate.get(), "nothing is visible until something finishes");

        write.publish();
        assert_eq!(6, gate.get());
    }

    #[test]
    fn a_write_still_applying_holds_the_watermark_below_itself() {
        let gate = VisibleSeqno::default();

        let slow = gate.begin(5);
        gate.begin(6).publish();

        assert!(
            gate.get() <= 5,
            "6 published past 5, which is still applying: a reader at {} is promised \
             rows that are not readable yet",
            gate.get()
        );

        slow.publish();
        assert_eq!(7, gate.get(), "with 5 landed, both are visible");
    }

    #[test]
    fn finishing_out_of_order_never_exposes_a_gap() {
        let gate = VisibleSeqno::default();

        let slow = gate.begin(5);
        let pending: Vec<_> = (6..=8).map(|seqno| gate.begin(seqno)).collect();

        for write in pending {
            write.publish();
            assert!(gate.get() <= 5, "the watermark reached {}", gate.get());
        }

        slow.publish();
        assert_eq!(9, gate.get(), "the whole run becomes visible at once");
    }

    /// **A write that fails still gives its number back.**
    ///
    /// The `Pending` is dropped rather than published, which is what an error path does
    /// by returning. Were that to leak the number, everything above it would be frozen.
    #[test]
    fn a_write_that_fails_does_not_freeze_the_watermark() {
        let gate = VisibleSeqno::default();

        let failed = gate.begin(5);
        let ok = gate.begin(6);

        drop(failed);
        ok.publish();

        assert_eq!(7, gate.get(), "a dropped write held the watermark");
    }

    /// **A group is one unit.** The run is invisible until all of it has landed, and
    /// then visible all at once — which is what lets a group commit hand out sequence
    /// numbers in order and let its members finish in any order at all.
    #[test]
    fn a_run_becomes_visible_as_a_whole_or_not_at_all() {
        let gate = VisibleSeqno::default();

        let first = gate.begin_range(5, 6); // 5..=10
        let second = gate.begin_range(11, 5); // 11..=15

        second.publish();
        assert!(
            gate.get() <= 5,
            "the second group published over the first, at {}",
            gate.get()
        );

        first.publish();
        assert_eq!(16, gate.get(), "both runs become visible together");
    }

    #[test]
    fn the_watermark_only_ever_moves_forward() {
        let gate = VisibleSeqno::default();

        gate.begin(5).publish();
        assert_eq!(6, gate.get());

        // A straggler beneath the watermark must not drag a reader backwards.
        gate.begin(3).publish();
        assert_eq!(6, gate.get(), "the watermark went backwards");
    }

    /// **The invariant, over arbitrary interleavings.**
    ///
    /// A hand-written case covers the orderings somebody thought of. This walks a
    /// deterministic pseudo-random schedule and asserts the one rule after every step.
    #[test]
    fn no_schedule_of_writers_exposes_a_write_that_has_not_landed() {
        let gate = VisibleSeqno::default();

        let mut next = 1u64;
        let mut outstanding: Vec<Pending> = Vec::new();
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mut finished = 0u64;

        for _ in 0..20_000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;

            if outstanding.is_empty() || rng % 3 != 0 {
                outstanding.push(gate.begin(next));
                next += 1;
            } else {
                let at = (rng >> 8) as usize % outstanding.len();
                outstanding.swap_remove(at).publish();
                finished += 1;
            }

            // A reader may see strictly below the lowest number still being applied.
            if let Some(lowest) = outstanding.iter().map(Pending::seqno).min() {
                assert!(
                    gate.get() <= lowest,
                    "watermark {} reaches {lowest}, which has not landed",
                    gate.get(),
                );
            }
        }

        assert!(finished > 1_000, "the schedule barely finished anything: {finished}");
    }
}
