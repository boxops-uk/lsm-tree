// Copyright (c) 2024-present, fjall-rs
// This source code is licensed under both the Apache 2.0 and MIT License
// (found in the LICENSE-* files in the repository)

use crate::key::InternalKey;
use crate::{
    value::{InternalValue, SeqNo, UserValue},
    ValueType,
};
use crossbeam_skiplist::SkipMap;
use std::ops::RangeBounds;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

/// A memtable's key filter, in 64-byte blocks — 1 MiB, about eight bits a key for a
/// default 64 MiB memtable of small rows.
///
/// A guess, and the thing to fix before this is proposed anywhere: the memtable is
/// bounded in *bytes* and a filter wants a count, so the honest version derives this
/// from `max_memtable_size` rather than assuming a row size.
const FILTER_BLOCKS: usize = 16 * 1_024;
/// One cache line, as [`u64`]s.
const BLOCK_WORDS: usize = 8;
const FILTER_PROBES: u32 = 4;

/// **A blocked Bloom filter over the user keys a memtable holds.**
///
/// Every sealed table already has a filter, and the memtable — the one part of the tree
/// searched on *every* point read — does not. A read that misses therefore pays a full
/// skiplist descent, comparing a key at each level, before it can conclude the key is
/// not there. For a write path that interns, that is the common case by a long way:
/// `resolve_or_create` asks "is this key present?" once per fact, and during a bulk
/// load the answer is always no.
///
/// **Blocked, and that is not an optimisation detail.** A classic Bloom filter puts its
/// `k` bits anywhere in the array, so each probe is its own cache miss and the write
/// path touches four lines per row. Measured that way the reads got cheaper and the
/// *writes* got dearer by more — 1 MiB of scattered `fetch_or` evicts the skiplist's
/// own working set. Confining a key's bits to one 64-byte block costs a little accuracy
/// and touches one line.
///
/// Lock-free, because the skiplist it guards is.
///
/// **Why relaxed is enough.** A false positive costs a skiplist descent and nothing
/// else. A false *negative* would be a lost read, and cannot happen for any write the
/// reader is entitled to see: the filter is set before the skiplist insert in program
/// order, and a reader only knows to look for a write once it has synchronised with the
/// sequence number that published it — which orders it after both. A reader racing an
/// unpublished write may see either answer, which is what racing means.
pub struct KeyFilter {
    /// `FILTER_BLOCKS` blocks of `BLOCK_WORDS` words, laid out flat.
    bits: Box<[AtomicU64]>,
}

impl KeyFilter {
    fn new() -> Self {
        Self {
            bits: (0..FILTER_BLOCKS * BLOCK_WORDS)
                .map(|_| AtomicU64::new(0))
                .collect(),
        }
    }

    /// The block a key lives in, and the `BLOCK_WORDS` masks its probes set there.
    ///
    /// One hash for everything: the high bits choose the block, the low bits walk the
    /// probes within it (Kirsch-Mitzenmacher), so `k` probes cost one hash of the key.
    fn probes(key: &[u8]) -> (usize, [u64; BLOCK_WORDS]) {
        let hash = crate::hash::hash64(key);
        let block = ((hash >> 32) as usize) % FILTER_BLOCKS;

        let (h1, h2) = (hash as u32 as u64, ((hash >> 16) as u32 as u64) | 1);
        let mut masks = [0u64; BLOCK_WORDS];

        for i in 0..u64::from(FILTER_PROBES) {
            // 512 bits to a block.
            let bit = (h1.wrapping_add(i.wrapping_mul(h2)) % 512) as usize;
            masks[bit / 64] |= 1 << (bit % 64);
        }

        (block * BLOCK_WORDS, masks)
    }

    fn set(&self, key: &[u8]) {
        let (at, masks) = Self::probes(key);

        for (word, mask) in masks.iter().enumerate() {
            if *mask != 0 {
                self.bits[at + word].fetch_or(*mask, Relaxed);
            }
        }
    }

    /// `false` is authoritative: the key was never inserted.
    fn might_hold(&self, key: &[u8]) -> bool {
        let (at, masks) = Self::probes(key);

        masks.iter().enumerate().all(|(word, mask)| {
            *mask == 0 || self.bits[at + word].load(Relaxed) & *mask == *mask
        })
    }
}

pub use crate::tree::inner::MemtableId;

/// The memtable serves as an intermediary, ephemeral, sorted storage for new items
///
/// When the Memtable exceeds some size, it should be flushed to a table.
pub struct Memtable {
    #[doc(hidden)]
    pub id: MemtableId,

    /// The actual content, stored in a lock-free skiplist.
    #[doc(hidden)]
    pub items: SkipMap<InternalKey, UserValue>,

    /// Approximate active memtable size.
    ///
    /// If this grows too large, a flush is triggered.
    pub(crate) approximate_size: AtomicU64,

    /// Highest encountered sequence number.
    ///
    /// This is used so that `get_highest_seqno` has O(1) complexity.
    pub(crate) highest_seqno: AtomicU64,

    pub(crate) requested_rotation: AtomicBool,

    /// See [`KeyFilter`], and `Config::memtable_filter` for why it is a choice.
    ///
    /// `None` unless the tree asked for it: **a filter is only worth its write cost to
    /// a tree whose point reads miss.** One read by a key that is usually present pays
    /// the set on every insert and collects nothing, because a hit has to do the
    /// skiplist descent regardless.
    filter: Option<KeyFilter>,
}

impl Memtable {
    /// Returns the memtable ID.
    pub fn id(&self) -> MemtableId {
        self.id
    }

    /// Returns `true` if the memtable was already flagged for rotation.
    pub fn is_flagged_for_rotation(&self) -> bool {
        self.requested_rotation
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Flags the memtable as requested for rotation.
    pub fn flag_rotated(&self) {
        self.requested_rotation
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    #[doc(hidden)]
    #[must_use]
    pub fn new(id: MemtableId, filtered: bool) -> Self {
        Self {
            id,
            items: SkipMap::default(),
            approximate_size: AtomicU64::default(),
            highest_seqno: AtomicU64::default(),
            requested_rotation: AtomicBool::default(),
            filter: filtered.then(KeyFilter::new),
        }
    }

    /// Creates an iterator over all items.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = InternalValue> + '_ {
        self.items.iter().map(|entry| InternalValue {
            key: entry.key().clone(),
            value: entry.value().clone(),
        })
    }

    /// Creates an iterator over a range of items.
    pub(crate) fn range<'a, R: RangeBounds<InternalKey> + 'a>(
        &'a self,
        range: R,
    ) -> impl DoubleEndedIterator<Item = InternalValue> + 'a {
        self.items.range(range).map(|entry| InternalValue {
            key: entry.key().clone(),
            value: entry.value().clone(),
        })
    }

    /// Returns the item by key if it exists.
    ///
    /// The item with the highest seqno will be returned, if `seqno` is None.
    #[doc(hidden)]
    pub fn get(&self, key: &[u8], seqno: SeqNo) -> Option<InternalValue> {
        if seqno == 0 {
            return None;
        }

        // NOTE: This range start deserves some explanation...
        // InternalKeys are multi-sorted by 2 categories: user_key and Reverse(seqno). (tombstone doesn't really matter)
        // We search for the lowest entry that is greater or equal the user's prefix key
        // and has the seqno (or lower) we want (because the seqno is stored in reverse order)
        //
        // Example: We search for "abc"
        //
        // key -> seqno
        //
        // a   -> 7
        // abc -> 5 <<< This is the lowest key (highest seqno) that matches the key with seqno=None
        // abc -> 4
        // abc -> 3 <<< If searching for abc and seqno=4, we would get this
        // abcdef -> 6
        // abcdef -> 5
        //
        // **The cheap no.** Every sealed table gets to answer this from a filter; the
        // memtable had to descend a skiplist to say the same thing.
        if let Some(filter) = &self.filter {
            if !filter.might_hold(key) {
                return None;
            }
        }

        let lower_bound = InternalKey::new(key, seqno - 1, ValueType::Value);

        let mut iter = self
            .items
            .range(lower_bound..)
            .take_while(|entry| &*entry.key().user_key == key);

        iter.next().map(|entry| InternalValue {
            key: entry.key().clone(),
            value: entry.value().clone(),
        })
    }

    /// Gets approximate size of memtable in bytes.
    pub fn size(&self) -> u64 {
        self.approximate_size
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Counts the number of items in the memtable.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if the memtable is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Inserts an item into the memtable
    #[doc(hidden)]
    pub fn insert(&self, item: InternalValue) -> (u64, u64) {
        #[expect(
            clippy::expect_used,
            reason = "keys are limited to 16-bit length + values are limited to 32-bit length"
        )]
        let item_size =
            (item.key.user_key.len() + item.value.len() + std::mem::size_of::<InternalValue>())
                .try_into()
                .expect("should fit into u64");

        let size_before = self
            .approximate_size
            .fetch_add(item_size, std::sync::atomic::Ordering::AcqRel);

        let key = InternalKey::new(item.key.user_key, item.key.seqno, item.key.value_type);

        // **Before the insert, not after.** A reader that finds the key in the skiplist
        // must never have been told the filter does not hold it. A tombstone is an
        // insert like any other, so removal needs nothing here — the filter answers
        // "this memtable has an entry for this key", not "this key exists".
        if let Some(filter) = &self.filter {
            filter.set(&key.user_key);
        }

        self.items.insert(key, item.value);

        self.highest_seqno
            .fetch_max(item.key.seqno, std::sync::atomic::Ordering::AcqRel);

        (item_size, size_before + item_size)
    }

    /// Returns the highest sequence number in the memtable.
    pub fn get_highest_seqno(&self) -> Option<SeqNo> {
        if self.is_empty() {
            None
        } else {
            Some(
                self.highest_seqno
                    .load(std::sync::atomic::Ordering::Acquire),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ValueType;
    use test_log::test;

    #[test]
    #[expect(clippy::unwrap_used)]
    fn memtable_mvcc_point_read() {
        let memtable = Memtable::new(0, true);

        memtable.insert(InternalValue::from_components(
            *b"hello-key-999991",
            *b"hello-value-999991",
            0,
            ValueType::Value,
        ));

        let item = memtable.get(b"hello-key-99999", SeqNo::MAX);
        assert_eq!(None, item);

        let item = memtable.get(b"hello-key-999991", SeqNo::MAX);
        assert_eq!(*b"hello-value-999991", &*item.unwrap().value);

        memtable.insert(InternalValue::from_components(
            *b"hello-key-999991",
            *b"hello-value-999991-2",
            1,
            ValueType::Value,
        ));

        let item = memtable.get(b"hello-key-99999", SeqNo::MAX);
        assert_eq!(None, item);

        let item = memtable.get(b"hello-key-999991", SeqNo::MAX);
        assert_eq!((*b"hello-value-999991-2"), &*item.unwrap().value);

        let item = memtable.get(b"hello-key-99999", 1);
        assert_eq!(None, item);

        let item = memtable.get(b"hello-key-999991", 1);
        assert_eq!((*b"hello-value-999991"), &*item.unwrap().value);

        let item = memtable.get(b"hello-key-99999", 2);
        assert_eq!(None, item);

        let item = memtable.get(b"hello-key-999991", 2);
        assert_eq!((*b"hello-value-999991-2"), &*item.unwrap().value);
    }

    #[test]
    fn memtable_get() {
        let memtable = Memtable::new(0, true);

        let value =
            InternalValue::from_components(b"abc".to_vec(), b"abc".to_vec(), 0, ValueType::Value);

        memtable.insert(value.clone());

        assert_eq!(Some(value), memtable.get(b"abc", SeqNo::MAX));
    }

    #[test]
    fn memtable_get_highest_seqno() {
        let memtable = Memtable::new(0, true);

        memtable.insert(InternalValue::from_components(
            b"abc".to_vec(),
            b"abc".to_vec(),
            0,
            ValueType::Value,
        ));
        memtable.insert(InternalValue::from_components(
            b"abc".to_vec(),
            b"abc".to_vec(),
            1,
            ValueType::Value,
        ));
        memtable.insert(InternalValue::from_components(
            b"abc".to_vec(),
            b"abc".to_vec(),
            2,
            ValueType::Value,
        ));
        memtable.insert(InternalValue::from_components(
            b"abc".to_vec(),
            b"abc".to_vec(),
            3,
            ValueType::Value,
        ));
        memtable.insert(InternalValue::from_components(
            b"abc".to_vec(),
            b"abc".to_vec(),
            4,
            ValueType::Value,
        ));

        assert_eq!(
            Some(InternalValue::from_components(
                b"abc".to_vec(),
                b"abc".to_vec(),
                4,
                ValueType::Value,
            )),
            memtable.get(b"abc", SeqNo::MAX)
        );
    }

    #[test]
    fn memtable_get_prefix() {
        let memtable = Memtable::new(0, true);

        memtable.insert(InternalValue::from_components(
            b"abc0".to_vec(),
            b"abc".to_vec(),
            0,
            ValueType::Value,
        ));
        memtable.insert(InternalValue::from_components(
            b"abc".to_vec(),
            b"abc".to_vec(),
            255,
            ValueType::Value,
        ));

        assert_eq!(
            Some(InternalValue::from_components(
                b"abc".to_vec(),
                b"abc".to_vec(),
                255,
                ValueType::Value,
            )),
            memtable.get(b"abc", SeqNo::MAX)
        );

        assert_eq!(
            Some(InternalValue::from_components(
                b"abc0".to_vec(),
                b"abc".to_vec(),
                0,
                ValueType::Value,
            )),
            memtable.get(b"abc0", SeqNo::MAX)
        );
    }

    #[test]
    fn memtable_get_old_version() {
        let memtable = Memtable::new(0, true);

        memtable.insert(InternalValue::from_components(
            b"abc".to_vec(),
            b"abc".to_vec(),
            0,
            ValueType::Value,
        ));
        memtable.insert(InternalValue::from_components(
            b"abc".to_vec(),
            b"abc".to_vec(),
            99,
            ValueType::Value,
        ));
        memtable.insert(InternalValue::from_components(
            b"abc".to_vec(),
            b"abc".to_vec(),
            255,
            ValueType::Value,
        ));

        assert_eq!(
            Some(InternalValue::from_components(
                b"abc".to_vec(),
                b"abc".to_vec(),
                255,
                ValueType::Value,
            )),
            memtable.get(b"abc", SeqNo::MAX)
        );

        assert_eq!(
            Some(InternalValue::from_components(
                b"abc".to_vec(),
                b"abc".to_vec(),
                99,
                ValueType::Value,
            )),
            memtable.get(b"abc", 100)
        );

        assert_eq!(
            Some(InternalValue::from_components(
                b"abc".to_vec(),
                b"abc".to_vec(),
                0,
                ValueType::Value,
            )),
            memtable.get(b"abc", 50)
        );
    }
}
