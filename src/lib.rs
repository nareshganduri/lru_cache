//! A least recently used (LRU) cache implementation.
//!
//! This module provides a `Cache` struct that implements a least recently used (LRU)
//! caching strategy. It combines a hash table for efficient key-value lookups with
//! a doubly linked list to maintain the order of item access.
//!
//! When the cache reaches its maximum capacity, inserting a new item will
//! automatically evict the least recently used item to make space.
//!
//! # Features
//!
//! * **LRU Eviction**: Automatically removes the least recently used item when capacity is reached.
//! * **Efficient Lookups**: Uses a hash table for $O(1)$ average-case time complexity for `get`, `insert`, `remove`, and `contains` operations.
//! * **Customizable Hashing**: Supports custom hash builders for different hashing needs.
//! * **Doubly Linked List**: Maintains item order for efficient LRU tracking.
//!
//! # Examples
//!
//! ```
//! use std::num::NonZeroUsize;
//! use lru_cache::Cache;
//!
//! // Create a new cache with a capacity of 2
//! let mut cache = Cache::new(NonZeroUsize::new(2).unwrap());
//!
//! // Insert some key-value pairs
//! cache.insert("apple", 1);
//! cache.insert("banana", 2);
//!
//! // "banana" is now the most recently used, "apple" is the least recently used
//! assert_eq!(cache.keys().copied().collect::<Vec<_>>(), vec!["banana", "apple"]);
//!
//! // Accessing "apple" makes it the most recently used
//! assert_eq!(cache.get("apple"), Some(&1));
//! assert_eq!(cache.keys().copied().collect::<Vec<_>>(), vec!["apple", "banana"]);
//!
//! // Inserting a new item when the cache is full evicts the least recently used ("banana")
//! cache.insert("orange", 3);
//! assert_eq!(cache.len(), 2);
//! assert_eq!(cache.get("banana"), None); // "banana" has been evicted
//! assert_eq!(cache.keys().copied().collect::<Vec<_>>(), vec!["orange", "apple"]);
//!
//! // Update an existing item, it also moves to the front
//! assert_eq!(cache.insert("apple", 10), Some(1));
//! assert_eq!(cache.get("apple"), Some(&10));
//! assert_eq!(cache.keys().copied().collect::<Vec<_>>(), vec!["apple", "orange"]);
//!
//! // Remove an item
//! assert_eq!(cache.remove("orange"), Some(3));
//! assert_eq!(cache.len(), 1);
//! assert_eq!(cache.keys().copied().collect::<Vec<_>>(), vec!["apple"]);
//! assert_eq!(cache.get("orange"), None);
//! ```
#![warn(clippy::pedantic)]
#![warn(missing_docs)]

use std::{
    borrow::Borrow,
    hash::{BuildHasher, Hash, RandomState},
    num::NonZeroUsize,
};

enum Bucket<K, V> {
    Empty,
    Deleted,
    Occupied {
        key: K,
        value: V,
        next: Option<usize>,
        prev: Option<usize>,
    },
}

impl<K, V> Bucket<K, V> {
    fn set_next(&mut self, value: Option<usize>) {
        let Bucket::Occupied { next, .. } = self else {
            return;
        };
        *next = value;
    }

    fn set_prev(&mut self, value: Option<usize>) {
        let Bucket::Occupied { prev, .. } = self else {
            return;
        };
        *prev = value;
    }
}

/// A least recently used (LRU) cache implementation.
///
/// This cache uses a hash table for efficient key lookups and a doubly linked list
/// to maintain the order of items by their recency of use. When the cache reaches
/// its capacity, the least recently used item is evicted to make room for new insertions.
///
/// `K` is the type of the keys, `V` is the type of the values, and `S` is the type
/// of the hasher builder.
pub struct Cache<K, V, S = RandomState> {
    items: Box<[Bucket<K, V>]>,
    len: usize,
    head: Option<usize>,
    tail: Option<usize>,
    hash_builder: S,
}

impl<K, V> Cache<K, V> {
    /// Creates a new `Cache` with the specified `capacity`.
    ///
    /// The cache will use the default `RandomState` hasher.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The maximum number of key-value pairs the cache can hold.
    ///                Must be a non-zero value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use lru_cache::Cache;
    ///
    /// let mut cache: Cache<i32, String> = Cache::new(NonZeroUsize::new(10).unwrap());
    /// ```
    pub fn new(capacity: NonZeroUsize) -> Self {
        let capacity = capacity.get();
        Self {
            items: std::iter::repeat_with(|| Bucket::Empty)
                .take(capacity)
                .collect(),
            len: 0,
            head: None,
            tail: None,
            hash_builder: RandomState::new(),
        }
    }
}

impl<K, V, S> Cache<K, V, S> {
    /// Creates a new `Cache` with the specified `capacity` and a custom `hash_builder`.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The maximum number of key-value pairs the cache can hold.
    /// * `hash_builder` - A custom hasher builder to use for hashing keys.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::hash_map::RandomState;
    /// use lru_cache::Cache;
    ///
    /// let hasher = RandomState::new();
    /// let mut cache: Cache<String, i32> = Cache::with_hasher(5, hasher);
    /// ```
    pub fn with_hasher(capacity: usize, hash_builder: S) -> Self {
        Self {
            items: std::iter::repeat_with(|| Bucket::Empty)
                .take(capacity)
                .collect(),
            len: 0,
            head: None,
            tail: None,
            hash_builder,
        }
    }

    /// Returns `true` if the cache contains no key-value pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use lru_cache::Cache;
    ///
    /// let mut cache: Cache<i32, i32> = Cache::new(NonZeroUsize::new(5).unwrap());
    /// assert!(cache.is_empty());
    /// cache.insert(1, 10);
    /// assert!(!cache.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of key-value pairs currently in the cache.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use lru_cache::Cache;
    ///
    /// let mut cache: Cache<i32, i32> = Cache::new(NonZeroUsize::new(5).unwrap());
    /// assert_eq!(cache.len(), 0);
    /// cache.insert(1, 10);
    /// assert_eq!(cache.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the maximum number of key-value pairs the cache can hold.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use lru_cache::Cache;
    ///
    /// let cache: Cache<i32, i32> = Cache::new(NonZeroUsize::new(5).unwrap());
    /// assert_eq!(cache.capacity(), 5);
    /// ```
    pub fn capacity(&self) -> usize {
        self.items.len()
    }

    /// Iterates over all the keys in the cache, from most recently used to least recently used.
    ///
    /// # Returns
    ///
    /// A `Keys` iterator that yields references to the keys.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use lru_cache::Cache;
    ///
    /// let mut cache = Cache::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    /// cache.insert("c", 3);
    ///
    /// let keys: Vec<&str> = cache.keys().copied().collect();
    /// assert_eq!(keys, vec!["c", "b", "a"]);
    ///
    /// cache.get("a"); // Make 'a' most recently used
    /// let keys_after_get: Vec<&str> = cache.keys().copied().collect();
    /// assert_eq!(keys_after_get, vec!["a", "c", "b"]);
    /// ```
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys {
            index: self.head,
            items: &self.items,
        }
    }

    fn extract_node(&mut self, index: usize) {
        let &Bucket::Occupied { prev, next, .. } = &self.items[index] else {
            return;
        };

        // extract node from linked list
        if let Some(prev) = prev {
            self.items[prev].set_next(next);
        }
        if let Some(next) = next {
            self.items[next].set_prev(prev);
        }

        // adjust head/tail if extracted
        if self.head == Some(index) {
            self.head = next;
        }
        if self.tail == Some(index) {
            self.tail = prev;
        }
    }

    /// Moves the node at `index` to the head of the linked list
    fn move_node_to_head(&mut self, index: usize) {
        self.extract_node(index);

        // insert node at head
        self.items[index].set_prev(None);
        self.items[index].set_next(self.head);
        if let Some(head) = self.head {
            self.items[head].set_prev(Some(index));
        }
        self.head = Some(index);

        if self.len == 1 {
            self.tail = Some(index);
        }
    }

    /// Pops the tail of the linked list
    fn pop_tail_if_full(&mut self) -> Option<usize> {
        if self.len != self.capacity() {
            return None;
        }
        let Some(tail) = self.tail else {
            unreachable!("list should not be empty");
        };
        self.extract_node(tail);

        Some(tail)
    }
}

impl<K, V, S> Cache<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Returns `true` if the cache contains a value for the specified key.
    ///
    /// If the key is found, it is considered "accessed" and its position in the
    /// LRU order is updated, making it the most recently used item.
    ///
    /// # Arguments
    ///
    /// * `key` - A reference to the key to check.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use lru_cache::Cache;
    ///
    /// let mut cache = Cache::new(NonZeroUsize::new(2).unwrap());
    /// cache.insert("a", 1);
    /// assert!(cache.contains("a"));
    /// assert!(!cache.contains("b"));
    /// ```
    pub fn contains<Q>(&mut self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Finds the index of the entry holding `key` using linear probing.
    ///
    /// Returns `Ok(index)` if the key is found at `index`.
    /// Returns `Err(index)` if the key is not in the map, but can be inserted at `index`
    fn probe_for_index<Q>(&self, key: &Q) -> Result<usize, usize>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let capacity = self.capacity();
        let hash = self.hash_builder.hash_one(key);
        let mut index = (hash % capacity as u64) as usize;
        let mut tombstone = None;

        for _ in 0..capacity {
            let bucket = &self.items[index];
            match bucket {
                Bucket::Empty => {
                    break;
                }
                Bucket::Deleted => {
                    tombstone = Some(index);
                }
                Bucket::Occupied { key: curr, .. } => {
                    let curr: &Q = curr.borrow();
                    let hash2 = self.hash_builder.hash_one(curr);

                    if hash == hash2 && key == curr {
                        return Ok(index);
                    }
                }
            }

            index += 1;
            index %= capacity;
        }

        Err(tombstone.unwrap_or(index))
    }

    /// Returns a reference to the value associated with `key`, if it exists.
    ///
    /// If the key is found, it is considered "accessed" and its position in the
    /// LRU order is updated, making it the most recently used item.
    ///
    /// # Arguments
    ///
    /// * `key` - A reference to the key to retrieve.
    ///
    /// # Returns
    ///
    /// An `Option<&V>` which is `Some(&value)` if the key was found, or `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use lru_cache::Cache;
    ///
    /// let mut cache = Cache::new(NonZeroUsize::new(2).unwrap());
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    ///
    /// assert_eq!(cache.get("a"), Some(&1));
    /// assert_eq!(cache.get("c"), None);
    ///
    /// // Accessing "a" makes it the most recently used
    /// let keys: Vec<&str> = cache.keys().copied().collect();
    /// assert_eq!(keys, vec!["a", "b"]);
    /// ```
    pub fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let index = self.probe_for_index(key).ok()?;
        self.move_node_to_head(index);
        let Bucket::Occupied { value, .. } = &self.items[index] else {
            unreachable!("entry should be occupied")
        };
        Some(value)
    }

    /// Returns a mutable reference to the value associated with `key`, if it exists.
    ///
    /// If the key is found, it is considered "accessed" and its position in the
    /// LRU order is updated, making it the most recently used item.
    ///
    /// # Arguments
    ///
    /// * `key` - A reference to the key to retrieve.
    ///
    /// # Returns
    ///
    /// An `Option<&mut V>` which is `Some(&mut value)` if the key was found, or `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use lru_cache::Cache;
    ///
    /// let mut cache = Cache::new(NonZeroUsize::new(2).unwrap());
    /// cache.insert("a", 1);
    ///
    /// if let Some(val) = cache.get_mut("a") {
    ///     *val = 10;
    /// }
    /// assert_eq!(cache.get("a"), Some(&10));
    /// ```
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let index = self.probe_for_index(key).ok()?;
        self.move_node_to_head(index);
        let Bucket::Occupied { value, .. } = &mut self.items[index] else {
            unreachable!("entry should be occupied")
        };
        Some(value)
    }

    /// Removes a key-value pair from the cache.
    ///
    /// If the key is found and removed, its associated value is returned.
    /// The cache's length is decreased.
    ///
    /// # Arguments
    ///
    /// * `key` - A reference to the key to remove.
    ///
    /// # Returns
    ///
    /// An `Option<V>` which is `Some(value)` if the key was found and removed, or `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use lru_cache::Cache;
    ///
    /// let mut cache = Cache::new(NonZeroUsize::new(2).unwrap());
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    ///
    /// assert_eq!(cache.remove("a"), Some(1));
    /// assert_eq!(cache.len(), 1);
    /// assert_eq!(cache.get("a"), None);
    /// assert_eq!(cache.remove("c"), None);
    /// ```
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.probe_for_index(key).ok()?;
        self.extract_node(index);
        self.len -= 1;
        let old_bucket = std::mem::replace(&mut self.items[index], Bucket::Deleted);
        let Bucket::Occupied { value, .. } = old_bucket else {
            unreachable!("bucket should have been occupied")
        };
        Some(value)
    }

    /// Inserts a key-value pair into the cache.
    ///
    /// If the key already exists in the cache, its value is updated, and the item
    /// is moved to the head (most recently used) of the LRU list. The old value
    /// associated with the key is returned.
    ///
    /// If the key does not exist and the cache is at full capacity, the least
    /// recently used item is evicted to make space for the new item. The new item
    /// is then inserted at the head of the LRU list.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert.
    /// * `value` - The value to associate with the key.
    ///
    /// # Returns
    ///
    /// An `Option<V>` which is `Some(old_value)` if the key was already present,
    /// or `None` if it was a new insertion.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use lru_cache::Cache;
    ///
    /// let mut cache = Cache::new(NonZeroUsize::new(2).unwrap());
    ///
    /// assert_eq!(cache.insert("a", 1), None); // New insertion
    /// assert_eq!(cache.len(), 1);
    /// assert_eq!(cache.keys().copied().collect::<Vec<_>>(), vec!["a"]);
    ///
    /// assert_eq!(cache.insert("b", 2), None); // New insertion
    /// assert_eq!(cache.len(), 2);
    /// assert_eq!(cache.keys().copied().collect::<Vec<_>>(), vec!["b", "a"]);
    ///
    /// assert_eq!(cache.insert("c", 3), None); // Cache is full, "a" is evicted
    /// assert_eq!(cache.len(), 2);
    /// assert_eq!(cache.get("a"), None);
    /// assert_eq!(cache.keys().copied().collect::<Vec<_>>(), vec!["c", "b"]);
    ///
    /// assert_eq!(cache.insert("b", 20), Some(2)); // Update existing key, "b" moves to head
    /// assert_eq!(cache.len(), 2);
    /// assert_eq!(cache.get("b"), Some(&20));
    /// assert_eq!(cache.keys().copied().collect::<Vec<_>>(), vec!["b", "c"]);
    /// ```
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self.probe_for_index(&key) {
            Ok(index) => {
                self.move_node_to_head(index);
                let Bucket::Occupied {
                    value: old_value, ..
                } = &mut self.items[index]
                else {
                    unreachable!("entry should be occupied")
                };
                let old_value = std::mem::replace(old_value, value);
                Some(old_value)
            }
            Err(index) => {
                let tail = self.pop_tail_if_full();
                let index = tail.unwrap_or(index);
                if tail.is_none() {
                    self.len += 1;
                }
                self.items[index] = Bucket::Occupied {
                    key,
                    value,
                    next: None,
                    prev: None,
                };
                self.move_node_to_head(index);

                None
            }
        }
    }
}

/// An iterator over the keys in an LRU `Cache`.
///
/// Keys are yielded in order from most recently used to least recently used.
///
/// This `struct` is created by the [`keys`] method on [`Cache`]. See its
/// documentation for more.
///
/// [`keys`]: struct.Cache.html#method.keys
/// [`Cache`]: struct.Cache.html
pub struct Keys<'a, K, V> {
    index: Option<usize>,
    items: &'a [Bucket<K, V>],
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index?;
        let &Bucket::Occupied { ref key, next, .. } = &self.items[index] else {
            unreachable!("bucket must be occupied");
        };
        self.index = next;
        Some(key)
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZero;

    use super::*;

    #[test]
    fn test_cache() {
        let mut cache = Cache::new(NonZero::new(2).unwrap());
        let mut keys: Vec<_> = cache.keys().copied().collect();
        assert_eq!(keys, vec![]);
        cache.insert(1, 1);
        keys = cache.keys().copied().collect();
        assert_eq!(keys, vec![1]);
        cache.insert(2, 2);
        keys = cache.keys().copied().collect();
        assert_eq!(keys, vec![2, 1]);
        assert_eq!(cache.get(&1), Some(&1));
        keys = cache.keys().copied().collect();
        assert_eq!(keys, vec![1, 2]);
        cache.insert(3, 3);
        keys = cache.keys().copied().collect();
        assert_eq!(keys, vec![3, 1]);
        assert_eq!(cache.get(&2), None);
        keys = cache.keys().copied().collect();
        assert_eq!(keys, vec![3, 1]);
        cache.insert(4, 4);
        keys = cache.keys().copied().collect();
        assert_eq!(keys, vec![4, 3]);
        assert_eq!(cache.get(&1), None);
        keys = cache.keys().copied().collect();
        assert_eq!(keys, vec![4, 3]);
        assert_eq!(cache.get(&3), Some(&3));
        keys = cache.keys().copied().collect();
        assert_eq!(keys, vec![3, 4]);
        assert_eq!(cache.get(&4), Some(&4));
        keys = cache.keys().copied().collect();
        assert_eq!(keys, vec![4, 3]);
    }

    #[test]
    fn test_cache_with_capacity_one() {
        let mut cache = Cache::new(NonZero::new(1).unwrap());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity(), 1);

        cache.insert(1, 10);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&1), Some(&10));
        let mut keys: Vec<_> = cache.keys().copied().collect();
        assert_eq!(keys, vec![1]);

        cache.insert(2, 20); // This should evict (1, 10)
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&2), Some(&20));
        keys = cache.keys().copied().collect();
        assert_eq!(keys, vec![2]);

        // Re-insert 1, should evict 2
        cache.insert(1, 100);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&2), None);
        assert_eq!(cache.get(&1), Some(&100));
    }

    #[test]
    fn test_cache_get_mut() {
        let mut cache = Cache::new(NonZero::new(2).unwrap());
        cache.insert("a", 1);
        cache.insert("b", 2);

        // Get mutable reference and modify
        if let Some(val) = cache.get_mut("a") {
            *val = 10;
        }
        assert_eq!(cache.get("a"), Some(&10));
        // Ensure "a" is now the most recently used
        let keys: Vec<_> = cache.keys().copied().collect();
        assert_eq!(keys, vec!["a", "b"]);

        // Try to get_mut for a non-existent key
        assert_eq!(cache.get_mut("c"), None);
    }

    #[test]
    fn test_cache_remove() {
        let mut cache = Cache::new(NonZero::new(3).unwrap());
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);

        assert_eq!(cache.len(), 3);
        let mut keys: Vec<_> = cache.keys().copied().collect();
        assert_eq!(keys, vec!["c", "b", "a"]);

        // Remove a middle element
        assert_eq!(cache.remove("b"), Some(2));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get("b"), None);
        keys = cache.keys().copied().collect();
        assert_eq!(keys, vec!["c", "a"]);

        // Remove the head
        assert_eq!(cache.remove("c"), Some(3));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get("c"), None);
        keys = cache.keys().copied().collect();
        assert_eq!(keys, vec!["a"]);

        // Remove the tail (now the only element)
        assert_eq!(cache.remove("a"), Some(1));
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.get("a"), None);
        keys = cache.keys().copied().collect();
        assert!(keys.is_empty());

        // Try to remove a non-existent element
        assert_eq!(cache.remove("d"), None);
        assert_eq!(cache.len(), 0); // Length should remain 0
    }

    #[test]
    fn test_cache_contains() {
        let mut cache = Cache::new(NonZero::new(2).unwrap());
        cache.insert("key1", "value1");
        cache.insert("key2", "value2");

        assert!(cache.contains("key1"));
        assert!(cache.contains("key2"));
        assert!(!cache.contains("key3"));

        // Accessing a key with contains should make it recently used
        cache.contains("key1");
        let keys: Vec<_> = cache.keys().copied().collect();
        assert_eq!(keys, vec!["key1", "key2"]);
    }

    #[test]
    fn test_insert_existing_key_updates_value_and_position() {
        let mut cache = Cache::new(NonZero::new(3).unwrap());
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);
        let mut keys: Vec<_> = cache.keys().copied().collect();
        assert_eq!(keys, vec!["c", "b", "a"]);

        // Insert "b" again with a new value
        assert_eq!(cache.insert("b", 20), Some(2)); // Should return the old value
        assert_eq!(cache.len(), 3); // Length should not change
        assert_eq!(cache.get("b"), Some(&20));
        // "b" should now be the most recently used
        keys = cache.keys().copied().collect();
        assert_eq!(keys, vec!["b", "c", "a"]);

        // Insert "a" again, it was the least recently used
        assert_eq!(cache.insert("a", 10), Some(1));
        keys = cache.keys().copied().collect();
        assert_eq!(keys, vec!["a", "b", "c"]);
    }
}
