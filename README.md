# LRU Cache

A simple implementation of an LRU (least recently used) cache in 100% safe Rust.

## Why?

As it turns out, an LRU cache isn't the most trivial thing to implement in safe Rust.

The standard implementation uses a hashmap to store the key-value pairs and a doubly linked list to manage the access order. This ensures $O(1)$ time
for accessing and updating keys, but it doesn't play nice with Rust's ownership rules, which requires every object to have at most one owner.

Doubly-linked lists have bidirectional edges between each node in the list, so there's no clear parent-child relationship. Similarly, in order to
move recently accessed keys to the front of the list *and* delete the least recently used keys (which are at the back of the list), every entry in the 
hashmap needs a bidirectional link to its corresponding node in the list.

Basically, any time you have an ownership graph that isn't fundamentally tree-like, Rust is going to complain. Now, there are two standard ways to get around this:
1. Use `Rc<RefCell>` for everything
2. Use unsafe code

Neither are really satisfactory for our purposes. `Rc` feels a bit like a hack - why box things unless we absolutely have to? And `RefCell`s litter our code with
a bunch of `.borrow()`s and `.borrow_mut()`s and defer a bunch of nice compile-time safety checks to runtime.
Likewise, unsafe code is notoriously hard to get right, so we want to avoid it unless it's absolutely necessary.

Well luckily, it turns out you don't really need either of those things. There's a classic workaround in Rust for working with complicated graph-like structures, 
and this library is meant to be a proof of concept that we can take advantage of that to write an LRU cache - and, as a nice little bonus, the resulting code 
is actually fairly simple.

## Indices as pointers
The workaround is a classic trick in Rust - put all your items in a `Vec` and use indices into that `Vec` like pointers. By doing so, you don't have to deal
with unsafe code or complicated lifetimes or refcounting or any other hack people like to resort to when writing non-trivial Rust.

Obviously, there are downsides to doing this - if you need to add items from your graph, you could potentially reallocate the whole `Vec` - which, depending on
how big your graph is, could be a lot more expensive than just boxing all your nodes and having them scattered all over the heap.

Similarly, if you want to delete nodes, a standard `Vec` and some `usize`s for indices won't suffice - look into "generational arenas" for more on that.

Luckily, for our purposes, the cache's capacity is fixed at initialization, so we can't pay any performance costs for reallocations that will never happen!
In fact, this implementation uses a boxed slice to enforce that.

On top of that, we don't hand out indices into our entry list for our key-value pairs (because that information isn't useful to the end user, who isn't expecting
anything outside of a standard hashmap API), so there's no "use-after-free" or ABA-style bugs lurking that we'd need a generational arena to fix.

## So should I actually use this?
Probably not - it's not published on [crates.io](https://crates.io/) for a reason. It's only meant to be an example of how you can use the "indices-as-pointers" trick to get around
Rust complexities in an area you might not expect it to work. It remains to be seen whether there's actually a performance benefit under realistic workloads
to doing things like this.

On the other hand, this is a fairly common LeetCode [problem](https://leetcode.com/problems/lru-cache/) that a lot of people try. If you're interested in solving that problem
using Rust, maybe there's some useful tricks in here you can steal...
