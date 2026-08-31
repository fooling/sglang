"""HiRadixCache eviction releases KV pages via free_segment, not torch.unique.

Eviction pops leaves one at a time, so while this path used the plain free()
each evicted leaf cost a torch.unique whose data-dependent output shape forces a
device sync inside the scheduler step. Converting it is only correct because a
tree node's value is a page-exact segment -- page-aligned start, page-multiple
length, and owned by that node alone. These tests pin exactly that: the freed
page set must equal the torch.unique reference, and no page may be released
twice across an eviction round.

A real HiRadixCache needs a pinned host pool (CUDA); this stays on CPU and
drives the eviction helpers against a real paged allocator instead.

    python -m pytest test/registered/unit/mem_cache/test_hiradix_evict_free_segment.py
"""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PAGE_SIZE = 4
NUM_PAGES = 64


def _make_allocator(need_sort: bool) -> PagedTokenToKVPoolAllocator:
    alloc = PagedTokenToKVPoolAllocator(
        size=NUM_PAGES * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )
    # Arms free_segment's in-tree oracle (stride reps == unique) plus the
    # cross-call no-duplicate-pages check.
    alloc.debug_mode = True
    return alloc


class _FreeWatch:
    """Records which pages an allocator gained, whichever container they land
    in (need_sort routes releases into release_pages)."""

    def __init__(self, alloc):
        self.alloc = alloc
        self.free_before = len(alloc.free_pages)
        self.release_before = len(alloc.release_pages)

    def freed(self) -> torch.Tensor:
        # _release_page_ids prepends, so the new page ids are the head slice.
        alloc = self.alloc
        new_free = alloc.free_pages[: len(alloc.free_pages) - self.free_before]
        new_release = alloc.release_pages[
            : len(alloc.release_pages) - self.release_before
        ]
        return torch.sort(torch.cat((new_free, new_release)))[0]


class _EvictHarness(HiRadixCache):
    """Stand-in carrying only the state the eviction helpers touch, so the real
    _evict_regular / _drop_subtree_no_host bodies run unmodified."""

    def __init__(self, allocator):
        self.page_size = PAGE_SIZE
        self.enable_kv_cache_events = False
        self.evicted_host = []
        self.cache_controller = SimpleNamespace(
            mem_pool_device_allocator=allocator,
            evict_host=self.evicted_host.append,
        )
        self.evictable_size_ = 0
        self.evictable_leaves = set()
        self.evictable_host_leaves = set()
        self.ongoing_write_through = {}
        self.metrics_collector = None
        self.root_node = TreeNode()
        self.root_node.key = RadixKey(array("q", []))
        self.root_node.value = None

    def add_leaf(self, allocator, num_pages: int, token_offset: int) -> TreeNode:
        """A leaf holding a page-exact value, as insert() would produce."""
        num_tokens = num_pages * PAGE_SIZE
        value = allocator.alloc(num_tokens)
        assert value is not None
        node = TreeNode()
        node.parent = self.root_node
        node.key = RadixKey(
            array("q", list(range(token_offset, token_offset + num_tokens)))
        )
        node.value = value
        self.root_node.children[node.key.child_key(PAGE_SIZE)] = node
        self.evictable_size_ += num_tokens
        self.evictable_leaves.add(node)
        return node


def _unique_pages(value: torch.Tensor) -> torch.Tensor:
    return torch.unique(value // PAGE_SIZE)


class TestHiRadixEvictFreeSegment(unittest.TestCase):
    def test_evict_regular_frees_the_unique_page_set(self):
        # need_sort=True is the PD-disaggregation prefill arm: reps are routed
        # into release_pages instead of free_pages.
        for need_sort in (False, True):
            for num_pages in (1, 2, 5):
                with self.subTest(need_sort=need_sort, num_pages=num_pages):
                    alloc = _make_allocator(need_sort)
                    cache = _EvictHarness(alloc)
                    node = cache.add_leaf(alloc, num_pages, token_offset=0)
                    expected = _unique_pages(node.value)
                    watch = _FreeWatch(alloc)

                    freed_tokens = cache._evict_regular(node)

                    self.assertEqual(freed_tokens, num_pages * PAGE_SIZE)
                    self.assertTrue(torch.equal(watch.freed(), expected))
                    self.assertEqual(cache.evictable_size_, 0)

    def test_evict_device_frees_the_unique_page_set(self):
        for need_sort in (False, True):
            with self.subTest(need_sort=need_sort):
                alloc = _make_allocator(need_sort)
                controller = SimpleNamespace(mem_pool_device_allocator=alloc)
                indices = alloc.alloc(3 * PAGE_SIZE)
                expected = _unique_pages(indices)
                watch = _FreeWatch(alloc)

                freed = HiCacheController.evict_device(controller, indices)

                self.assertEqual(freed, 3 * PAGE_SIZE)
                self.assertTrue(torch.equal(watch.freed(), expected))

    def test_drop_subtree_frees_every_node_once(self):
        alloc = _make_allocator(need_sort=False)
        cache = _EvictHarness(alloc)
        root = cache.add_leaf(alloc, num_pages=2, token_offset=0)
        child = cache.add_leaf(alloc, num_pages=3, token_offset=100)
        # re-parent: add_leaf hangs everything off root_node
        cache.root_node.children.pop(child.key.child_key(PAGE_SIZE))
        child.parent = root
        root.children[child.key.child_key(PAGE_SIZE)] = child
        cache.evictable_leaves.discard(root)
        expected = torch.sort(
            torch.cat((_unique_pages(root.value), _unique_pages(child.value)))
        )[0]
        watch = _FreeWatch(alloc)

        freed_tokens = cache._drop_subtree_no_host(root)

        self.assertEqual(freed_tokens, 5 * PAGE_SIZE)
        self.assertTrue(torch.equal(watch.freed(), expected))
        self.assertEqual(cache.evictable_size_, 0)

    def test_eviction_round_releases_each_page_exactly_once(self):
        # The debug allocator asserts no duplicate pages after every release;
        # this drives a multi-leaf round the way _evict_write_through does.
        alloc = _make_allocator(need_sort=False)
        cache = _EvictHarness(alloc)
        nodes = [
            cache.add_leaf(alloc, num_pages=n, token_offset=1000 * i)
            for i, n in enumerate((1, 4, 2, 3))
        ]
        expected = torch.sort(torch.cat([_unique_pages(n.value) for n in nodes]))[0]
        watch = _FreeWatch(alloc)

        for node in nodes:
            cache._evict_regular(node)

        freed = watch.freed()
        self.assertTrue(torch.equal(freed, expected))
        self.assertEqual(len(torch.unique(freed)), len(freed))
        self.assertEqual(alloc.available_size(), NUM_PAGES * PAGE_SIZE)

    def test_non_page_exact_value_is_caught(self):
        # Negative control for the invariant the conversion rests on: a value
        # whose start is not page-aligned shifts the stride's phase, so a page
        # at the tail gets no representative and would leak. The debug oracle
        # catches it -- which is why _evict_regular may only pass node values.
        alloc = _make_allocator(need_sort=False)
        row = alloc.alloc(2 * PAGE_SIZE)
        with self.assertRaises(AssertionError):
            alloc.free_segment(row[1 : PAGE_SIZE + 1], start_pos=0)


if __name__ == "__main__":
    unittest.main()
