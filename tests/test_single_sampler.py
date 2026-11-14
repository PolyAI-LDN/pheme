import unittest

from data.sampler import SingleRandomBucketSampler, DistributedRandomBucketSampler


class SingleBucketBatchTests(unittest.TestCase):
    def test_oversized_sample_is_a_nonempty_singleton(self):
        batches = list(SingleRandomBucketSampler(1, [11], 10, False))
        self.assertEqual(batches, [[0]])

    def test_mixed_lengths_have_no_empty_batches_or_lost_samples(self):
        lengths = [11, 1, 2, 20, 3]
        batches = list(SingleRandomBucketSampler(2, lengths, 10, False))
        self.assertTrue(all(batches))
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(5)))
        for batch in batches:
            self.assertTrue(
                len(batch) == 1 or max(lengths[i] for i in batch) * len(batch) <= 10
            )

    def test_distributed_oversized_samples_do_not_emit_empty_batches(self):
        batches = list(DistributedRandomBucketSampler(1, [11] * 4, 10, False, 1, 0))
        self.assertTrue(batches)
        self.assertTrue(all(batches))
        self.assertTrue(all(len(batch) == 1 for batch in batches))
