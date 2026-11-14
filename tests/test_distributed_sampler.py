import unittest

from data.sampler import DistributedRandomBucketSampler


class DistributedBucketCoverageTests(unittest.TestCase):
    def batches_for_ranks(self, lengths, replicas, drop_last=False):
        return [
            list(
                DistributedRandomBucketSampler(1, lengths, 2, drop_last, replicas, rank)
            )
            for rank in range(replicas)
        ]

    def test_all_samples_are_kept(self):
        ranks = self.batches_for_ranks([1] * 8, 2)
        self.assertEqual([len(batches) for batches in ranks], [2, 2])
        self.assertEqual(
            sorted(i for batches in ranks for batch in batches for i in batch),
            list(range(8)),
        )

    def test_padding_keeps_uneven_batch_counts(self):
        ranks = self.batches_for_ranks([1] * 5, 2)
        self.assertEqual(len(ranks[0]), len(ranks[1]))
        self.assertEqual(
            {i for batches in ranks for batch in batches for i in batch}, set(range(5))
        )

    def test_fewer_samples_than_replicas(self):
        self.assertEqual(self.batches_for_ranks([1], 2), [[[0]], [[0]]])

    def test_empty_dataset_has_no_batches(self):
        self.assertEqual(self.batches_for_ranks([], 2), [[], []])

    def test_drop_last_has_equal_nonempty_batches_per_rank(self):
        ranks = self.batches_for_ranks([1] * 9, 2, drop_last=True)
        self.assertEqual(len(ranks[0]), len(ranks[1]))
        self.assertTrue(all(len(batch) == 2 for batches in ranks for batch in batches))
