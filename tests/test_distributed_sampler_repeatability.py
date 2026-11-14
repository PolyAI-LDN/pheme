import unittest

from data.sampler import DistributedRandomBucketSampler


class DistributedBucketRepeatabilityTests(unittest.TestCase):
    def membership(self, sampler):
        # Batch order is intentionally stochastic; membership must stay
        # deterministic so replicas agree on which samples they own.
        return sorted(tuple(sorted(batch)) for batch in sampler)

    def test_repeated_iteration_keeps_rank_membership(self):
        sampler = DistributedRandomBucketSampler(4, list(range(1, 33)), 60, False, 2, 0)
        sampler.set_epoch(2)
        self.assertEqual(self.membership(sampler), self.membership(sampler))

    def test_returning_to_an_epoch_reproduces_membership(self):
        sampler = DistributedRandomBucketSampler(4, list(range(1, 33)), 60, False, 2, 1)
        sampler.set_epoch(2)
        expected = self.membership(sampler)
        sampler.set_epoch(3)
        self.membership(sampler)
        sampler.set_epoch(2)
        self.assertEqual(self.membership(sampler), expected)
