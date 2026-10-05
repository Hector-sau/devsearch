"""Latency protocol: workload fixing, nearest-rank percentiles, warm-up/pass accounting, degraded evidence."""

import unittest

from devsearch.evaluation.latency import measure, pass_order, percentile, workload


class PercentileTests(unittest.TestCase):
    def test_nearest_rank_known_answers(self):
        self.assertEqual(percentile([5, 1, 3, 2, 4], 0.5), 3)
        self.assertEqual(percentile(list(range(1, 101)), 0.95), 95)
        self.assertEqual(percentile(list(range(1, 151)), 0.95), 143)          # ceil(0.95 * 150) = 143
        self.assertEqual(percentile([7], 0.95), 7)
        self.assertEqual(percentile([1, 2], 0.01), 1)
        self.assertEqual(percentile([1, 2], 1.0), 2)

    def test_invalid_inputs_are_rejected(self):
        for values, q in (([], 0.5), ([1], 0.0), ([1], 1.5)):
            with self.assertRaises(ValueError):
                percentile(values, q)


class WorkloadTests(unittest.TestCase):
    def test_the_workload_is_fixed_independent_of_input_order_and_seeded(self):
        ids = [f"q{i:03d}" for i in range(100)]
        first = workload(ids, size=50, seed=7)
        self.assertEqual(len(first), 50)
        self.assertEqual(first, workload(list(reversed(ids)), size=50, seed=7))
        self.assertNotEqual(first, workload(ids, size=50, seed=8))
        self.assertEqual(workload(ids[:10], size=50, seed=7).__len__(), 10, "fewer queries than requested are never padded")
        self.assertEqual(workload(ids + ids, size=100, seed=7).__len__(), 100)

    def test_each_pass_has_its_own_seeded_order_that_is_reproducible(self):
        queries = [f"q{i}" for i in range(20)]
        one, two = pass_order(queries, seed=3, index=0), pass_order(queries, seed=3, index=1)
        self.assertEqual(sorted(one), sorted(queries))
        self.assertNotEqual(one, two)
        self.assertEqual(one, pass_order(queries, seed=3, index=0))


class MeasureTests(unittest.TestCase):
    def run_measure(self, status="ok", **kw):
        calls = []
        ticks = iter(x * 0.001 for x in range(10_000))

        def search(query):
            calls.append(query)
            return {"status": status, "candidate_count": 5}
        result = measure(search, [f"q{i}" for i in range(4)], warmup=kw.get("warmup", 10), passes=kw.get("passes", 3), seed=1, clock=lambda: next(ticks))
        return result, calls

    def test_warmup_is_not_timed_and_every_pass_covers_every_query_once(self):
        result, calls = self.run_measure()
        self.assertEqual(len(calls), 10 + 3 * 4)
        self.assertEqual(result["requests"], 12)
        self.assertEqual(calls[:10], [f"q{i % 4}" for i in range(10)])
        for index in range(3):
            self.assertEqual(sorted(calls[10 + 4 * index:14 + 4 * index]), ["q0", "q1", "q2", "q3"])
        self.assertEqual((result["concurrency"], result["response_cache"], result["valid"]), (1, False, True))
        self.assertAlmostEqual(result["p50_ms"], 1.0)                     # the fake clock advances 1 ms per reading

    def test_degraded_responses_invalidate_the_evidence_and_are_counted(self):
        result, _ = self.run_measure(status="degraded")
        self.assertFalse(result["valid"])
        self.assertEqual(result["degraded_requests"], 12)

    def test_the_candidate_count_distribution_is_recorded(self):
        result, _ = self.run_measure()
        self.assertEqual(result["candidate_count"], {"min": 5, "median": 5, "max": 5})

    def test_invalid_configuration_is_rejected(self):
        for kwargs in (dict(queries=[], warmup=1, passes=1), dict(queries=["q"], warmup=0, passes=1), dict(queries=["q"], warmup=1, passes=0)):
            with self.assertRaises(ValueError):
                measure(lambda q: {}, seed=1, **kwargs)


if __name__ == "__main__":
    unittest.main()
