"""Single-owner leases: exclusion, identity-verified stale recovery, races, and release only by the owner."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from devsearch.execution import lock
from devsearch.execution.lock import Lease, LeaseHeld, host_name, process_alive, start_token


RACER = """
import sys, time
from pathlib import Path
from devsearch.execution.lock import Lease, LeaseHeld
try:
    with Lease(Path(sys.argv[1]), purpose="race", plan_id="p"):
        print("won", flush=True)
        time.sleep(0.5)
except LeaseHeld:
    print("lost", flush=True)
"""


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="devsearch-lock-")).resolve()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.path = self.dir / "campaign.lock"

    def write_owner(self, **over):
        record = {"host": host_name(), "pid": 1, "start": None, "token": "t" * 32, "purpose": "other-run", "plan_id": "plan-x"}
        record.update(over)
        self.path.write_text(json.dumps(record) + "\n")

    def dead_pid(self):
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        return child.pid

    def test_acquire_release_and_reacquire(self):
        lease = Lease(self.path, purpose="campaign", plan_id="plan-1")
        owner = lease.acquire()
        self.assertTrue(self.path.is_file())
        self.assertEqual((owner.pid, owner.host, owner.plan_id), (os.getpid(), host_name(), "plan-1"))
        lease.release()
        self.assertFalse(self.path.exists())
        with Lease(self.path, purpose="campaign", plan_id="plan-1"):
            self.assertTrue(self.path.exists())
        self.assertFalse(self.path.exists())

    def test_a_second_owner_is_refused_even_in_the_same_process(self):
        with Lease(self.path, purpose="campaign", plan_id="p"):
            with self.assertRaisesRegex(LeaseHeld, "second owner is refused"):
                Lease(self.path, purpose="serve", plan_id="p").acquire()
            with self.assertRaises(LeaseHeld):
                Lease(self.path, purpose="campaign", plan_id="p").acquire()

    def test_acquiring_twice_through_one_object_is_refused(self):
        lease = Lease(self.path, purpose="campaign", plan_id="p")
        lease.acquire()
        with self.assertRaises(LeaseHeld):
            lease.acquire()
        lease.release()

    def test_a_lease_of_a_dead_process_on_this_host_is_recovered_and_the_evidence_is_kept(self):
        self.write_owner(pid=self.dead_pid(), token="dead" + "0" * 28)
        with Lease(self.path, purpose="campaign", plan_id="p") as lease:
            self.assertEqual(json.loads(self.path.read_text())["token"], lease.owner.token)
        asides = [p.name for p in self.dir.iterdir() if ".stale-" in p.name]
        self.assertEqual(len(asides), 1)
        self.assertIn("other-run", (self.dir / asides[0]).read_text())

    def test_a_reused_pid_with_a_different_start_time_is_stale_but_the_same_start_is_held(self):
        me = os.getpid()
        token = start_token(me)
        if token is None:
            self.skipTest("this platform exposes no process start time")
        self.write_owner(pid=me, start=token + "-different")
        with Lease(self.path, purpose="campaign", plan_id="p"):
            pass
        self.write_owner(pid=me, start=token)
        with self.assertRaises(LeaseHeld):
            Lease(self.path, purpose="campaign", plan_id="p").acquire()

    def test_another_host_or_an_unreadable_file_is_never_broken(self):
        self.write_owner(host="some-other-machine", pid=self.dead_pid())
        with self.assertRaisesRegex(LeaseHeld, "cannot verify"):
            Lease(self.path, purpose="campaign", plan_id="p").acquire()
        self.assertTrue(self.path.exists())
        self.path.write_text("not json\n")
        with self.assertRaisesRegex(LeaseHeld, "unreadable"):
            Lease(self.path, purpose="campaign", plan_id="p").acquire()
        self.assertEqual(self.path.read_text(), "not json\n")

    def test_only_the_owner_removes_the_file(self):
        lease = Lease(self.path, purpose="campaign", plan_id="p")
        lease.acquire()
        self.write_owner(token="x" * 32, pid=os.getpid(), start=start_token(os.getpid()))   # someone else now owns it
        lease.release()
        self.assertTrue(self.path.exists(), "a release by a non-owner must not delete another owner's lease")
        Lease(self.path, purpose="x", plan_id="p").release()                                  # releasing without acquiring is a no-op

    def test_concurrent_acquisition_has_exactly_one_winner(self):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        racers = [subprocess.Popen([sys.executable, "-c", RACER, str(self.path)], stdout=subprocess.PIPE, text=True, env=env) for _ in range(6)]
        results = sorted(p.communicate(timeout=60)[0].strip() for p in racers)
        self.assertEqual(results.count("won"), 1, results)
        self.assertEqual(results.count("lost"), 5, results)
        self.assertFalse(self.path.exists())

    def test_helpers(self):
        self.assertTrue(process_alive(os.getpid()))
        self.assertFalse(process_alive(self.dead_pid()))
        self.assertIsNotNone(lock.host_name())


if __name__ == "__main__":
    unittest.main()
