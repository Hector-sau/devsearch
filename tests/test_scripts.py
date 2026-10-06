"""The packaging helpers in scripts/: lock resolution (with a fake resolver), offline install from a wheelhouse, receipts' plumbing.

No network: the resolver is a stand-in executable and the "wheelhouse" holds a hand-built minimal wheel.
"""

import hashlib
import importlib.util
import json
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from helpers import REPO


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lock_profile = load("lock_profile")
bootstrap_env = load("bootstrap_env")
record_receipt = load("record_receipt")

FAKE_UV = '''#!{python}
import sys
args = sys.argv[1:]
if args[:1] == ["--version"]:
    print("uv 0.0-fake")
    sys.exit(0)
out = args[args.index("--output-file") + 1]
open(out, "w").write(OUTPUT)
open(out + ".args", "w").write(" ".join(args))
'''


def fake_uv(directory: Path, output: str) -> Path:
    path = directory / "uv"
    path.write_text(FAKE_UV.format(python=sys.executable).replace("OUTPUT", repr(output)))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def make_wheel(directory: Path, name="devsearch", version="0.0.0") -> Path:
    wheel = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{name}/__init__.py", "__version__ = '0.0.0'\n")
        archive.writestr(f"{name}-{version}.dist-info/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        archive.writestr(f"{name}-{version}.dist-info/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{name}-{version}.dist-info/RECORD", "")
    return wheel


class LockTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="devsearch-lock-")).resolve()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.out = self.dir / "locks"
        self.out.mkdir()

    def run_lock(self, uv, *extra, profile="apple-mps"):
        return subprocess.run([sys.executable, str(REPO / "scripts" / "lock_profile.py"), "--profile", profile, "--uv", str(uv), "--output", str(self.out), *extra],
                              capture_output=True, text=True, timeout=60)

    def test_the_requirement_names_come_from_the_two_extras(self):
        self.assertEqual(lock_profile.requirement_names(REPO / "pyproject.toml"),
                         ["torch", "numpy", "transformers", "tokenizers", "safetensors", "fastapi", "uvicorn", "httpx", "streamlit"])

    def test_a_hashed_resolution_becomes_a_lock_with_an_honest_header(self):
        body = "torch==2.0.0 \\\n    --hash=sha256:" + "a" * 64 + "\nnumpy==2.0.0 \\\n    --hash=sha256:" + "b" * 64 + "\n"
        result = self.run_lock(fake_uv(self.dir, body))
        self.assertEqual(result.returncode, 0, result.stderr)
        lock = (self.out / "apple-mps-cp312.lock").read_text()
        self.assertIn("UNVERIFIED on the target host", lock)
        self.assertIn("uv 0.0-fake", lock)
        self.assertIn("torch==2.0.0", lock)
        self.assertIn("2 pinned packages", result.stdout)
        self.assertEqual(hashlib.sha256((self.out / "apple-mps-cp312.lock").read_bytes()).hexdigest(), result.stdout.split("sha256 ")[1].strip())
        self.assertEqual([p.name for p in self.out.iterdir()], ["apple-mps-cp312.lock"], "the scratch directory is removed")

    def test_the_resolver_is_pointed_at_the_target_platform_not_this_machine(self):
        recorded = {}
        original = lock_profile.build_command
        for profile, platform in (("apple-mps", "aarch64-apple-darwin"), ("nvidia-cuda", "x86_64-manylinux_2_28")):
            command = original("uv", Path("in.txt"), lock_profile.TARGETS[profile]["uv_platform"], "3.12", Path("out"))
            recorded[profile] = command
            self.assertIn(platform, command)
            self.assertIn("--generate-hashes", command)
            self.assertEqual(command[command.index("--python-version") + 1], "3.12")

    def test_an_unhashed_resolution_a_failure_and_an_existing_lock_are_refused(self):
        result = self.run_lock(fake_uv(self.dir, "torch==2.0.0\n"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no hashes", result.stderr)
        self.assertEqual(list(self.out.iterdir()), [])
        (self.out / "apple-mps-cp312.lock").write_text("keep")
        result = self.run_lock(fake_uv(self.dir, "x==1 --hash=sha256:" + "c" * 64 + "\n"))
        self.assertIn("never overwritten", result.stderr)
        self.assertEqual((self.out / "apple-mps-cp312.lock").read_text(), "keep")

    def test_a_missing_tool_a_missing_directory_and_an_old_python_are_refused(self):
        self.assertIn("not found", self.run_lock(self.dir / "no-such-uv").stderr)
        shutil.rmtree(self.out)
        self.assertIn("never created", self.run_lock(fake_uv(self.dir, "")).stderr)
        self.out.mkdir()
        self.assertNotEqual(self.run_lock(fake_uv(self.dir, ""), "--python-version", "3.9").returncode, 0)

    def test_the_shipped_locks_are_hashed_pinned_and_say_they_are_unverified(self):
        for path in sorted((REPO / "locks").glob("*.lock")):
            text = path.read_text()
            with self.subTest(path.name):
                self.assertIn("UNVERIFIED on the target host", text)
                pins = [line for line in text.splitlines() if line and not line.startswith(("#", " "))]
                self.assertTrue(pins)
                self.assertTrue(all("==" in line for line in pins), "every package is pinned to an exact version")
                self.assertEqual(text.count("=="), len(pins))
                self.assertTrue(all(len(part) == 64 for part in [h.split("sha256:")[1].split()[0] for h in text.splitlines() if "--hash=sha256:" in h]))
                names = {line.split("==")[0] for line in pins}
                self.assertTrue({"torch", "numpy", "transformers", "tokenizers", "safetensors", "fastapi", "uvicorn", "httpx", "streamlit"} <= names)
        mac = (REPO / "locks" / "apple-mps-cp312.lock").read_text()
        self.assertFalse(any(line.startswith(("nvidia-", "cuda-")) for line in mac.splitlines()), "no CUDA package in the macOS lock")


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="devsearch-boot-")).resolve()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.house = self.dir / "wheelhouse"
        self.house.mkdir()
        self.wheel = make_wheel(self.house)
        digest = hashlib.sha256(self.wheel.read_bytes()).hexdigest()
        self.lock = self.dir / f"apple-mps-cp{sys.version_info.major}{sys.version_info.minor}.lock"
        self.lock.write_text(f"devsearch==0.0.0 \\\n    --hash=sha256:{digest}\n")

    def install(self, lock=None, venv=None):
        return subprocess.run([sys.executable, str(REPO / "scripts" / "bootstrap_env.py"), "install", "--lock", str(lock or self.lock), "--wheelhouse", str(self.house),
                               "--wheel", str(self.wheel), "--venv", str(venv or self.dir / "env")], capture_output=True, text=True, timeout=300)

    def test_an_offline_install_from_a_hashed_wheelhouse_builds_a_new_environment_and_reports_the_lock_hash(self):
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr[-1500:])
        report = json.loads(result.stdout)
        self.assertEqual(report["lock_sha256"], hashlib.sha256(self.lock.read_bytes()).hexdigest())
        self.assertEqual(report["pip_check_exit"], 0)
        self.assertTrue(Path(report["devsearch_imported_from"]).is_relative_to(self.dir / "env"))
        again = self.install()
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("always builds a new environment", again.stderr)

    def test_a_lock_for_another_python_minor_is_refused_with_a_clear_message(self):
        other = self.dir / f"apple-mps-cp{sys.version_info.major}{sys.version_info.minor + 1}.lock"
        other.write_text(self.lock.read_text())
        result = self.install(lock=other)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("was resolved for CPython", result.stderr)
        self.assertFalse((self.dir / "env").exists())

    def test_a_hash_that_does_not_match_the_wheelhouse_is_rejected(self):
        self.lock.write_text("devsearch==0.0.0 \\\n    --hash=sha256:" + "0" * 64 + "\n")
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("command failed", result.stderr)

    def test_download_refuses_a_missing_destination_and_an_unknown_platform(self):
        base = [sys.executable, str(REPO / "scripts" / "bootstrap_env.py"), "download", "--lock"]
        result = subprocess.run([*base, str(self.lock), "--dest", str(self.dir / "nowhere")], capture_output=True, text=True, timeout=60)
        self.assertIn("never created", result.stderr)
        odd = self.dir / "mystery.lock"
        odd.write_text("x==1\n")
        result = subprocess.run([*base, str(odd), "--dest", str(self.house)], capture_output=True, text=True, timeout=60)
        self.assertIn("cannot infer the target platform", result.stderr)
        cuda = bootstrap_env.PLATFORM_TAGS["nvidia-cuda"]
        self.assertTrue({"manylinux_2_28_x86_64", "manylinux_2_27_x86_64", "manylinux_2_25_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64"} <= set(cuda),
                        "vendor wheels use older glibc tags than the profile's own, and pip does not expand them")
        self.assertEqual(bootstrap_env.profile_of(Path("nvidia-cuda-cp312.lock")), "nvidia-cuda")
        self.assertIsNone(bootstrap_env.profile_of(Path("other.lock")))


class MeasureBaselineTests(unittest.TestCase):
    def test_the_declared_protocol_runs_on_a_fixture_index_without_a_model(self):
        from helpers import MINI_CORPUS, CONFIGS, index_config
        from devsearch.retrieval.index_store import build_index
        root = Path(tempfile.mkdtemp(prefix="devsearch-measure-")).resolve()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        shutil.copytree(MINI_CORPUS, root / "corpus")
        (root / "index").mkdir()
        build_index(index_config(), root / "corpus" / "corpus-manifest.json", root / "index" / "index-manifest.json", root)
        (root / "queries.txt").write_text("how do I load_state_dict\nwhat is a checkpoint\n\nsave model weights\n")
        result = subprocess.run([sys.executable, str(REPO / "scripts" / "measure_baseline.py"), "--index", str(root / "index" / "index-manifest.json"),
                                 "--search", str(CONFIGS / "search.toml"), "--queries", str(root / "queries.txt"), "--size", "3", "--warmup", "2", "--passes", "2",
                                 "--runtime-root", str(root), "--label", "unit-test stand-in"], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual((report["queries"], report["requests"], report["concurrency"], report["response_cache"], report["valid"]), (3, 6, 1, False, True))
        self.assertIn("no model", report["mode"])
        self.assertEqual(report["workload_label"], "unit-test stand-in")
        by_directory = subprocess.run([sys.executable, str(REPO / "scripts" / "measure_baseline.py"), "--index", str(root / "index"), "--search", str(CONFIGS / "search.toml"),
                                       "--queries", str(root / "queries.txt"), "--size", "3", "--warmup", "1", "--passes", "1", "--runtime-root", str(root)],
                                      capture_output=True, text=True, timeout=60)
        self.assertEqual(by_directory.returncode, 0, by_directory.stderr)
        self.assertEqual(json.loads(by_directory.stdout)["index_id"], report["index_id"])


class PortableReceiptTests(unittest.TestCase):
    """The portable receipt installs only from the hashed lock and the offline wheelhouse and proves that nothing is borrowed. The wheels here are
    hand-built stand-ins, so the campaign itself cannot run; what is checked is the dependency evidence and the refusals."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="devsearch-portable-test-")).resolve()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.house = self.dir / "wheelhouse"
        self.house.mkdir()
        self.dep = make_wheel(self.house, name="tinydep", version="1.0")
        self.project = make_wheel(self.dir, name="devsearch")
        self.out = self.dir / "receipts"
        self.out.mkdir()
        self.tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        self.lock = self.dir / f"nvidia-cuda-{self.tag}.lock"
        self.lock.write_text(self.pin(self.dep, "tinydep", "1.0"))

    @staticmethod
    def pin(wheel, name, version):
        return f"{name}=={version} \\\n    --hash=sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}\n"

    def run_portable(self, lock=None, python=None):
        return subprocess.run([sys.executable, str(REPO / "scripts" / "record_receipt.py"), "portable", "--wheel", str(self.project), "--lock", str(lock or self.lock),
                               "--wheelhouse", str(self.house), "--output", str(self.out), "--python", python or sys.executable], capture_output=True, text=True, timeout=600)

    def test_everything_comes_from_the_lock_and_the_receipt_records_the_evidence(self):
        result = self.run_portable()
        self.assertEqual(result.returncode, 1, "the stand-in wheel cannot run the campaign, so the run is not a pass")
        report = json.loads(result.stdout)
        deps = report["summary"]["dependencies"]
        self.assertEqual(deps["lock_sha256"], hashlib.sha256(self.lock.read_bytes()).hexdigest())
        self.assertEqual((deps["lock"], deps["lock_profile"], deps["lock_python"]), (self.lock.name, "nvidia-cuda", f"{sys.version_info.major}.{sys.version_info.minor}"))
        self.assertEqual((deps["pip_check_exit"], deps["installed_matches_lock"], deps["borrowed_dependencies"], deps["foreign_sys_path_entries"]), (0, True, False, []))
        self.assertEqual((deps["wheelhouse_wheels"], deps["locked_packages"]), (1, 1))
        self.assertEqual(deps["install_host"]["machine"], __import__("platform").machine())
        install = report["commands"][0][0]
        self.assertIn("--require-hashes", install)
        self.assertIn("--no-index", install)
        self.assertFalse(report["summary"]["rehearsal_complete"])
        self.assertFalse(report["summary"]["wheel_matches_checkout"], "a stand-in wheel never matches this checkout's snapshot")
        self.assertEqual(len(list(self.out.glob("portable_bundle_verified-*.json"))), 1, "a failed run still leaves its sealed receipt, which the gate rejects")

    def test_a_locked_package_missing_from_the_wheelhouse_fails_the_install_and_the_receipt_says_so(self):
        ghost = self.dir / "ghostpkg-1.0-py3-none-any.whl"
        ghost.write_bytes(b"not in the wheelhouse")
        self.lock.write_text(self.lock.read_text() + self.pin(ghost, "ghostpkg", "1.0"))
        result = self.run_portable()
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertNotEqual(report["commands"][0][1], 0, "pip install from the lock must fail, nothing may fill the gap")
        self.assertFalse(report["summary"]["dependencies"]["installed_matches_lock"])
        self.assertEqual(report["summary"]["dependencies"]["pip_check_exit"], None)

    def test_the_interpreter_must_match_the_lock_and_nothing_is_written_otherwise(self):
        other = self.dir / f"nvidia-cuda-cp{sys.version_info.major}{sys.version_info.minor + 1}.lock"
        other.write_text(self.lock.read_text())
        result = self.run_portable(lock=other)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("was resolved for CPython", result.stderr)
        self.assertEqual(list(self.out.iterdir()), [])

    def test_each_operating_system_gets_its_own_network_isolation(self):
        linux, _ = record_receipt.isolation_command("Linux")
        self.assertEqual(linux, ["unshare", "-rn"])
        mac, method = record_receipt.isolation_command("Darwin")
        self.assertEqual(mac[:2], ["sandbox-exec", "-p"])
        self.assertIn("(deny network*)", mac[2])
        self.assertIn("macOS", method)
        self.assertIsNone(record_receipt.isolation_command("Windows"))
        unknown = record_receipt.network_isolation(sys.executable, system="Windows")
        self.assertEqual((unknown["verified"], unknown["prefix"]), (False, []))

    def test_an_isolation_that_still_connects_is_not_accepted(self):
        from unittest import mock
        with mock.patch.object(record_receipt, "isolation_command", return_value=(["env"], "no isolation at all")), \
                mock.patch.object(record_receipt, "PROBE", "print('CONNECTED')"):
            leaky = record_receipt.network_isolation(sys.executable)
        self.assertFalse(leaky["verified"])
        self.assertEqual(leaky["prefix"], [])
        self.assertIn("not usable", leaky["method"])

    def test_the_isolation_on_this_host_is_either_proven_or_refused_never_assumed(self):
        system = __import__("platform").system()
        result = record_receipt.network_isolation(sys.executable)
        works = system in ("Linux", "Darwin") and subprocess.run([*record_receipt.isolation_command(system)[0], "true"], capture_output=True, check=False).returncode == 0
        if works:                                            # e.g. this development container
            self.assertTrue(result["verified"], result)
            self.assertTrue(result["probe"].startswith("BLOCKED"), result)
        else:                                                # e.g. a CI runner that forbids unprivileged namespaces: offline is then not claimed
            self.assertFalse(result["verified"], result)
            self.assertEqual(result["prefix"], [])

    def test_lock_pins_and_target_are_read_from_the_lock(self):
        pins = record_receipt.lock_pins("# comment\nTorch==2.14.1 \\\n    --hash=sha256:aa\nnvidia_cuda-runtime==12.9.79 ; platform_system == 'Linux' \\\n    --hash=sha256:bb\n")
        self.assertEqual(pins, {"torch": "2.14.1", "nvidia-cuda-runtime": "12.9.79"})
        self.assertEqual(record_receipt.lock_target(Path("locks/apple-mps-cp312.lock")), {"profile": "apple-mps", "python": "3.12"})
        self.assertEqual(record_receipt.lock_target(Path("odd.lock")), {"profile": None, "python": None})

    def test_borrowed_paths_modules_and_unlocked_packages_are_detected(self):
        env = self.dir / "env"
        inspection = {"sys_path": [str(env / "lib" / "python3.12" / "site-packages"), "/usr/lib/python3.12", "/usr/lib/python312.zip"], "prefix": str(env),
                      "stdlib": ["/usr/lib/python3.12", "/usr/lib/python3.12"], "installed": {"torch": "2.14.1", "pip": "24.0", "devsearch": "0.0.0"},
                      "origins": {"torch": str(env / "lib" / "python3.12" / "site-packages" / "torch" / "__init__.py"), "numpy": None}}
        clean = record_receipt.dependency_facts(inspection, {"torch": "2.14.1"}, env)
        self.assertEqual((clean["borrowed_dependencies"], clean["installed_matches_lock"], clean["foreign_sys_path_entries"]), (False, True, []))
        borrowed = dict(inspection, sys_path=[*inspection["sys_path"], "/home/dev/venv/lib/python3.12/site-packages"])
        facts = record_receipt.dependency_facts(borrowed, {"torch": "2.14.1"}, env)
        self.assertTrue(facts["borrowed_dependencies"])
        self.assertEqual(facts["foreign_sys_path_entries"], ["/home/dev/venv/lib/python3.12/site-packages"])
        outside = dict(inspection, origins={"torch": "/home/dev/venv/lib/python3.12/site-packages/torch/__init__.py", "numpy": None})
        self.assertTrue(record_receipt.dependency_facts(outside, {"torch": "2.14.1"}, env)["borrowed_dependencies"])
        extra = dict(inspection, installed={**inspection["installed"], "requests": "2.0"})
        facts = record_receipt.dependency_facts(extra, {"torch": "2.14.1"}, env)
        self.assertFalse(facts["installed_matches_lock"])
        self.assertEqual(facts["unlocked_packages_installed"], ["requests"])
        wrong = record_receipt.dependency_facts(inspection, {"torch": "2.15.0", "numpy": "2.5.3"}, env)
        self.assertFalse(wrong["installed_matches_lock"])
        self.assertEqual(wrong["locked_packages_missing_or_different"], ["numpy", "torch"])


class ReceiptPlumbingTests(unittest.TestCase):
    def test_the_unittest_summary_parser_reads_ok_failed_and_skips(self):
        record_receipt = load("record_receipt")
        for text, expected in (("Ran 12 tests in 1.0s\n\nOK\n", (12, "OK", {})), ("Ran 5 tests in 1.0s\n\nOK (skipped=2)\n", (5, "OK", {"skipped": "2"})),
                               ("Ran 7 tests in 1.0s\n\nFAILED (failures=1, errors=2, skipped=3)\n", (7, "FAILED", {"failures": "1", "errors": "2", "skipped": "3"}))):
            ran, verdict = record_receipt._RAN.search(text), record_receipt._VERDICT.search(text)
            counts = dict(part.split("=") for part in (verdict.group(2) or "").split(", ") if "=" in part)
            self.assertEqual((int(ran.group(1)), verdict.group(1), counts), expected)


if __name__ == "__main__":
    unittest.main()
