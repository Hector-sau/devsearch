"""Original fixture evidence only: import integrity and fail-closed runner integration."""

import copy
import errno
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from helpers import REPO
from devsearch.artifacts.canonical import seal_manifest
from devsearch.artifacts.identity import code_identity
from devsearch.execution.acceptance import make_receipt

sys.path.insert(0, str(REPO / "scripts"))
import ci_receipts as CI
import gpu_fixture

GPU = gpu_fixture.load_gpu()


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="ci-import-test-"))
        self.addCleanup(shutil.rmtree, self.base)
        self.root = self.base / "checkout"
        self.root.mkdir()
        (self.root / "src/devsearch").mkdir(parents=True)
        (self.root / "src/devsearch/__init__.py").write_text('# original test package\n')
        (self.root / "scripts").mkdir()
        (self.root / "scripts/gpu.py").write_text('# original test runner\n')
        (self.root / "locks").mkdir()
        self.lock = self.root / "locks/nvidia-cuda-cp312.lock"
        self.lock.write_text('fixture==1 --hash=sha256:abc\n')
        self.directory = self.base / "evidence"
        self.directory.mkdir()
        wheel = self.directory / "devsearch-0-py3-none-any.whl"
        wheel.write_bytes(b'fixture wheel; never installed')
        snapshot = code_identity(self.root / "src/devsearch")["source_snapshot_sha256"]
        commands = [{"exit_status": 0, "argv": ["fixture"]}]
        portable = {"outside_checkout": True, "rehearsal_complete": True, "offline": True,
                    "wheel_matches_checkout": True, "wheel_sha256": CI.sha(wheel),
                    "isolation_probe": {"control": {"interfaces": ["lo", "eth0"], "connect": {"result": "connected"}},
                                        "isolated": {"interfaces": ["lo"], "connect": {"result": "error", "errno": 101}}},
                    "dependencies": {"lock_sha256": CI.sha(self.lock), "lock_profile": "nvidia-cuda", "lock_python": "3.12",
                                     "borrowed_dependencies": False, "foreign_sys_path_entries": [], "pip_check_exit": 0,
                                     "installed_matches_lock": True, "wheelhouse_wheels": 1,
                                     "install_host": {"system": "Linux", "machine": "x86_64", "python": "3.12"}}}
        self.receipts = {
            "implementation_verified-fixture.json": make_receipt("implementation_verified", snapshot, commands,
                                                                 {"tests_run": 1, "skipped": 0}, {}, "2026-10-08T00:00:00Z"),
            "portable_bundle_verified-fixture.json": make_receipt("portable_bundle_verified", snapshot, commands, portable, {}, "2026-10-08T00:00:00Z")}
        for name, receipt in self.receipts.items():
            CI.write(self.directory / name, receipt)
        self.manifest = {"schema_version": 1, "repository": CI.REPOSITORY, "workflow": CI.WORKFLOW, "commit": "a" * 40,
                         "run_id": 123, "run_attempt": 1, "profile": "nvidia-cuda", "lock_sha256": CI.sha(self.lock),
                         "code_snapshot": snapshot, "deployment": {p.relative_to(self.root).as_posix(): CI.sha(p) for p in self.root.rglob('*') if p.is_file()},
                         "wheel": wheel.name, "receipts": list(self.receipts), "wheelhouse": {"fixture.whl": "b" * 64},
                         "files": {p.name: CI.sha(p) for p in self.directory.iterdir()}}
        self.origin = {"schema_version": 1, "repository": CI.REPOSITORY, "workflow": CI.WORKFLOW, "commit": "a" * 40,
                       "run_id": 123, "run_attempt": 1, "conclusion": "success"}
        self.repin()

    def repin(self):
        CI.write(self.directory / CI.MANIFEST, self.manifest)
        self.origin["manifest_sha256"] = CI.sha(self.directory / CI.MANIFEST)
        CI.write(self.directory / CI.ORIGIN, self.origin)
        self.pin = CI.sha(self.directory / CI.ORIGIN)

    def verify(self):
        return CI.verify(self.directory, self.pin, self.root, self.lock)

    def change_receipt(self, gate, change):
        name = gate + '-fixture.json'
        body = copy.deepcopy(self.receipts[name])
        change(body)
        CI.write(self.directory / name, seal_manifest(body))
        self.manifest['files'][name] = CI.sha(self.directory / name)
        self.repin()

    def test_valid_evidence_is_accepted_without_running_unshare(self):
        with mock.patch.object(CI.subprocess, 'run', side_effect=AssertionError('no subprocess needed')):
            self.assertEqual(self.verify()['commit'], 'a' * 40)

    def test_linux_probe_errnos_are_interpreted_as_linux_even_on_a_mac(self):
        for number in (101, 111):
            self.change_receipt('portable_bundle_verified', lambda r: r['summary']['isolation_probe']['isolated']['connect'].update(errno=number))
            self.verify()
        for number in (51, 61, 110, 1):
            self.change_receipt('portable_bundle_verified', lambda r: r['summary']['isolation_probe']['isolated']['connect'].update(errno=number))
            with self.assertRaisesRegex(ValueError, 'positive control'):
                self.verify()

    def test_changed_pin_manifest_receipt_and_wheel_are_refused(self):
        for name in [CI.ORIGIN, CI.MANIFEST, *self.receipts, self.manifest['wheel']]:
            with self.subTest(name=name):
                path = self.directory / name
                old = path.read_bytes()
                path.write_bytes(old + b' ')
                with self.assertRaises(ValueError):
                    self.verify()
                path.write_bytes(old)

    def test_changed_runner_and_unrecorded_source_are_refused(self):
        runner = self.root / 'scripts/gpu.py'
        runner.write_text('changed runner, same package identity')
        with self.assertRaisesRegex(ValueError, 'deployment file changed'):
            self.verify()
        runner.write_text('# original test runner\n')
        (self.root / 'scripts/extra.py').write_text('additional source')
        with self.assertRaisesRegex(ValueError, 'unrecorded deployment'):
            self.verify()

    def test_wrong_commit_attempt_lock_platform_python_and_wheel_fail(self):
        for field, wrong in [('commit', 'c' * 40), ('run_attempt', 2), ('lock_sha256', '0' * 64), ('profile', 'apple-mps')]:
            with self.subTest(field=field):
                old = self.manifest[field]
                self.manifest[field] = wrong
                self.repin()
                with self.assertRaises(ValueError):
                    self.verify()
                self.manifest[field] = old
        self.change_receipt('portable_bundle_verified', lambda r: r['summary']['dependencies']['install_host'].update(python='3.11'))
        with self.assertRaisesRegex(ValueError, 'Python mismatch'):
            self.verify()
        self.change_receipt('portable_bundle_verified', lambda r: r['summary'].update(wheel_sha256='0' * 64))
        with self.assertRaisesRegex(ValueError, 'different project wheel'):
            self.verify()

    def test_skips_failed_commands_and_inconclusive_isolation_are_refused(self):
        cases = [('implementation_verified', lambda r: r['summary'].update(skipped=1)),
                 ('portable_bundle_verified', lambda r: r['commands'][0].update(exit_status=1)),
                 ('portable_bundle_verified', lambda r: r['summary']['isolation_probe']['isolated']['connect'].update(result='timeout'))]
        for gate, change in cases:
            with self.subTest(gate=gate):
                self.change_receipt(gate, change)
                with self.assertRaises(ValueError):
                    self.verify()
                self.change_receipt(gate, lambda r: r.update(self.receipts[gate + '-fixture.json']))

    def test_github_provenance_requires_both_jobs_exact_commit_and_digest(self):
        run = {'head_sha': 'a'*40, 'status': 'completed', 'conclusion': 'success', 'path': CI.WORKFLOW,
               'repository': {'full_name': CI.REPOSITORY}, 'head_repository': {'full_name': CI.REPOSITORY}}
        jobs = [{'name': n, 'status': 'completed', 'conclusion': 'success'} for n in ('locked-env', 'real-model')]
        artifact = {'name': 'locked-env-receipts-' + 'a'*40, 'expired': False, 'workflow_run': {'head_sha': 'a'*40}, 'digest': 'sha256:' + 'd'*64}
        CI.validate_run(run, jobs, artifact, 'a'*40)
        for obj, field, wrong in [(run, 'head_sha', 'b'*40), (run, 'conclusion', 'failure'), (run, 'path', 'unrelated.yml'),
                                  (artifact, 'digest', None), (artifact, 'expired', True), (jobs[1], 'conclusion', 'skipped')]:
            old = obj[field]
            obj[field] = wrong
            with self.assertRaises(ValueError):
                CI.validate_run(run, jobs, artifact, 'a'*40)
            obj[field] = old

    def test_archive_path_traversal_and_duplicates_are_refused(self):
        archive = self.base / 'bad.zip'
        for names in [('../escape',), ('/absolute',), ('x', 'x')]:
            with zipfile.ZipFile(archive, 'w') as z:
                for name in names:
                    z.writestr(name, 'fixture')
            with self.assertRaises(ValueError):
                CI.unpack(archive, self.base / 'unpack')


class RunnerImportTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix='ci-runner-test-'))
        self.addCleanup(shutil.rmtree, self.base)
        self.root = self.base / 'runtime'
        self.root.mkdir()
        self.config = self.base / 'gpu.toml'
        raw = (REPO / 'configs/gpu.toml').read_text().replace('/workspace/devsearch-runtime', str(self.root))
        raw = raw.replace('source = "local"', 'source = "ci"')
        raw += f'\nimport_dir = "{self.base}/ci"\nimport_sha256 = "{"f"*64}"\n'
        self.config.write_text(raw)
        self.rt = GPU.Runtime(GPU.Config(self.config))
        self.rt.state.mkdir()

    def test_ci_import_is_explicit_and_cannot_disable_receipts(self):
        self.config.write_text(self.config.read_text().replace('record = true', 'record = false'))
        with self.assertRaises(GPU.Stop):
            GPU.Config(self.config)

    def test_ci_namespace_is_not_applicable_only_with_valid_import(self):
        def probe(argv, **kw):
            self.assertNotIn('unshare', argv)
            return type('Result', (), {'returncode': 0, 'stdout': '3.12\n', 'stderr': ''})()
        with mock.patch.object(GPU, 'imported_evidence', return_value={'commit': 'a'*40}), \
                mock.patch.object(GPU, 'gpu_facts', return_value={'error': 'fixture no GPU'}), \
                mock.patch.object(GPU, 'reachable', return_value=None), mock.patch.object(GPU.subprocess, 'run', side_effect=probe), \
                mock.patch.object(GPU.shutil, 'which', return_value=sys.executable):
            checks = GPU.run_check(self.rt, quiet=True)
            self.assertEqual(next(c for c in checks if c['name'] == 'network-namespace')['status'], 'not_applicable')
            with mock.patch.object(GPU, 'imported_evidence', side_effect=GPU.prerequisite('ci-evidence', 'wrong pin')):
                checks = GPU.run_check(self.rt, quiet=True)
                self.assertEqual(next(c for c in checks if c['name'] == 'ci-evidence')['status'], 'failed')

    def test_changed_prepared_evidence_is_refused_on_resume(self):
        with self.assertRaisesRegex(GPU.Stop, 'changed'):
            GPU.validate_imported_state(self.rt, {'directory': str(self.base), 'files_sha256': {}, 'import_sha256': 'old'}, {'origin_sha256': 'new'})

    def test_import_never_substitutes_for_local_diagnosis(self):
        self.rt.record('prepare')
        self.rt.record('bundle', diagnostic=False)
        self.rt.record('receipts', code_key=self.rt.code_key())
        with mock.patch.object(GPU, 'imported_evidence', return_value={'commit': 'a'*40}), \
                mock.patch.object(GPU, 'validate_imported_state'):
            with self.assertRaises(GPU.Stop) as caught:
                GPU.stage_run(self.rt)
            self.assertEqual(caught.exception.prerequisite, 'diagnose')


if __name__ == '__main__':
    unittest.main()
