"""CPU contract tests; fixtures are synthetic and never load/merge a model."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from ops_v831_eval.contract import (ACTOR_REQUIRED, FROZEN_PROTOCOL_SHA256, begin_attempt, canonical, digest, materialize_manifest,
    validate_checkpoint, validate_frozen, validate_ready, write)
from ops_v831_eval.merge import merge_commands
from ops_v831_eval.prepare import prepare

ROOT = Path(__file__).resolve().parent


class EvaluationContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.data = self.root / 'v83_data/alfworld'
        reference = json.loads((ROOT / 'reports/v83_20261030/evaluation_protocol.json').read_text())
        self.protocol = dict(protocol=reference['protocol'], task_manifest=dict(splits={}))
        for split, native, n in [('seen', 'valid_seen', 140), ('unseen', 'valid_unseen', 134)]:
            tasks = []
            for i in range(n):
                game_id = f'pick_and_place_simple-Test-{i:03}/trial'
                relative = f'json_2.1.1/{native}/{game_id}/game.tw-pddl'
                path = self.data / relative
                path.parent.mkdir(parents=True)
                path.write_text(f'synthetic {split} {i}')
                tasks.append(dict(game_id=game_id, relative_gamefile=relative,
                    sha256=digest(path), manifest_index=i, task_family='pick_and_place_simple'))
            self.protocol['task_manifest']['splits'][split] = dict(tasks=tasks, by_family={'pick_and_place_simple': n})

    def tearDown(self):
        self.temp.cleanup()

    def test_frozen_274_manifest_mapping_and_rejections(self):
        result = materialize_manifest(self.protocol, self.data)
        self.assertEqual([len(result['splits'][s]['tasks']) for s in ['seen', 'unseen']], [140, 134])
        self.assertTrue(result['splits']['seen']['tasks'][0]['gamefile'].startswith(str(self.data.resolve())))
        for mutation in ['duplicate', 'index', 'path', 'hash', 'protocol', 'missing']:
            bad = copy.deepcopy(self.protocol)
            rows = bad['task_manifest']['splits']['seen']['tasks']
            if mutation == 'duplicate': rows[1]['game_id'] = rows[0]['game_id']
            if mutation == 'index': rows[0]['manifest_index'] = 1
            if mutation == 'path': rows[0]['relative_gamefile'] = '../outside/game.tw-pddl'
            if mutation == 'hash': rows[0]['sha256'] = '0'*64
            if mutation == 'protocol': bad['protocol']['temperature'] = .8
            if mutation == 'missing': rows.pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                materialize_manifest(bad, self.data)

    def checkpoint(self, tier2=False):
        folder = self.root / 'global_step_150'
        actor, base = folder / 'actor', self.root / 'base'
        actor.mkdir(parents=True); base.mkdir()
        metadata = dict(model_type='qwen2', architectures=['Qwen2ForCausalLM'], hidden_size=1536,
            num_hidden_layers=28, num_attention_heads=12, num_key_value_heads=2, vocab_size=151936,
            tie_word_embeddings=True)
        for name in ACTOR_REQUIRED:
            (actor / name).write_text('synthetic artifact ' + name)
        write(actor / 'config.json', metadata); write(base / 'config.json', metadata)
        options = dict(run_id='synthetic-run', config_hash='config', model_contract_hash='model', enable_tier2=tier2)
        state = dict(options, iteration_id=150, version='8.3.1')
        if tier2: state['tier2_state'] = dict(last_committed_iteration=150)
        write(folder / 'v83_runtime.json', state)
        (folder / 'v83_driver_rng.pt').write_bytes(b'synthetic rng')
        marker = dict(state, runtime_sha256=digest(folder / 'v83_runtime.json'), rng_sha256=digest(folder / 'v83_driver_rng.pt'))
        write(folder / 'V83_COMPLETE.json', marker)
        return folder, dict(repro_v83=options), base

    def test_checkpoint_identity_history_and_two_rank_completeness(self):
        folder, config, base = self.checkpoint(tier2=True)
        marker, files = validate_checkpoint(folder, config, base)
        self.assertEqual(marker['iteration_id'], 150)
        self.assertTrue(ACTOR_REQUIRED.issubset(files))
        wrong = copy.deepcopy(config); wrong['repro_v83']['config_hash'] = 'foreign'
        with self.assertRaises(ValueError): validate_checkpoint(folder, wrong, base)
        path = folder / 'actor/model_world_size_2_rank_1.pt'
        original = path.read_bytes(); path.unlink()
        with self.assertRaises(ValueError): validate_checkpoint(folder, config, base)
        path.write_bytes(original)
        state = json.loads((folder / 'v83_runtime.json').read_text())
        state['tier2_state']['last_committed_iteration'] = 149
        write(folder / 'v83_runtime.json', state)
        marker = dict(state, runtime_sha256=digest(folder / 'v83_runtime.json'), rng_sha256=digest(folder / 'v83_driver_rng.pt'))
        write(folder / 'V83_COMPLETE.json', marker)
        with self.assertRaisesRegex(ValueError, 'Incomplete Tier2'): validate_checkpoint(folder, config, base)

    def test_checkpoint_rng_and_model_metadata_tampering_rejected(self):
        folder, config, base = self.checkpoint()
        original = (folder / 'v83_driver_rng.pt').read_bytes()
        (folder / 'v83_driver_rng.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'digest'): validate_checkpoint(folder, config, base)
        (folder / 'v83_driver_rng.pt').write_bytes(original)
        metadata = json.loads((folder / 'actor/config.json').read_text())
        metadata['hidden_size'] += 1
        write(folder / 'actor/config.json', metadata)
        with self.assertRaisesRegex(ValueError, 'model metadata'): validate_checkpoint(folder, config, base)

    def test_all_actor_metadata_is_captured_and_foreign_shards_rejected(self):
        folder, config, base = self.checkpoint()
        (folder / 'actor/custom_metadata.json').write_text('{}')
        (folder / 'actor/optim_world_size_2_rank_0.pt').write_bytes(b'synthetic optimizer')
        _, files = validate_checkpoint(folder, config, base)
        self.assertIn('custom_metadata.json', files)
        self.assertNotIn('optim_world_size_2_rank_0.pt', files)
        (folder / 'actor/model_world_size_4_rank_0.pt').write_bytes(b'foreign shard')
        with self.assertRaisesRegex(ValueError, 'two-rank'): validate_checkpoint(folder, config, base)

    def test_protocol_identity_is_pinned_to_pretraining_artifact(self):
        self.assertEqual(digest(ROOT / 'reports/v83_20261030/evaluation_protocol.json'), FROZEN_PROTOCOL_SHA256)

    def test_prepare_captures_synthetic_host_and_rejects_existing_destination(self):
        checkpoint, config, base = self.checkpoint()
        target = self.root / 'runs/synthetic-run/checkpoints/global_step_150'
        target.parent.mkdir(parents=True)
        checkpoint.rename(target)
        reports = self.root / 'reports'; reports.mkdir()
        vendor = self.root / 'official'; vendor.mkdir()
        (vendor / 'native.py').write_text('pass\n')
        upstream = dict(commit='synthetic', files={'native.py': {'sha256': digest(vendor / 'native.py')}})
        model_contract = dict(files={'config.json': {'sha256': digest(base / 'config.json')}})
        datasets = {}
        for split, native in [('seen', 'valid_seen'), ('unseen', 'valid_unseen')]:
            datasets[native] = {t['game_id']+'/game.tw-pddl': {'sha256': t['sha256']}
                for t in self.protocol['task_manifest']['splits'][split]['tasks']}
        write(reports / 'official_source_manifest.json', upstream)
        write(reports / 'model_contract.json', model_contract)
        write(self.root / 'v83_data/dataset_manifest.json', dict(datasets=datasets))
        options = config['repro_v83']
        options.update(enable_tier1a=True, enable_tier1b=False, deployment_host='a800',
            spec_version='8.3.1', config_hash='PENDING', model_contract_hash=canonical(model_contract))
        config.update(trainer=dict(default_local_dir=str(target.parent)),
            actor_rollout_ref=dict(model=dict(path=str(base))),
            ray_init=dict(runtime_env=dict(env_vars=dict(ALFWORLD_DATA=str(self.data.resolve()),
                PYTHONPATH=':'.join(map(str, [self.root / 'release', vendor]))))))
        contract = dict(config=copy.deepcopy(config), upstream_manifest_sha256=canonical(upstream),
            datasets_sha256=canonical(datasets))
        options['config_hash'] = canonical(contract)
        state = json.loads((target / 'v83_runtime.json').read_text())
        state.update(config_hash=options['config_hash'], model_contract_hash=options['model_contract_hash'])
        write(target / 'v83_runtime.json', state)
        write(target / 'V83_COMPLETE.json', dict(state, runtime_sha256=digest(target / 'v83_runtime.json'),
            rng_sha256=digest(target / 'v83_driver_rng.pt')))
        config_dir = self.root / 'configs'; config_dir.mkdir()
        config_path = config_dir / 'synthetic.yaml'
        write(config_path, config)
        write(config_path.with_suffix('.contract.json'), contract)
        write(config_path.with_suffix('.card.json'), dict(config_sha256=digest(config_path),
            config_hash=options['config_hash'], run_id='synthetic-run', host='a800', phase='tier1a'))
        protocol_path = self.root / 'protocol.json'
        write(protocol_path, dict(self.protocol, checkpoints=[150], inference=dict(tensor_parallel_size=2), data_visibility='synthetic'))
        output = self.root / 'evaluations/synthetic'
        # JSON is a YAML subset. Use the actual parser on deployed hosts; only
        # the local CPU-only environment needs a fallback for this dependency.
        fake_omega = types.SimpleNamespace(OmegaConf=types.SimpleNamespace(
            load=lambda p: json.loads(Path(p).read_text()), to_container=lambda c, resolve: c))
        try:
            import omegaconf
            parser_context = nullcontext()
        except ImportError:
            parser_context = patch.dict(sys.modules, {'omegaconf': fake_omega})
        inventory = {2: ['gpu-a', 'NVIDIA A800'], 3: ['gpu-b', 'NVIDIA A800']}
        with parser_context, patch('ops_v831_eval.prepare.gpu_inventory', return_value=inventory), patch('ops_v831_eval.prepare.FROZEN_PROTOCOL_SHA256', digest(protocol_path)):
            self.assertEqual(prepare(config_path, protocol_path, output, [2, 3], '/tmp/v831-test-unused'), output)
            with self.assertRaisesRegex(ValueError, 'new evaluation directory'):
                prepare(config_path, protocol_path, output, [2, 3], '/tmp/v831-test-unused')
        card = validate_frozen(output)
        self.assertEqual(card['entrypoint_environment']['CUDA_VISIBLE_DEVICES'], '2,3')
        self.assertEqual(card['entrypoint_environment']['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')
        self.assertEqual(card['gpu_uuids'], ['gpu-a', 'gpu-b'])
        self.assertEqual(card['training_run_id'], 'synthetic-run')
        self.assertTrue((output / 'checkpoint_proof/v83_driver_rng.pt').is_file())

    def frozen_evaluation(self):
        root = self.root / 'evaluation'; root.mkdir()
        manifest = materialize_manifest(self.protocol, self.data)
        write(root / 'evaluation_protocol.json', self.protocol)
        write(root / 'task_manifest.json', manifest)
        write(root / 'experiment_card.json', dict(training_run_id='synthetic-run', training_config_hash='config',
            resources=dict(tensor_parallel_size=2, physical_gpu_indices=[2, 3]), gpu_uuids=['gpu-a', 'gpu-b'],
            gpu_inventory={'2': ['gpu-a', 'NVIDIA A800'], '3': ['gpu-b', 'NVIDIA A800']},
            dataset_root=str(self.data), execution_host='a800', phase='tier1a'))
        (root / 'training_config.yaml').write_text('synthetic: true\n')
        proof = root / 'checkpoint_proof'; proof.mkdir()
        for name in ['V83_COMPLETE.json', 'v83_runtime.json']: write(proof / name, dict(synthetic=True))
        (proof / 'v83_driver_rng.pt').write_bytes(b'synthetic rng')
        source = self.root / 'portable-source.py'; source.write_text('pass\n')
        local = {str(p.relative_to(root)): digest(p) for p in root.rglob('*') if p.is_file()}
        write(root / 'freeze.json', dict(local_files=local, external_files={str(source): digest(source)}))
        model = root / 'hf_model'; model.mkdir(); (model / 'model.safetensors').write_bytes(b'synthetic model')
        write(root / 'merge_verification.json', dict(status='passed', run_id='synthetic-run',
            freeze_sha256=digest(root / 'freeze.json'), model_files={'model.safetensors': digest(model / 'model.safetensors')}))
        return root, manifest, source

    def test_unconditional_source_and_model_hash_checks_on_non_h800_path(self):
        root, _, source = self.frozen_evaluation()
        self.assertEqual(validate_ready(root)['execution_host'], 'a800')
        source.write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'official source'): validate_frozen(root)
        source.write_text('pass\n')
        (root / 'hf_model/model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Merged model changed'): validate_ready(root)

    def episodes(self, root, manifest, duplicate=False):
        splits = {}
        for split in ['seen', 'unseen']:
            tasks = manifest['splits'][split]['tasks']; wins = 0
            for batch in range(5):
                episodes = []
                for env_index, task in enumerate(tasks[batch*32:(batch+1)*32]):
                    won = task['manifest_index'] % 2 == 0; wins += won
                    episodes.append(dict(split=split, batch_index=batch, env_index=env_index,
                        manifest_index=task['manifest_index'], gamefile=task['gamefile'], game_id=task['game_id'],
                        task_family=task['task_family'], episode_id=('duplicate' if duplicate else f'{split}-{task["manifest_index"]}'),
                        won=won, episode_reward=10.*won, steps=1,
                        trajectory=[dict(step=1, won=won, reward=10.*won, format_valid=True)]))
                (root / f'{split}_batch_{batch}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in episodes))
            n = len(tasks)
            splits[split] = dict(n=n, successes=wins, success_rate=wins/n, unique_games=n, expected_tasks=n,
                by_family={'pick_and_place_simple': dict(n=n, successes=wins, success_rate=wins/n)})
        write(root / 'results.json', dict(status='complete', checkpoint=150, elapsed_seconds=1, splits=splits))
        write(root / 'evaluation_attempt.json', dict(attempt_id='synthetic-attempt', run_id='synthetic-run'))
        write(root / 'execution.json', dict(attempt_id='synthetic-attempt', run_id='synthetic-run',
            execution_host='a800', tensor_parallel_size=2, physical_gpu_indices=[2, 3],
            gpu_inventory={'2': ['gpu-a', 'NVIDIA A800'], '3': ['gpu-b', 'NVIDIA A800']}))
        write(root / 'evaluation_cost.json', dict(run_id='synthetic-run', returncode=0, physical_gpus=2,
            attempt_id='synthetic-attempt', gpu_hours=2/3600))

    def test_full_cpu_recount_emits_gate_compatible_summary(self):
        root, manifest, _ = self.frozen_evaluation(); self.episodes(root, manifest)
        result = subprocess.run([sys.executable, str(ROOT / 'ops_v831_eval/verify_results.py'), str(root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((root / 'phase_evaluation.json').read_text())
        self.assertEqual({k: summary[k] for k in ['status', 'run_id', 'checkpoint_iteration', 'seen_games', 'unseen_games']},
            dict(status='complete', run_id='synthetic-run', checkpoint_iteration=150, seen_games=140, unseen_games=134))
        self.assertEqual(summary['hashes']['results.json'], digest(root / 'results.json'))

    def test_recount_rejects_duplicate_episode_identity(self):
        root, manifest, _ = self.frozen_evaluation(); self.episodes(root, manifest, duplicate=True)
        # Verification must not silently disappear under Python optimization.
        result = subprocess.run([sys.executable, '-O', str(ROOT / 'ops_v831_eval/verify_results.py'), str(root)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((root / 'phase_evaluation.json').exists())

    def test_fresh_attempt_required_even_before_any_batch_is_saved(self):
        root, _, _ = self.frozen_evaluation()
        first = begin_attempt(root, dict(training_run_id='synthetic-run'))
        self.assertEqual(first['run_id'], 'synthetic-run')
        with self.assertRaises(FileExistsError): begin_attempt(root, dict(training_run_id='synthetic-run'))

    def test_numeric_visibility_is_verified_against_frozen_uuid_mapping(self):
        root, _, _ = self.frozen_evaluation()
        inventory = {2: ['gpu-a', 'NVIDIA A800'], 3: ['gpu-b', 'NVIDIA A800']}
        env = dict(CUDA_VISIBLE_DEVICES='2,3', CUDA_DEVICE_ORDER='PCI_BUS_ID',
            ALFWORLD_DATA=str(self.data), VLLM_WORKER_MULTIPROC_METHOD='spawn')
        with patch('ops_v831_eval.contract.gpu_inventory', return_value=inventory), patch.dict('os.environ', env):
            validate_ready(root, check_environment=True, check_gpus=True)
        with patch('ops_v831_eval.contract.gpu_inventory', return_value=inventory), patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': 'gpu-a,gpu-b'}):
            with self.assertRaisesRegex(ValueError, 'numeric GPU'): validate_ready(root, check_gpus=True)
        with patch.dict('os.environ', dict(env, CUDA_DEVICE_ORDER='FASTEST_FIRST')):
            with self.assertRaisesRegex(ValueError, 'PCI_BUS_ID'): validate_ready(root, check_environment=True)

    def test_merge_commands_use_official_two_rank_cpu_merger_and_verification(self):
        commands = merge_commands('/eval', '/official', '/python')
        self.assertEqual([cmd[2] for cmd in commands], ['merge', 'test'])
        for cmd in commands:
            self.assertEqual(cmd[1], '/official/scripts/model_merger.py')
            self.assertIn('fsdp', cmd); self.assertIn('/eval/checkpoint_actor', cmd)
        self.assertIn('--test_hf_dir', commands[1])


if __name__ == '__main__': unittest.main()
