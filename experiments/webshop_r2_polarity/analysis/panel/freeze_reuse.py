"""Bind the existing development panel, final checkpoint, and unchanged references."""
from pathlib import Path
import shutil
from common import ROOT, BASE, R2, SOURCE, checkpoint_info, digest, now, read, resources, write

OLD = BASE / 'evaluations/gradus-webshop-r2aligned-panel-20261109'
ORIGINAL_RUN = BASE / 'runs/gradus-webshop-h800-s0-r2aligned-20261109'


def freeze():
    assert not (ROOT / 'experiment_card.json').exists(), 'Already frozen'
    inventory = resources()
    old = read(OLD / 'experiment_card.json')
    for filename, expected in old['frozen_files'].items():
        assert digest(filename) == expected, filename
    assert digest(ROOT / 'task_manifest.json') == digest(OLD / 'task_manifest.json')
    shared = {}
    for filename, expected in old['frozen_files'].items():
        path = Path(filename)
        if path.is_relative_to(ORIGINAL_RUN / 'source'):
            current = SOURCE / path.relative_to(ORIGINAL_RUN / 'source')
            assert digest(current) == expected, str(current)
            shared[str(current)] = expected
    assert shared
    assert digest(ROOT/'native_projection.py') == digest(SOURCE/'agent_system/environments/env_package/webshop/projection.py')
    assert read(R2 / 'final_training_verification.json')['status'] == 'PASS'
    checkpoint = checkpoint_info(R2, 150)
    references = {}
    for tag, original_tag in [('aligned_150', 'r2_150'), ('gigpo_150', 'gigpo_150')]:
        source = OLD / 'jobs' / original_tag
        proof = read(source / 'verification.json')
        assert proof['task_manifest_sha256'] == digest(ROOT / 'task_manifest.json')
        assert proof['episodes'] == 1536 and proof['unique_tasks'] == 512
        for name, expected in proof['source_hashes'].items():
            assert digest(source / name) == expected, name
        shutil.copytree(source, ROOT / 'jobs' / tag)
        references[tag] = dict(original_path=str(source), original_record_tag=original_tag,
                               verification_sha256=digest(source / 'verification.json'),
                               raw_source_hashes=proof['source_hashes'], new_inference=False)
        shared.update({str(p): digest(p) for p in source.glob('*') if p.is_file()})
    paths = list(ROOT.glob('*.py')) + [ROOT/'launch.sh', ROOT/'task_manifest.json', R2/'config.json']
    frozen = {str(p): digest(p) for p in paths}
    frozen.update(shared)
    card = {key: old[key] for key in [
        'panel_tasks', 'replicate_seeds', 'batch_size', 'max_steps', 'temperature', 'top_p',
        'top_k', 'history_length', 'max_prompt_length', 'max_response_length',
        'tensor_parallel_size', 'gpu_indices', 'gpu_memory_utilization', 'seed_protocol', 'analysis']}
    card.update(schema='webshop-polarity-reused-panel-v1', frozen_at=now(),
        authorization='User selected aligned complete115 continuation to150 with existing fixed512x3 panel.',
        candidate_run=R2.name, checkpoints={'r2_150': checkpoint}, evaluation_order=['r2_150'],
        reference_tags=['aligned_150', 'gigpo_150'],
        expected_record_tags={'r2_150': 'r2_150', 'aligned_150': 'r2_150', 'gigpo_150': 'gigpo_150'},
        original_card_sha256=digest(OLD/'experiment_card.json'),
        task_manifest_sha256=digest(ROOT/'task_manifest.json'), reference_reuse=references,
        primary='Intervention150 minus original aligned150 and GiGPO150 task-averaged success.',
        secondary='Taskscore, format-invalid counts and rates, option loops, timeouts, failed purchases.',
        targets=dict(format_invalid_count_lt=50, format_invalid_rate_lt=.006, option_loop_count=0,
                     timeout_rate_le=.04, success_count_gt=1085),
        budget=dict(new_evaluation_episodes=1536, reference_reused_episodes=3072,
                    max_wall_hours=3, max_two_gpu_hours=6),
        stops=['All1536new episodes verified', 'Environment/inference/integrity error', '3hour wall cap'],
        limitation='Single training seed; previously used development panel; combined intervention, no component causal attribution.',
        resource_preflight=inventory, frozen_files=frozen)
    write(ROOT/'experiment_card.json', card)
    print('PANEL_FROZEN:512tasks x3replicates; final150 only; two references reused', flush=True)


if __name__ == '__main__':
    freeze()
