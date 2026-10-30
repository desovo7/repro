"""Bind observed CPU evidence to a prepared run; later phases need their predecessor."""
import argparse
import hashlib
import json
from pathlib import Path


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cpu-result', type=Path, required=True)
    parser.add_argument('--native-flow', type=Path, required=True)
    parser.add_argument('--evidence-root', type=Path, required=True)
    args = parser.parse_args()
    card = json.loads(args.config.with_suffix('.card.json').read_text())
    contract = json.loads(args.config.with_suffix('.contract.json').read_text())
    assert card['phase'] == 'tier1a', 'Later phases need complete same-host predecessor evidence'
    cpu = json.loads(args.cpu_result.read_text())
    flow = json.loads(args.native_flow.read_text())
    assert cpu['status'] == 'passed' and not any(cpu[k] for k in ['failures', 'errors', 'skipped'])
    assert cpu['release_identity'] == card['release_identity']
    assert cpu['config_sha256'] == card['config_sha256']
    assert flow['collector']['status'] == flow['ray_entrypoint']['status'] == 'passed'
    assert not cpu['cuda_initialized']
    for name, digest in cpu['sources'].items():
        assert contract['release_files'][name] == digest, 'CPU tests ran different source'
    required_py = {name for name in contract['release_files'] if name.endswith('.py')}
    assert required_py == set(cpu['sources']), 'CPU source manifest incomplete'
    assert flow['source_sha256'] == cpu['sources'], 'Native entrypoint probe ran different source'
    assert flow['config_sha256'] == card['config_sha256'], 'Native entrypoint probe used another config'
    root = args.evidence_root
    paths = dict(specification=root/'implementation_spec.v831.original.md',
                 spec_errata=root/'implementation_errata.v831.json', step0_report=root/'step0_training_audit.json',
                 rule_audit=root/'rule_audit.json', native_cpu_checks=args.cpu_result,
                 native_flow=args.native_flow, cpu_cost_report=root/'v831_decisions.json')
    evidence = {name:record(path) for name, path in paths.items()}
    assert evidence['specification']['sha256'] == contract['spec_sha256']
    assert json.loads(paths['step0_report'].read_text())['status'] == 'complete'
    gate = dict(status='PASSED', config_sha256=card['config_sha256'],
                spec_errata_resolved=True, step0_complete=True, real_trace_rules_passed=True,
                tier2_rules_passed=True, native_cpu_checks_passed=True,
                cpu_cost_accepted=True, preceding_phase_passed=True, evidence_files=evidence,
                limits='CPU acceptance only; live two-rank actor updates remain to be verified after launch')
    destination = args.config.with_suffix('.gate.json')
    if destination.exists(): raise FileExistsError(destination)
    destination.write_text(json.dumps(gate, indent=2)+'\n')
    print(json.dumps(dict(path=str(destination), status=gate['status'])))


if __name__ == '__main__': main()
