"""Keep the three best validation checkpoints and the latest complete state.

The native FSDP manager may rotate its chronological checkpoints.  This store
owns separate hard links to completed files, so native unlinking cannot remove
the selected model, optimizer, RNG, data-loader or native driver state.  Completed source
directories must never be overwritten; a rewind uses a fresh output directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid
from datetime import datetime, timezone


METRIC = 'val/success_rate'
SCHEMA = 'gigpo_native_best_checkpoints_v1'
KEEP_BEST = 3
IDENTITY_KEYS = ('run_id', 'config_hash', 'model_contract_hash')
REQUIRED = {
    'NATIVE_COMPLETE.json', 'native_runtime.json', 'native_driver_rng.pt', 'data.pt',
    'actor/config.json', 'actor/generation_config.json', 'actor/tokenizer.json',
    'actor/tokenizer_config.json', 'actor/special_tokens_map.json',
    'actor/vocab.json', 'actor/merges.txt',
} | {f'actor/{kind}_world_size_2_rank_{rank}.pt'
     for kind in ('model', 'optim', 'extra_state') for rank in (0, 1)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_bytes(path, data):
    """Publish one complete file; fsync files and containing directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    write_bytes(path, (json.dumps(value, ensure_ascii=False, sort_keys=True,
                                 indent=2, allow_nan=False) + '\n').encode('utf-8'))


def validation_score(metrics):
    if METRIC not in metrics or isinstance(metrics[METRIC], bool):
        raise ValueError('Best-checkpoint retention requires ' + METRIC)
    score = float(metrics[METRIC])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Validation success rate must be finite and in [0, 1]')
    return score


class CheckpointRetention:
    """Single trainer-writer, atomic manifest, explicit same-score tie policy."""

    def __init__(self, checkpoint_root, identity):
        self.checkpoint_root = Path(checkpoint_root).resolve()
        self.root = self.checkpoint_root / 'best_checkpoints'
        self.index = self.root / 'index.json'
        self.identity = {key: identity[key] for key in IDENTITY_KEYS}
        self.root.mkdir(parents=True, exist_ok=True)
        self.state = self._load()
        self._cleanup_unpublished()
        self._recover_completed_saves()

    def _load(self):
        if not self.index.exists():
            return dict(schema=SCHEMA, metric=METRIC, keep_best=KEEP_BEST,
                        tie_break='later iteration first', identity=self.identity,
                        best=[], latest=None, history=[])
        state = json.loads(self.index.read_text())
        if (state.get('schema') != SCHEMA or state.get('identity') != self.identity
                or state.get('keep_best') != KEEP_BEST or state.get('metric') != METRIC):
            raise ValueError('Best-checkpoint retention identity or policy mismatch')
        scores = [(row['step'], row['score']) for row in state['history'] if row['score'] is not None]
        expected = sorted(scores, key=lambda row: (-row[1], -row[0]))[:KEEP_BEST]
        if [(row['step'], row['score']) for row in state['best']] != expected:
            raise ValueError('Best-checkpoint ranking does not match validation history')
        for row in self._protected(state):
            self._check_archive(row['step'])
        return state

    @staticmethod
    def _protected(state):
        rows = {row['step']: row for row in state['best']}
        if state['latest'] is not None:
            rows[state['latest']['step']] = state['latest']
        return list(rows.values())

    def _discard_archive(self, step):
        """Unpublish atomically before deleting files from an obsolete archive.

        A crash during rmtree can leave a partial .trash directory, but never a
        partial global_step directory that recovery could mistake for a save.
        Trash is deliberately ignored by recovery and is not a resume source.
        """
        if step in {row['step'] for row in self._protected(self.state)}:
            raise ValueError('Cannot discard a checkpoint referenced by the committed index')
        self._check_archive(step)
        target = self.root / f'global_step_{step}'
        trash = self.root / f'.trash-global_step_{step}-{uuid.uuid4().hex}'
        target.rename(trash)
        sync_directory(self.root)
        shutil.rmtree(trash)
        sync_directory(self.root)

    def _cleanup_unpublished(self):
        # This store has one trainer writer. These names can only be left by a
        # previous interrupted capture/delete; none is a published checkpoint.
        for path in sorted(self.root.iterdir()):
            if re.fullmatch(r'\.pending-global_step_\d+-[0-9a-f]{32}', path.name):
                if path.is_symlink() or not path.is_dir():
                    raise ValueError('Unexpected pending-checkpoint object')
                trash = self.root / ('.trash-' + path.name[1:])
                path.rename(trash)
                sync_directory(self.root)
                shutil.rmtree(trash)
                sync_directory(self.root)
            elif re.fullmatch(r'\.trash-(?:global_step_\d+-[0-9a-f]{32}|pending-global_step_\d+-[0-9a-f]{32})', path.name):
                if path.is_symlink() or not path.is_dir():
                    raise ValueError('Unexpected discarded-checkpoint object')
                shutil.rmtree(path)
                sync_directory(self.root)

    def _tracker_info(self):
        tracker = self.checkpoint_root / 'latest_checkpointed_iteration.txt'
        if not tracker.exists():
            return dict(status='absent', step=None)
        raw = tracker.read_bytes()
        info = dict(status='invalid', step=None, sha256=hashlib.sha256(raw).hexdigest(),
                    content=raw.decode('utf-8', errors='replace')[:256])
        try:
            step = int(raw.decode('utf-8').strip())
            if step <= 0:
                raise ValueError('nonpositive iteration')
            info.update(status='valid_integer', step=step)
        except (UnicodeError, ValueError):
            pass
        return info

    def _scan_native(self):
        rows = []
        for folder in sorted(self.checkpoint_root.glob('global_step_*')):
            if not re.fullmatch(r'global_step_[1-9]\d*', folder.name):
                continue
            step = int(folder.name.removeprefix('global_step_'))
            row = dict(step=step, path=str(folder), kind='native')
            try:
                files = self._check_complete(folder, step)
                row.update(status='complete', file_count=len(files),
                           completion_sha256=digest(folder / 'NATIVE_COMPLETE.json'))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row.update(status='incomplete_or_invalid', error=f'{type(exc).__name__}: {exc}')
            rows.append(row)
        return rows

    def resolve_resume(self, mode, explicit_path=None):
        """Choose before loading; explicit paths never silently fall back."""
        decision = dict(identity=self.identity, created_at=datetime.now(timezone.utc).isoformat(),
                        requested_mode=mode, requested_path=str(explicit_path) if explicit_path else None,
                        tracker_before=self._tracker_info(), native_candidates=self._scan_native(),
                        selected=None, status='preparing')
        path = self.root / 'resume_decisions' / (uuid.uuid4().hex + '.json')
        decision['record_path'] = str(path)
        try:
            if mode not in {'disable', 'auto', 'resume_path'}:
                raise ValueError('Unsupported checkpoint resume mode')
            evidence = (bool(decision['native_candidates']) or bool(self.state['history'])
                        or decision['tracker_before']['status'] != 'absent'
                        or any((self.root / 'validation_scores').glob('iteration_*.json')))
            if mode == 'disable':
                if evidence:
                    raise ValueError('resume_mode=disable requires a fresh output directory')
                decision['status'] = 'fresh_start'
            elif mode == 'resume_path':
                if not explicit_path:
                    raise ValueError('An explicit resume checkpoint path is required')
                folder = Path(explicit_path).absolute()
                if not re.fullmatch(r'global_step_[1-9]\d*', folder.name):
                    raise ValueError('Explicit resume path must identify one global_step directory')
                step = int(folder.name.removeprefix('global_step_'))
                self._check_complete(folder, step)
                if folder.parent.resolve() == self.root:
                    self._check_archive(step)
                self.validate_resume_step(step)
                decision['selected'] = dict(step=step, path=str(folder), kind='explicit')
                decision['status'] = 'ready'
            else:
                candidates = [row for row in decision['native_candidates'] if row['status'] == 'complete']
                for row in self._protected(self.state):
                    receipt = self._check_archive(row['step'])
                    candidates.append(dict(step=row['step'], path=str(self.root / f"global_step_{row['step']}"),
                                           kind='protected', completion_sha256=receipt['completion_sha256']))
                if not candidates:
                    if evidence:
                        raise ValueError('No complete checkpoint available; refusing an implicit fresh start')
                    decision['status'] = 'fresh_start'
                else:
                    chosen = max(candidates, key=lambda row: (row['step'], row['kind'] == 'native'))
                    same_step = [row for row in candidates if row['step'] == chosen['step']]
                    if len({row['completion_sha256'] for row in same_step}) > 1:
                        raise ValueError('Conflicting complete checkpoints at the same iteration')
                    self.validate_resume_step(chosen['step'])
                    decision['selected'] = chosen
                    decision['selection_rule'] = 'highest complete iteration; matching native preferred over protected copy'
                    decision['status'] = 'ready'
            write_json(path, decision)
            return decision
        except Exception as exc:
            decision.update(status='rejected', error=f'{type(exc).__name__}: {exc}')
            write_json(path, decision)
            raise

    def prepare_native_resume(self, decision):
        """Make native's preliminary tracker read safe without changing its model choice."""
        selected = decision['selected']
        if selected is None:
            return
        # A failed save can leave a score and partial directory ahead of the
        # chosen complete state. Preserve those failed-attempt artifacts before
        # replaying; the same iteration must be free to acquire a new score.
        quarantine = []
        for row in self._scan_native():
            if row['step'] > selected['step']:
                if row['status'] == 'complete':
                    raise ValueError('A newer complete checkpoint appeared during recovery preparation')
                quarantine.append((Path(row['path']), 'checkpoints/' + Path(row['path']).name, row['step']))
        for score_file in (self.root / 'validation_scores').glob('iteration_*.json'):
            step = int(score_file.stem.removeprefix('iteration_'))
            if step > selected['step']:
                quarantine.append((score_file, 'validation_scores/' + score_file.name, step))
        interrupted = Path(decision['record_path']).with_suffix('.interrupted')
        decision['interrupted_artifacts'] = [dict(source=str(source), destination=str(interrupted / relative), step=step)
                                             for source, relative, step in quarantine]
        decision['replay_after_step'] = selected['step']
        decision['trace_tail_handling'] = 'Recorded here; training trace/log segmentation remains the recovery workflow responsibility'
        write_json(decision['record_path'], decision)
        for source, relative, step in quarantine:
            target = interrupted / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            sync_directory(source.parent)
            sync_directory(target.parent)
        tracker = self.checkpoint_root / 'latest_checkpointed_iteration.txt'
        folder = Path(selected['path'])
        direct_native = folder.parent.resolve() == self.checkpoint_root
        raw = tracker.read_bytes() if tracker.exists() else None
        expected = str(selected['step']).encode()
        action = 'unchanged'
        if direct_native and raw != expected:
            action = 'atomically point to verified complete native checkpoint'
        elif not direct_native and self._tracker_info()['status'] == 'invalid':
            action = 'remove invalid tracker before explicit protected-path loading'
        if action != 'unchanged' and raw is not None:
            backup = Path(decision['record_path']).with_suffix('.tracker_before.bin')
            write_bytes(backup, raw)
            decision['tracker_backup'] = str(backup)
        decision['tracker_action'] = action
        write_json(decision['record_path'], decision)
        if direct_native and raw != expected:
            write_bytes(tracker, expected)
        elif action.startswith('remove invalid'):
            tracker.unlink()
            sync_directory(self.checkpoint_root)

    @staticmethod
    def finish_resume(decision, *, error=None):
        decision['status'] = 'failed' if error is not None else 'loaded' if decision['selected'] else 'fresh_start'
        if error is not None:
            decision['error'] = f'{type(error).__name__}: {error}'
        write_json(decision['record_path'], decision)

    def validate_resume_step(self, step):
        latest = self.state['latest']
        if latest is not None and int(step) < latest['step']:
            raise ValueError('Rewinding a completed branch requires a fresh checkpoint output directory')

    def qualifies(self, step, score):
        score = validation_score({METRIC: score})
        candidates = [row for row in self.state['best'] if row['step'] != int(step)]
        candidates.append(dict(step=int(step), score=score))
        selected = sorted(candidates, key=lambda row: (-row['score'], -row['step']))[:KEEP_BEST]
        return any(row['step'] == int(step) for row in selected)

    def assert_new_destination(self, folder):
        folder = Path(folder)
        if folder.parent.resolve() != self.checkpoint_root:
            raise ValueError('Checkpoint destination differs from the configured root')
        if (folder / 'NATIVE_COMPLETE.json').exists():
            raise FileExistsError('Never overwrite a completed checkpoint; use a new branch directory: ' + str(folder))
        if folder.exists() and any(p.is_file() and p.stat().st_nlink > 1 for p in folder.rglob('*')):
            raise FileExistsError('Never overwrite files linked into an existing checkpoint capture')

    def record_validation(self, step, score):
        """Durably bind the actual score before any native checkpoint writes."""
        step = int(step)
        if step <= 0:
            raise ValueError('Initial validation is not a trainable checkpoint')
        score = validation_score({METRIC: score})
        path = self.root / 'validation_scores' / f'iteration_{step}.json'
        record = dict(identity=self.identity, step=step, metric=METRIC, score=score)
        if path.exists():
            if json.loads(path.read_text()) != record:
                raise ValueError('Repeated evaluation cannot replace the recorded selection score')
        else:
            write_json(path, record)
        return score

    def score_for(self, step):
        path = self.root / 'validation_scores' / f'iteration_{step}.json'
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        if (record.get('identity') != self.identity or record.get('step') != step
                or record.get('metric') != METRIC):
            raise ValueError('Saved validation score identity mismatch')
        return validation_score({METRIC: record['score']})

    def _recover_completed_saves(self):
        """Reconcile a crash after native completion or archive rename.

        Validation scores are persisted first, and archive receipts last.  A
        pending directory is never selected.  Recovery does not rerun validation
        or substitute a later stochastic score for the recorded one.
        """
        history = {row['step']: row for row in self.state['history']}
        archives = []
        for target in sorted(self.root.glob('global_step_*')):
            step = int(target.name.removeprefix('global_step_'))
            self._check_archive(step)
            score = self.score_for(step)
            old = history.get(step)
            if old is not None and old['score'] is not None and old['score'] != score:
                raise ValueError('Protected checkpoint lost its original selection score')
            history[step] = dict(step=step, score=score)
            archives.append(step)
        if archives:
            rows = sorted(history.values(), key=lambda row: row['step'])
            ranked = sorted((row for row in rows if row['score'] is not None),
                            key=lambda row: (-row['score'], -row['step']))[:KEEP_BEST]
            latest_step = max(archives + ([self.state['latest']['step']] if self.state['latest'] else []))
            new = dict(self.state, history=rows, best=ranked, latest=history[latest_step])
            for row in self._protected(new):
                self._check_archive(row['step'])
            if new != self.state:
                write_json(self.index, new)
                self.state = new
            retained = {row['step'] for row in self._protected(self.state)}
            for step in archives:
                if step not in retained:
                    self._discard_archive(step)
        # Scores and complete native directories discover a save whose pointer
        # was truncated or whose archive/index had not yet been published.
        for path in (self.root / 'validation_scores').glob('iteration_*.json'):
            step = int(path.stem.removeprefix('iteration_'))
            self.score_for(step)
        for candidate in sorted(self._scan_native(), key=lambda row: row['step']):
            step = candidate['step']
            latest = self.state['latest']['step'] if self.state['latest'] else 0
            known = {row['step'] for row in self.state['history']}
            if candidate['status'] == 'complete' and step not in known and step >= latest:
                self.preserve(Path(candidate['path']), self.score_for(step))

    def _check_complete(self, folder, step):
        folder = Path(folder)
        if folder.name != f'global_step_{step}' or folder.is_symlink():
            raise ValueError('Unexpected checkpoint directory')
        files = {}
        for path in sorted(folder.rglob('*')):
            if path.is_symlink():
                raise ValueError('Checkpoint files must not be symbolic links')
            if path.is_file() and path.name != 'RETENTION_COMPLETE.json':
                stat = path.stat()
                files[str(path.relative_to(folder))] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if not REQUIRED.issubset(files) or any(files[name]['size'] <= 0 for name in REQUIRED):
            raise ValueError('Best checkpoint needs both model/optimizer/RNG ranks and all runtime files')
        marker = json.loads((folder / 'NATIVE_COMPLETE.json').read_text())
        if marker.get('iteration_id') != step or any(marker.get(key) != val for key, val in self.identity.items()):
            raise ValueError('Completed checkpoint identity mismatch')
        runtime = json.loads((folder / 'native_runtime.json').read_text())
        if any(marker.get(key) != val for key, val in runtime.items()):
            raise ValueError('Completion marker differs from the saved runtime')
        for name, key in [('native_runtime.json', 'runtime_sha256'), ('native_driver_rng.pt', 'rng_sha256')]:
            if digest(folder / name) != marker[key]:
                raise ValueError('Completed checkpoint digest mismatch: ' + name)
        if 'gradus_state' not in marker or marker['gradus_state']['last_iteration'] != step:
            raise ValueError('Checkpoint GRADUS history is incomplete')
        return files

    def _check_archive(self, step):
        target = self.root / f'global_step_{step}'
        receipt = json.loads((target / 'RETENTION_COMPLETE.json').read_text())
        if receipt.get('identity') != self.identity or receipt.get('step') != step:
            raise ValueError('Protected checkpoint receipt identity mismatch')
        if self._check_complete(target, step) != receipt['files']:
            raise ValueError('Protected checkpoint files changed after capture')
        if digest(target / 'NATIVE_COMPLETE.json') != receipt['completion_sha256']:
            raise ValueError('Protected checkpoint completion marker changed')
        return receipt

    def preserve(self, folder, score=None):
        """Protect a complete save; None never enters the validation ranking."""
        folder = Path(folder)
        step = int(folder.name.removeprefix('global_step_'))
        if step <= 0:
            raise ValueError('Initial validation has no trainable checkpoint')
        if score is not None:
            score = self.record_validation(step, score)
        else:
            score = self.score_for(step)
        if folder.parent.resolve() != self.checkpoint_root:
            raise ValueError('Only checkpoints from this output directory can be registered')
        self.validate_resume_step(step)
        old = next((row for row in self.state['history'] if row['step'] == step), None)
        if old is not None and old['score'] is not None and score is not None and old['score'] != score:
            raise ValueError('Repeated evaluation cannot replace the recorded selection score')
        score = old['score'] if old is not None and score is None else score
        files = self._check_complete(folder, step)
        target = self.root / folder.name
        if target.exists():
            receipt = self._check_archive(step)
            if receipt['files'] != files or receipt['completion_sha256'] != digest(folder / 'NATIVE_COMPLETE.json'):
                raise ValueError('A different checkpoint already owns this protected iteration')
        else:
            staging = self.root / ('.pending-' + folder.name + '-' + uuid.uuid4().hex)
            try:
                shutil.copytree(folder, staging, copy_function=os.link)
                for name, meta in files.items():
                    source, copied = folder / name, staging / name
                    before, after = source.stat(), copied.stat()
                    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                        raise ValueError('Checkpoint preservation did not create a hard link')
                    if (before.st_size, before.st_mtime_ns) != (meta['size'], meta['mtime_ns']):
                        raise ValueError('Checkpoint changed while being preserved')
                receipt = dict(schema=SCHEMA, identity=self.identity, step=step, source=str(folder),
                               completion_sha256=digest(folder / 'NATIVE_COMPLETE.json'), files=files,
                               preservation='hard links to completed immutable files')
                write_json(staging / 'RETENTION_COMPLETE.json', receipt)
                staging.rename(target)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        history = [row for row in self.state['history'] if row['step'] != step] + [dict(step=step, score=score)]
        history.sort(key=lambda row: row['step'])
        ranked = sorted((row for row in history if row['score'] is not None),
                        key=lambda row: (-row['score'], -row['step']))[:KEEP_BEST]
        new = dict(self.state, history=history, best=ranked, latest=dict(step=step, score=score))
        # New archive exists before publishing. Old selections remain available
        # until the atomic manifest commits, including after partial failures.
        for row in self._protected(new):
            self._check_archive(row['step'])
        previous = self._protected(self.state)
        write_json(self.index, new)
        self.state = new
        retained = {row['step'] for row in self._protected(new)}
        for row in previous:
            if row['step'] not in retained:
                self._discard_archive(row['step'])
        return new
