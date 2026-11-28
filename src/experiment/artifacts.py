"""Validate and normalize prediction archives before cross-module evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np


def read_predictions(path):
    with np.load(path, allow_pickle=False) as archive:
        keys = [key for key in ('mc_probs', 'probs_mc', 'probs', 'logits') if key in archive]
        if len(keys) != 1:
            raise ValueError('exactly one prediction key required: mc_probs/probs_mc/probs/logits')
        key = keys[0]
        values = archive[key]
        expected_ndim = 3 if key in ('mc_probs', 'probs_mc') else 2
        if values.ndim != expected_ndim or not all(values.shape) or values.shape[-1] < 2:
            raise ValueError('invalid prediction dimensions')
        if values.dtype.kind not in 'fi' or not np.isfinite(values).all():
            raise ValueError('predictions must be finite numbers')
        if key != 'logits':
            if np.any((values < 0) | (values > 1)) or not np.allclose(values.sum(-1), 1, atol=1e-6, rtol=0):
                raise ValueError('invalid probability distributions')
        n = values.shape[-2]
        result = {'mc_probs' if key == 'probs_mc' else key: values.copy()}
        if 'labels' in archive:
            y = archive['labels']
            if y.shape != (n,) or y.dtype.kind not in 'iu' or np.any((y < 0) | (y >= values.shape[-1])):
                raise ValueError('labels must be integer class indices [N]')
            result['labels'] = y.copy()
        if 'sample_ids' in archive:
            ids = archive['sample_ids']
            if ids.shape != (n,) or ids.dtype.kind not in 'US' or len(np.unique(ids)) != n:
                raise ValueError('sample_ids must be unique strings [N]')
            if np.any(np.char.str_len(ids.astype(str)) == 0):
                raise ValueError('sample_ids cannot be empty')
            result['sample_ids'] = ids.astype(str)
        return result


def normalize_archive(source, target):
    """Rename legacy key only; never fabricate labels or sample identifiers."""
    result = read_predictions(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as handle:
        np.savez_compressed(handle, **result)
    return {'output': str(target), 'keys': sorted(result),
            'identity_verified': False, 'status': 'format_validated_only'}


def validate_pair(calibration, test):
    cal, test = read_predictions(calibration), read_predictions(test)
    for item in (cal, test):
        if 'labels' not in item or 'sample_ids' not in item:
            raise ValueError('calibration/test require labels and globally stable sample_ids')
    if set(cal['sample_ids']) & set(test['sample_ids']):
        raise ValueError('calibration/test overlap detected')
    ck = next(k for k in ('mc_probs', 'probs', 'logits') if k in cal)
    tk = next(k for k in ('mc_probs', 'probs', 'logits') if k in test)
    if ck != tk or cal[ck].shape[-1] != test[tk].shape[-1]:
        raise ValueError('calibration/test representations and class counts must match')
    return {'calibration_examples': len(cal['labels']), 'test_examples': len(test['labels']),
            'overlap': 0, 'class_order_verified': False,
            'note': 'Check dataset, ordered class_names, checkpoint and split manifest separately.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    convert = sub.add_parser('normalize')
    convert.add_argument('source')
    convert.add_argument('target')
    pair = sub.add_parser('check-pair')
    pair.add_argument('calibration')
    pair.add_argument('test')
    args = parser.parse_args(argv)
    try:
        result = (normalize_archive(args.source, args.target) if args.command == 'normalize'
                  else validate_pair(args.calibration, args.test))
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, EOFError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
