"""Stage verified model files once and optionally create a PRIVATE Kaggle Dataset.

Runs on the host/CPU, never inside a GPU notebook. No model downloads occur here.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def prepare(manifest, source, destination, dataset, license_name):
    source, destination = source.resolve(), destination.resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', dataset):
        raise ValueError('dataset must be owner/slug')
    weights = manifest.get('weight_files') or manifest.get('weights')
    if not isinstance(weights, list) or not weights:
        raise ValueError('manifest must contain reviewed weight_files')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('destination must be empty')
    checked = []
    for item in weights:
        target = item.get('target') or item.get('filename')
        path = (source / target).resolve()
        if source not in path.parents or not path.is_file():
            raise ValueError(f'missing model or unsafe path: {target}')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != item['sha256'] or (item.get('size') and path.stat().st_size != item['size']):
            raise ValueError(f'model hash/size mismatch: {target}')
        checked.append((path, digest, target))
    destination.mkdir(parents=True, exist_ok=True)
    for path, digest, target in checked:
        dest = destination / digest
        if not dest.exists():
            try:
                os.link(path, dest)
            except OSError:
                shutil.copyfile(path, dest)
    (destination / 'dataset-metadata.json').write_text(json.dumps({
        'id': dataset, 'title': dataset.split('/')[1], 'licenses': [{'name': license_name}],
    }), encoding='utf-8')
    (destination / 'studio-models.json').write_text(json.dumps({
        'models': [{'target': target, 'sha256': digest} for _, digest, target in checked],
    }), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--staging-dir', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--license', required=True, help='Kaggle license identifier matching model redistribution rights')
    parser.add_argument('--upload', action='store_true', help='Create the private Dataset now; requires Kaggle credentials')
    args = parser.parse_args()
    prepare(json.loads(args.manifest.read_text(encoding='utf-8')), args.model_dir,
            args.staging_dir, args.dataset, args.license)
    if args.upload:
        # The CLI defaults to private. Never supply --public.
        subprocess.run([sys.executable, '-m', 'kaggle', 'datasets', 'create', '-p', str(args.staging_dir)], check=True)
    print('Prepared private Dataset model files. Do not edit source files until upload finishes.')


if __name__ == '__main__':
    main()
