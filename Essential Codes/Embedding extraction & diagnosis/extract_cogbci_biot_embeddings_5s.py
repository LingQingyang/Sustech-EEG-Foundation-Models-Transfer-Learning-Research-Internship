"""
CogBCI BIOT embedding extraction from the flat 5-second H5 files produced by
flow_process_cogbci_onesession.py (e.g. /mnt/dataset4/DATASETS/COGBCI/processed_5s_new_ses{1,2,3}).

Replicates the output schema of extract_cogbci_biot_embeddings.ipynb, plus a
per-sample `session` field so multiple sessions can be merged into one file:
    signal / embedding / trial_id / task_id / subject_id / session / label / rest_label / electrodes

Key adaptations vs. the notebook:
- Input is already segmented into 5 s windows (N, 62, 1000); no task_bounds needed.
- Multiple session directories can be passed; the per-sample `session` value is read
  from the `session` dataset of each source H5 (1/2/3 = ses-S1/S2/S3).
- Each sample carries montage_type (0=V1, 1=V2). The 58 common channels are selected
  per-sample using the ch_names_v1 / ch_names_v2 attributes of the source H5, so all
  output signals share the identical real-electrode set (no interpolation).
- label = task_id * 10 + local_label (resting=0, nback=1, matb=2), rest_label is the
  local label for resting samples and -1 otherwise. trial_id = file_idx of the source run.

Usage:
    python extract_cogbci_biot_embeddings_5s.py --smoke            # one-batch shape test
    python extract_cogbci_biot_embeddings_5s.py --limit 256       # small end-to-end test
    python extract_cogbci_biot_embeddings_5s.py                   # full extraction (ses1+2+3)
    python extract_cogbci_biot_embeddings_5s.py --input-dir /path/ses1   # single session
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x

# =============================================================================
# 1. Configuration
# =============================================================================
MODEL_NAME = 'BIOT'
MODEL_DIR = Path('/mnt/dataset4/fuxy/FMS/FM/BIOT')
CHECKPOINT_PATH = MODEL_DIR / 'pretrained-models/EEG-six-datasets-18-channels.ckpt'

DEFAULT_INPUT_DIRS = [
    '/mnt/dataset4/DATASETS/COGBCI/processed_5s_new_ses1',
    '/mnt/dataset4/DATASETS/COGBCI/processed_5s_new_ses2',
    '/mnt/dataset4/DATASETS/COGBCI/processed_5s_new_ses3',
]
DEFAULT_OUTPUT_H5 = MODEL_DIR / 'output_emb_5s/cogbci_allsessions_5sflat_embeddings.h5'

# task file stem -> task_id; label = task_id * 10 + local_label
TASK_FILES = {'resting': 0, 'nback': 1, 'matb': 2}

COGBCI_COMMON_CHANNELS_58 = [
    'Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP1', 'Pz',
    'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2', 'FCz',
    'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz',
    'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8',
    'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2',
]

BIOT_N_FFT = 200
BIOT_HOP_LENGTH = 100
BIOT_EMB_SIZE = 256
BIOT_HEADS = 8
BIOT_DEPTH = 4

ALLOWED_H5_KEYS = {'signal', 'embedding', 'trial_id', 'task_id',
                   'subject_id', 'session', 'label', 'rest_label', 'electrodes'}

CHANNEL_ALIASES = {
    'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'FCZ': 'FCz',
    'CZ': 'Cz', 'CPZ': 'CPz', 'PZ': 'Pz', 'POZ': 'POz', 'OZ': 'Oz',
}


def decode_str(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def canonicalize_channel_name(name):
    name = decode_str(name).strip()
    return CHANNEL_ALIASES.get(name.upper(), name)


def select_common58_indices(raw_channels, common_channels):
    """Return indices selecting common_channels (in order) from raw_channels."""
    raw_to_index = {canonicalize_channel_name(ch): i for i, ch in enumerate(raw_channels)}
    indices, missing = [], []
    for ch in common_channels:
        ch = canonicalize_channel_name(ch)
        if ch not in raw_to_index:
            missing.append(ch)
        else:
            indices.append(raw_to_index[ch])
    if missing:
        raise ValueError(f'Montage is missing common58 channels: {missing}')
    return np.asarray(indices, dtype=np.int64)


def get_requested_device(device):
    requested = torch.device(device)
    if requested.type == 'cuda' and not torch.cuda.is_available():
        warnings.warn(f'Requested {device}, but CUDA is unavailable. Falling back to CPU.')
        return torch.device('cpu')
    return requested


# =============================================================================
# 2. Dataset over the flat 5 s H5 files
# =============================================================================
class CogBCI5sFlatDataset(Dataset):
    """
    Reads cogbci_{resting,nback,matb}_5s.h5 produced by flow_process_cogbci_onesession.py.
    Every record is one pre-cut 5 s window; channels are reduced from 62 to the
    common-58 set according to the sample's montage_type.
    """

    def __init__(self, input_dirs, tasks_to_use=('resting', 'nback', 'matb'), limit=None):
        if isinstance(input_dirs, (str, Path)):
            input_dirs = [input_dirs]
        self.input_dirs = [Path(d) for d in input_dirs]
        self.records = []           # (h5_path, row_index)
        self.record_meta = []       # dict per record
        self._handles = {}
        self._ch_idx_by_file = {}   # h5_path -> {montage_id: indices}
        self.electrodes = list(COGBCI_COMMON_CHANNELS_58)

        for input_dir in self.input_dirs:
            for task_name, task_id in TASK_FILES.items():
                if task_name not in tasks_to_use:
                    continue
                h5_path = input_dir / f'cogbci_{task_name}_5s.h5'
                if not h5_path.exists():
                    print(f'  [Skip] missing {h5_path}')
                    continue
                with h5py.File(h5_path, 'r') as h5:
                    labels = h5['labels'][:].astype(int)
                    subjects = h5['subject'][:].astype(int)
                    sessions = h5['session'][:].astype(int)
                    montages = h5['montage_type'][:].astype(int)
                    file_idx = h5['file_idx'][:].astype(int)

                    for montage_id, attr in ((0, 'ch_names_v1'), (1, 'ch_names_v2')):
                        chs = [canonicalize_channel_name(x) for x in h5.attrs[attr]]
                        sel = select_common58_indices(chs, self.electrodes)
                        # sanity: selected names must equal the common58 list exactly
                        assert [chs[i] for i in sel] == self.electrodes, \
                            f'{h5_path.name} montage {montage_id}: common58 selection mismatch'
                        self._ch_idx_by_file.setdefault(str(h5_path), {})[montage_id] = sel

                for row in range(len(labels)):
                    local = int(labels[row])
                    self.records.append((str(h5_path), row))
                    self.record_meta.append({
                        'trial_id': int(file_idx[row]),
                        'task_id': task_id,
                        'subject_id': int(subjects[row]),
                        'session': int(sessions[row]),
                        'label': task_id * 10 + local,
                        'rest_label': local if task_id == 0 else -1,
                        'montage': int(montages[row]),
                    })

        if limit is not None:
            self.records = self.records[:limit]
            self.record_meta = self.record_meta[:limit]
        if not self.records:
            raise RuntimeError(f'No windows indexed under {self.input_dirs}')

    def __len__(self):
        return len(self.records)

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles.clear()

    def _get_handle(self, h5_path):
        if h5_path not in self._handles:
            self._handles[h5_path] = h5py.File(h5_path, 'r')
        return self._handles[h5_path]

    def __getitem__(self, index):
        h5_path, row = self.records[index]
        meta = self.record_meta[index]
        h5 = self._get_handle(h5_path)
        eeg62 = h5['eeg'][row]                       # (62, 1000) float32
        ch_idx = self._ch_idx_by_file[h5_path][meta['montage']]
        eeg = np.ascontiguousarray(eeg62[ch_idx])    # (58, 1000)
        return {
            'eeg': torch.from_numpy(eeg),
            'trial_id': meta['trial_id'],
            'task_id': meta['task_id'],
            'subject_id': meta['subject_id'],
            'session': meta['session'],
            'label': meta['label'],
            'rest_label': meta['rest_label'],
        }


# =============================================================================
# 3. BIOT extractor (identical to the notebook)
# =============================================================================
class BIOTEmbeddingExtractor(nn.Module):
    embedding_source = ('BIOTEncoder.forward; output after STFT tokenization, '
                        'LinearAttentionTransformer, and mean pooling')

    def __init__(self, repo, checkpoint_path, n_channels, device='cuda:0'):
        super().__init__()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from model import BIOTEncoder
        self.device = get_requested_device(device)
        self.model = BIOTEncoder(
            emb_size=BIOT_EMB_SIZE, heads=BIOT_HEADS, depth=BIOT_DEPTH,
            n_channels=n_channels, n_fft=BIOT_N_FFT, hop_length=BIOT_HOP_LENGTH,
        ).to(self.device)
        self.skipped_checkpoint_keys = []
        if checkpoint_path is not None:
            try:
                state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            except TypeError:
                state = torch.load(checkpoint_path, map_location='cpu')
            model_state = self.model.state_dict()
            filtered = {}
            for key, value in state.items():
                clean_key = key[len('module.'):] if key.startswith('module.') else key
                if clean_key in model_state and tuple(model_state[clean_key].shape) == tuple(value.shape):
                    filtered[clean_key] = value
                else:
                    self.skipped_checkpoint_keys.append(key)
            missing, unexpected = self.model.load_state_dict(filtered, strict=False)
            print(f'loaded checkpoint: {checkpoint_path}')
            print(f'loaded_keys={len(filtered)}, missing={len(missing)}, '
                  f'unexpected={len(unexpected)}, skipped={len(self.skipped_checkpoint_keys)}')
            if self.skipped_checkpoint_keys:
                print('skipped examples:', self.skipped_checkpoint_keys[:5])
        else:
            print('CHECKPOINT_PATH=None; using random initialization for shape smoke test only')
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, return_cpu=False):
        emb = self.model(x.to(self.device, non_blocking=True).float()).detach()
        return emb.cpu() if return_cpu else emb


# =============================================================================
# 4. H5 writing (same schema as the notebook)
# =============================================================================
def append_or_create_dataset(h5, name, data, compression=None):
    data = np.asarray(data)
    if name not in h5:
        maxshape = (None,) + data.shape[1:]
        h5.create_dataset(name, data=data, maxshape=maxshape, chunks=True, compression=compression)
    else:
        dset = h5[name]
        old_n = dset.shape[0]
        dset.resize((old_n + data.shape[0],) + dset.shape[1:])
        dset[old_n:] = data


def write_common_attrs(h5, dataset, extractor, input_dirs):
    sessions = sorted({m['session'] for m in dataset.record_meta})
    h5.attrs['model'] = MODEL_NAME
    h5.attrs['dataset'] = 'CogBCI'
    h5.attrs['source_h5_dirs'] = json.dumps([str(d) for d in input_dirs])
    h5.attrs['sessions'] = json.dumps(sessions)
    h5.attrs['session_field'] = 'session'
    h5.attrs['session_meaning'] = 'session is the per-sample recording session id (1/2/3 = ses-S1/S2/S3)'
    h5.attrs['window_seconds'] = 5.0
    h5.attrs['stride_seconds'] = 5.0
    h5.attrs['sfreq'] = 200
    h5.attrs['checkpoint_path'] = str(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else ''
    h5.attrs['model_input_format'] = 'B_C_T'
    h5.attrs['channel_selection_mode'] = 'common58'
    h5.attrs['raw_cogbci_n_channels'] = 62
    h5.attrs['selected_channel_count'] = len(dataset.electrodes)
    h5.attrs['selected_channels'] = json.dumps(dataset.electrodes)
    h5.attrs['channel_note'] = (
        'CogBCI has two 62-channel montages. This extraction selects the 58-channel '
        'intersection per sample according to montage_type, so all windows share the '
        'same real EEG channels without interpolation.'
    )
    h5.attrs['signal_note'] = 'signal is the exact model input EEG window aligned one-to-one with embedding.'
    h5.attrs['label_field'] = 'label'
    h5.attrs['label_meaning'] = 'label = task_id * 10 + local_label (task_id: resting=0, nback=1, matb=2)'
    h5.attrs['rest_label_field'] = 'rest_label'
    h5.attrs['rest_label_meaning'] = 'rest_label is local_label for Resting (0=EC, 1=EO), -1 for non-resting'
    h5.attrs['trial_id_meaning'] = 'trial_id = file_idx of the source .set file within its task'
    if hasattr(extractor, 'embedding_source'):
        h5.attrs['embedding_source'] = extractor.embedding_source
    if hasattr(extractor, 'skipped_checkpoint_keys'):
        h5.attrs['skipped_checkpoint_keys'] = ','.join(map(str, extractor.skipped_checkpoint_keys))


def validate_output_h5(output_h5):
    with h5py.File(output_h5, 'r') as h5:
        print('Saved:', output_h5)
        for k in sorted(h5.keys()):
            print(f'  {k}: {h5[k].shape}')
        keys = set(h5.keys())
        extra = keys - ALLOWED_H5_KEYS
        missing = ALLOWED_H5_KEYS - keys
        if extra:
            raise RuntimeError(f'Unexpected extra H5 keys: {extra}')
        if missing:
            raise RuntimeError(f'Missing required H5 keys: {missing}')
        assert h5['electrodes'].shape[0] == 58
        assert h5['signal'].shape[1] == 58
        n = h5['embedding'].shape[0]
        for k in ('signal', 'trial_id', 'task_id', 'subject_id', 'session', 'label', 'rest_label'):
            assert h5[k].shape[0] == n, f'{k} row count mismatch'
        print('unique task_id:', sorted(set(h5['task_id'][:].tolist())))
        print('unique session:', sorted(set(h5['session'][:].tolist())))
        print('unique label:', sorted(set(h5['label'][:].tolist())))


def extract_embeddings_to_h5(dataset, loader, extractor, output_h5, input_dirs):
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    if output_h5.exists():
        output_h5.unlink()
    with h5py.File(output_h5, 'w') as h5:
        write_common_attrs(h5, dataset, extractor, input_dirs)
        h5.create_dataset('electrodes',
                          data=np.asarray(dataset.electrodes, dtype=object),
                          dtype=h5py.string_dtype(encoding='utf-8'))
        n_written = 0
        for batch in tqdm(loader, desc=f'Extracting {MODEL_NAME} embeddings'):
            signal_np = batch['eeg'].numpy().astype(np.float32)
            embedding = extractor(batch['eeg'], return_cpu=True).numpy().astype(np.float32)
            append_or_create_dataset(h5, 'signal', signal_np, compression='gzip')
            append_or_create_dataset(h5, 'embedding', embedding, compression='gzip')
            append_or_create_dataset(h5, 'trial_id', batch['trial_id'].numpy().reshape(-1).astype(np.int64))
            append_or_create_dataset(h5, 'task_id', batch['task_id'].numpy().reshape(-1).astype(np.int64))
            append_or_create_dataset(h5, 'subject_id', batch['subject_id'].numpy().reshape(-1).astype(np.int64))
            append_or_create_dataset(h5, 'session', batch['session'].numpy().reshape(-1).astype(np.int16))
            append_or_create_dataset(h5, 'label', batch['label'].numpy().reshape(-1).astype(np.int16))
            append_or_create_dataset(h5, 'rest_label', batch['rest_label'].numpy().reshape(-1).astype(np.int16))
            n_written += signal_np.shape[0]
        h5.attrs['n_windows'] = n_written
    validate_output_h5(output_h5)
    return output_h5


# =============================================================================
# 5. Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='BIOT embedding extraction from flat CogBCI 5s H5 files')
    parser.add_argument('--input-dir', type=str, nargs='+', default=DEFAULT_INPUT_DIRS,
                        help='one or more session dirs (default: ses1 ses2 ses3)')
    parser.add_argument('--output-h5', type=str, default=str(DEFAULT_OUTPUT_H5))
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--limit', type=int, default=None, help='only process the first N windows (test)')
    parser.add_argument('--smoke', action='store_true', help='build dataset + one forward batch, no writing')
    args = parser.parse_args()

    dataset = CogBCI5sFlatDataset(args.input_dir, limit=args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())

    print('input dirs:', args.input_dir)
    print('windows:', len(dataset))
    print('sessions:', sorted({m['session'] for m in dataset.record_meta}))
    print('electrodes:', len(dataset.electrodes))
    sample = dataset[0]
    print('sample input shape:', tuple(sample['eeg'].shape))
    print('sample task_id/session/label/rest_label:',
          sample['task_id'], sample['session'], sample['label'], sample['rest_label'])

    extractor = BIOTEmbeddingExtractor(MODEL_DIR, CHECKPOINT_PATH,
                                       n_channels=len(dataset.electrodes), device=args.device)

    batch = next(iter(loader))
    with torch.no_grad():
        emb = extractor(batch['eeg'])
    print('smoke batch input:', tuple(batch['eeg'].shape), '-> embedding:', tuple(emb.shape))

    if args.smoke:
        print('Smoke test done; skipping extraction.')
        return

    out = extract_embeddings_to_h5(dataset, loader, extractor, args.output_h5, args.input_dir)
    print('Wrote:', out)
    dataset.close()


if __name__ == '__main__':
    main()
