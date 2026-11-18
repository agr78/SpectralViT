"""QSM slice datasets with volume caching and resampling, slice-level flattening, IXI and HCP
volume loaders, NIMH Data Archive export cleanup, synthetic point-distance volumes, and
class-balanced sampling."""
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import affine_transform, gaussian_filter
from util import robust_flatten
import pandas as pd
import re
import glob
import io

def get_slice_level_data(dataset, target_dim, include_unlabeled=False):
    all_imgs, all_clins, all_lbls, subj_map = [], [], [], []
    for i in range(len(dataset)):
        img, clin, lbl, _ = dataset[i]
        if (i % 72) in [0]: continue
        if (lbl != -1) or (include_unlabeled and lbl == -1):
            all_imgs.append(robust_flatten(img.numpy().squeeze(), target_dim))
            all_clins.append(clin.numpy())
            all_lbls.append(lbl)
            subj_map.append(i)
    return np.array(all_imgs), np.array(all_clins), np.array(all_lbls), np.array(subj_map)

def fast_resample_sharp(volume, msw_res=0.5, chh_res=0.9, target_shape=(64, 64, 72), sharpen=True):
    m_res, c_res = float(msw_res), float(chh_res)
    t_shape = target_shape if target_shape else (64, 64, 72)
    
    scaling_factor = m_res / c_res  
    in_center = np.array(volume.shape) / 2.0
    out_center = np.array(t_shape) / 2.0
    offset = in_center - (scaling_factor * out_center)
    
    resampled = affine_transform(
        volume,
        matrix=scaling_factor * np.eye(3),
        offset=offset,
        output_shape=t_shape,
        order=3,
        mode='nearest' 
    )
    
    if sharpen:
        blurred = gaussian_filter(resampled, sigma=0.8)
        resampled = resampled + 0.5 * (resampled - blurred)

    mask = np.ones(t_shape)
    mask = gaussian_filter(mask, sigma=1.5) 
    mask = mask / mask.max()

    return (resampled * mask).astype(np.float32)

class QSM_Dataset(Dataset):
    def __init__(self, nii_dir, seg_dir, mask_crop_fn, clinical_dict, label_map, 
                 limit=None, cache_path=None, load_cache=False, return_index=False,
                 slices_per_subject=None, random_seed=0, ext_mask_dir=None,
                 resample_fn=None):
        
        self.samples = []
        self.volumes = {} 
        self.seg_masks = {}
        self.clinical_dict = clinical_dict
        self.label_map = label_map
        self.train_mode = False 
        self.return_index = return_index
        self.slices_per_subject = slices_per_subject
        self.random_seed = random_seed
        self.ext_mask_dir = ext_mask_dir
        self.resample_fn = resample_fn

        if load_cache and cache_path and os.path.exists(cache_path):
            cached_data = torch.load(cache_path, weights_only=False)
            samples = cached_data['samples']
            mismatches = [s for s in samples if s['label'] != self.label_map.get(str(s['sub_id']), -1)]
            
            if len(mismatches) > 0:
                print(f"Updating {len(mismatches)} labels in cache...")
                for s in samples:
                    s['label'] = self.label_map.get(str(s['sub_id']), -1)
                cached_data['samples'] = samples
                torch.save(cached_data, cache_path)

            self.volumes = cached_data['volumes']
            self.samples = samples
            self.seg_masks = cached_data.get('seg_masks', {})
            print(f"Loaded cache with {len(self.samples)} slices.")
            return

        all_potential = [f for f in os.listdir(nii_dir) if f.startswith('qsm_') and f.endswith('.nii.gz')]
        loaded_count = 0
        
        for f in tqdm(all_potential, desc="Caching Volumes"):
            if limit and loaded_count >= limit: break
            try:
                raw_id = f.split('_')[1].split('.')[0]
                sub_id_str = str(int(raw_id))
                sub_id_int = int(sub_id_str)
                case_id = f"{sub_id_int:02d}"
                
                mask_path = os.path.join(seg_dir, f'seg_{case_id}.nii.gz')
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(seg_dir, f'seg_{case_id}.nii')
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(seg_dir, f'{sub_id_int}_roi_combined.nii')
                
                if not os.path.exists(mask_path): continue
                
                actual_label = self.label_map.get(sub_id_str, -1)
                if actual_label == -1 and sub_id_str not in self.clinical_dict: continue 

                img_nib = nib.load(os.path.join(nii_dir, f))
                mask_nib = nib.load(mask_path)

                if img_nib.shape == mask_nib.shape:
                    mask_nib = nib.Nifti1Image(mask_nib.get_fdata(), img_nib.affine, img_nib.header)

                img_obj = nib.as_closest_canonical(img_nib)
                mask_obj = nib.as_closest_canonical(mask_nib)
                
                raw_data = img_obj.get_fdata()
                mask_data = mask_obj.get_fdata()

                unique_labels = set(np.unique(mask_data))

                if 3 in unique_labels or 4 in unique_labels:
                    anchor_labels = [1, 2, 3, 4]
                elif 7 in unique_labels or 8 in unique_labels:
                    anchor_labels = [5, 6, 7, 8]
                else:
                    anchor_labels = [l for l in unique_labels if l > 0]

                mask_data = np.where(np.isin(mask_data, anchor_labels), mask_data, 0)
                anchor_mask = (mask_data > 0).astype(np.float32)

                if np.sum(anchor_mask) == 0:
                    print(f"Skipping {sub_id_int}: Target ROI not found after filtering.")
                    continue

                target_shape = (64, 64, 72)

                if self.resample_fn is not None:
                    m_res, c_res = 0.5, 0.9 
                    
                    native_crop_shape = tuple(int(round((ts * m_res) / c_res)) for ts in target_shape)
                    
                    raw_patch = mask_crop_fn(raw_data, anchor_mask, native_crop_shape)
                    mask_patch = mask_crop_fn(mask_data, anchor_mask, native_crop_shape)
                    
                    standardized_data = self.resample_fn(raw_patch, msw_res=m_res, chh_res=c_res, 
                                                        target_shape=target_shape, sharpen=True)
                    m_patch = self.resample_fn(mask_patch, msw_res=m_res, chh_res=c_res, 
                                            target_shape=target_shape, sharpen=False)
                    m_patch = (m_patch > 0.5).astype(np.uint8)
                else:
                    standardized_data = mask_crop_fn(raw_data, anchor_mask, target_shape)
                    m_patch = (mask_crop_fn(mask_data, anchor_mask, target_shape) > 0).astype(np.uint8)

                if loaded_count == 0:
                    print(f"DEBUG: Showing orientation for Subject {sub_id_int}...")
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    mid_z = standardized_data.shape[2] // 2
                    ax[0].imshow(np.rot90(standardized_data[:, :, mid_z]), cmap='gray')
                    ax[0].set_title(f"QSM Crop (Sub {sub_id_int})")
                    ax[1].imshow(np.rot90(m_patch[:, :, mid_z]), cmap='jet')
                    ax[1].set_title("Target ROI (Filtered)")
                    plt.show()

                standardized_data[standardized_data <= -2000] = 0.0
                if self.ext_mask_dir: 
                    skull_mask_path = os.path.join(self.ext_mask_dir, f"mag_{case_id}BrainExtractionMask.nii.gz")
                    if os.path.exists(skull_mask_path):
                        skull_mask = mask_crop_fn(nib.load(skull_mask_path).get_fdata(), anchor_mask, target_shape)
                        standardized_data = standardized_data * (skull_mask > 0)

                standardized_data = np.clip(standardized_data, -250.0, 250.0) / 250.0

                processed_vol = standardized_data.astype(np.float32) 
                
                self.volumes[sub_id_int] = processed_vol
                self.seg_masks[sub_id_int] = m_patch

                total_slices = processed_vol.shape[2] 
                rng = np.random.RandomState(self.random_seed + sub_id_int)
                rng = np.random.RandomState(self.random_seed + sub_id_int)
                
                if self.slices_per_subject is not None:
                    n_slices = self.slices_per_subject.get(actual_label, total_slices) if isinstance(self.slices_per_subject, dict) else self.slices_per_subject
                    slice_indices = rng.choice(total_slices, min(n_slices, total_slices), replace=False)
                else:
                    slice_indices = range(total_slices)
                
                for idx in slice_indices:
                    self.samples.append({'sub_id': sub_id_int, 'slice_idx': idx, 'label': actual_label})
                
                loaded_count += 1
            except Exception as e:
                print(f"Skipping {f}: {e}")

        if cache_path:
            torch.save({'volumes': self.volumes, 'samples': self.samples, 'seg_masks': self.seg_masks}, cache_path)
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, index):
            meta = self.samples[index]
            vol = self.volumes[meta['sub_id']]
            img_slice = vol[:, :, meta['slice_idx']]
            img_tensor = torch.from_numpy(img_slice).unsqueeze(0) 
            
            _sid = str(meta['sub_id'])
            if _sid not in self.clinical_dict:
                raise KeyError(
                    f"subject {_sid} has no clinical covariates; "
                    f"dataset and clinical CSV disagree ({len(self.clinical_dict)} subjects in dict)")
            clin_data = self.clinical_dict[_sid]
            return img_tensor, torch.tensor(clin_data, dtype=torch.float32), int(meta['label']), index
    

class QSM_c1_Dataset(Dataset):
    def __init__(self, nii_dir, seg_dir, mask_crop_fn, clinical_dict, label_map, 
                 limit=None, cache_path=None, load_cache=False, return_index=False,
                 slices_per_subject=None, random_seed=0):
        self.samples = []
        self.volumes = {} 
        self.seg_masks = {}
        self.clinical_dict = clinical_dict
        self.label_map = label_map
        self.train_mode = False 
        self.return_index = return_index
        self.slices_per_subject = slices_per_subject
        self.random_seed = random_seed

        if load_cache and cache_path and os.path.exists(cache_path):
            cached_data = torch.load(cache_path, weights_only=False)
            samples = cached_data['samples']
            
            bad_samples = [s for s in samples if s['label'] == -1 and str(s['sub_id']) in self.label_map]
            
            if len(bad_samples) > 0:
                print(f"Found {len(bad_samples)} mismatched labels. Forcing a RE-CACHE...")
            else:
                self.volumes = cached_data['volumes']
                self.samples = samples
                self.seg_masks = cached_data.get('seg_masks', {})
                print(f"Loaded cache with {len(self.samples)} slices from {len(self.volumes)} subjects")
                return

        all_potential = [f for f in os.listdir(nii_dir) if f.startswith('qsm_') and f.endswith('.nii.gz')]
        loaded_count = 0
        
        for f in tqdm(all_potential, desc="Caching Volumes"):
            if limit and loaded_count >= limit: 
                break
            try:
                raw_id = f.split('_')[1].split('.')[0]
                sub_id_str = str(int(raw_id))
                sub_id_int = int(sub_id_str)
                
                case_id = f"{sub_id_int:02d}"
                mask_path = os.path.join(seg_dir, f'seg_{case_id}.nii.gz')
                if not os.path.exists(mask_path): 
                    continue
                
                actual_label = self.label_map.get(sub_id_str, -1)
                
                if actual_label == -1 and sub_id_str not in self.clinical_dict:
                    continue 

                raw_data = nib.load(os.path.join(nii_dir, f)).get_fdata()
                mask_data = nib.load(mask_path).get_fdata()
                mask_data[mask_data <= 2] = 0
                
                img = mask_crop_fn(raw_data, mask_data, (72, 64, 64)) / 1000.0
                binary_mask = (mask_data > 0).astype(np.uint8)
                m_patch = mask_crop_fn(binary_mask, mask_data, (72, 64, 64))
                
                brain_indices = m_patch > 0
                if np.any(brain_indices):
                    img = (img - np.mean(img[brain_indices])) / (np.std(img[brain_indices]) + 1e-8)
                
                processed_vol = np.transpose(np.clip(img, -5.0, 5.0), (1, 2, 0)).astype(np.float32)
                
                self.volumes[sub_id_int] = processed_vol
                self.seg_masks[sub_id_int] = m_patch
                
                total_slices = processed_vol.shape[2]
                
                if self.slices_per_subject is not None:
                    if isinstance(self.slices_per_subject, dict):
                        n_slices = self.slices_per_subject.get(actual_label, total_slices)
                    else:
                        n_slices = self.slices_per_subject
                    
                    n_slices = min(n_slices, total_slices)
                    
                    rng = np.random.RandomState(self.random_seed + sub_id_int)
                    slice_indices = rng.choice(total_slices, n_slices, replace=False)
                else:
                    slice_indices = range(total_slices)
                
                for slice_idx in slice_indices:
                    self.samples.append({
                        'sub_id': sub_id_int, 
                        'slice_idx': slice_idx, 
                        'label': actual_label
                    })
                
                loaded_count += 1
            except Exception as e:
                print(f"Skipping {f}: {e}")
                continue
        
        final_labels = np.unique([s['label'] for s in self.samples])
        label_counts = {lbl: sum(1 for s in self.samples if s['label'] == lbl) for lbl in final_labels}
        print(f"\n>>> Cache built with {len(self.volumes)} subjects, {len(self.samples)} total slices")
        print(f">>> Label distribution:")
        for lbl in sorted(label_counts.keys()):
            name = "Responder" if lbl == 1 else "Non-Responder" if lbl == 0 else "Unlabeled"
            print(f"    Label {lbl} ({name}): {label_counts[lbl]} slices")
        
        if cache_path:
            torch.save({
                'volumes': self.volumes, 
                'samples': self.samples,
                'seg_masks': self.seg_masks
            }, cache_path)
            print(f">>> Cache saved to {cache_path}")
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, index):
        meta = self.samples[index]
        vol = self.volumes[meta['sub_id']]
        img = vol[:, :, meta['slice_idx']]
        img_tensor = torch.from_numpy(img).unsqueeze(0) / 5.0
        
        clin_data = self.clinical_dict.get(str(meta['sub_id']))
        if clin_data is not None:
            clin_vec = torch.tensor(clin_data, dtype=torch.float32)
        else:
            dim = getattr(self, 'clin_dim', 9) 
            clin_vec = torch.zeros(dim, dtype=torch.float32)
            
        label = int(meta['label'])
        
        return img_tensor, clin_vec, label, index

def load_ixi_data(mni_dir, csv_path, vol_size=96):
    df = pd.read_csv(csv_path)
    id_col = [c for c in df.columns if 'ID' in c.upper()][0]
    sex_col = [c for c in df.columns if 'SEX' in c.upper()][0]
    sex_lookup = dict(zip(df[id_col].astype(int), df[sex_col].map({1: 0, 2: 1})))
    
    files = sorted([f for f in os.listdir(mni_dir) if f.endswith('.nii.gz')])
    vols, labels = [], []
    
    for f in tqdm(files, desc="Loading Data"):
        match = re.search(r'(\d+)', f)
        if match and int(match.group(1)) in sex_lookup:
            img = nib.load(os.path.join(mni_dir, f)).get_fdata()
            c = np.array(img.shape) // 2
            r = vol_size // 2
            crop = img[c[0]-r:c[0]+r, c[1]-r:c[1]+r, c[2]-r:c[2]+r]
            if crop.shape == (vol_size, vol_size, vol_size):
                crop = (crop - np.mean(crop)) / (np.std(crop) + 1e-8)
                vols.append(crop.astype(np.float32))
                labels.append(sex_lookup[int(match.group(1))])
    
    return np.array(vols), np.array(labels).astype(np.float32)


def _parse_age_value(val, age_in_months=False):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        v = float(val)
    else:
        s = str(val).strip()
        m = re.match(r'^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$', s)
        if m:
            v = (float(m.group(1)) + float(m.group(2))) / 2.0
        else:
            m = re.match(r'^(\d+(?:\.\d+)?)\s*\+$', s)
            if m:
                v = float(m.group(1)) + 2.5
            else:
                m = re.match(r'^(\d+(?:\.\d+)?)', s)
                v = float(m.group(1)) if m else np.nan
    if np.isnan(v):
        return v
    return v / 12.0 if age_in_months else v


def load_hcp_data(data_dir, csv_path, vol_size=96,
                   file_glob='*T1w*restore*brain*.nii.gz',
                   id_col=None, age_col=None, age_in_months=False,
                   verbose=True):
    df = pd.read_csv(csv_path)

    if id_col is None:
        cols_upper = {c.upper(): c for c in df.columns}
        if 'SRC_SUBJECT_ID' in cols_upper:
            id_col = cols_upper['SRC_SUBJECT_ID']
        else:
            id_candidates = [c for c in df.columns if 'SUBJ' in c.upper() or c.upper() == 'ID']
            if not id_candidates:
                id_candidates = [c for c in df.columns if 'ID' in c.upper()]
            if not id_candidates:
                raise ValueError("Could not auto-detect a subject-ID column; pass id_col explicitly.")
            id_col = id_candidates[0]

    if age_col is None:
        age_candidates = [c for c in df.columns if 'AGE' in c.upper()]
        if not age_candidates:
            raise ValueError("Could not auto-detect an age column; pass age_col explicitly.")
        yrs = [c for c in age_candidates if 'YR' in c.upper() or 'YEAR' in c.upper()]
        interview = [c for c in age_candidates if 'INTERVIEW' in c.upper()]
        if yrs:
            age_col = yrs[0]
        elif interview:
            age_col = interview[0]
            age_in_months = True
        else:
            age_col = age_candidates[0]

    if verbose:
        unit = " (months)" if age_in_months else ""
        print(f"Using ID column: '{id_col}', age column: '{age_col}'{unit}")

    age_lookup = {
        str(sid).strip(): _parse_age_value(age, age_in_months)
        for sid, age in zip(df[id_col], df[age_col])
    }
    age_lookup = {k: v for k, v in age_lookup.items() if not np.isnan(v)}
    id_lookup_upper = {k.upper(): k for k in age_lookup}

    files = sorted(glob.glob(os.path.join(data_dir, '**', file_glob), recursive=True))
    if verbose:
        print(f"Found {len(files)} candidate volumes under {data_dir}")

    vols, labels, subj_ids = [], [], []
    for fpath in tqdm(files, desc="Loading HCP Data"):
        rel = os.path.relpath(fpath, data_dir)
        tokens = re.split(r'[\\/_.\-]', rel)

        matched_id = None
        for tok in tokens:
            if tok.upper() in id_lookup_upper:
                matched_id = id_lookup_upper[tok.upper()]
                break
        if matched_id is None:
            digit_match = re.search(r'(\d{5,})', rel)
            if digit_match and digit_match.group(1) in age_lookup:
                matched_id = digit_match.group(1)
        if matched_id is None:
            continue

        try:
            img = nib.load(fpath).get_fdata()
        except Exception as e:
            if verbose:
                print(f"Skipping {fpath}: {e}")
            continue

        c = np.array(img.shape[:3]) // 2
        r = vol_size // 2
        crop = img[c[0]-r:c[0]+r, c[1]-r:c[1]+r, c[2]-r:c[2]+r]
        if crop.shape != (vol_size, vol_size, vol_size):
            continue

        crop = (crop - np.mean(crop)) / (np.std(crop) + 1e-8)
        vols.append(crop.astype(np.float32))
        labels.append(age_lookup[matched_id])
        subj_ids.append(matched_id)

    if verbose:
        print(f"Loaded {len(vols)} / {len(files)} volumes with matched ages")

    return np.array(vols), np.array(labels).astype(np.float32), subj_ids


def prepare_nda_csv(txt_path, out_csv_path=None, sep='\t'):
    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        raw_lines = f.readlines()
    raw_lines = [ln.rstrip('\n').rstrip('\r') for ln in raw_lines if ln.strip() != '']

    idx = 0
    if sep not in raw_lines[0] and ',' in raw_lines[0]:
        idx = 1

    header_fields = raw_lines[idx].split(sep)
    data_start = idx + 1

    if data_start < len(raw_lines):
        next_fields = raw_lines[data_start].split(sep)
        if len(next_fields) == len(header_fields):
            nonempty = [f for f in next_fields if f.strip()]
            avg_words = np.mean([len(f.split()) for f in nonempty]) if nonempty else 0
            if avg_words > 1.5:
                data_start += 1

    csv_text = sep.join(header_fields) + '\n' + '\n'.join(raw_lines[data_start:])
    df = pd.read_csv(io.StringIO(csv_text), sep=sep)

    if out_csv_path is None:
        out_csv_path = os.path.splitext(txt_path)[0] + '_clean.csv'
    df.to_csv(out_csv_path, index=False)
    print(f"Wrote cleaned CSV ({len(df)} rows, {len(df.columns)} cols) to {out_csv_path}")
    print(f"Columns: {list(df.columns)}")
    return out_csv_path


def generate_distance_data(n_samples, vol_size=32, swapped=False, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    X = torch.zeros(n_samples, 1, vol_size, vol_size, vol_size)
    Y = torch.zeros(n_samples, dtype=torch.long)
    
    for i in range(n_samples):
        label = np.random.randint(0, 2)
        
        if label == 0:
            distance = np.random.uniform(3, 8)
        else:
            distance = np.random.uniform(12, 18)
        
        center = np.random.uniform(vol_size * 0.3, vol_size * 0.7, 3)
        
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        point1 = center - direction * distance / 2
        point2 = center + direction * distance / 2
        
        if swapped:
            if point1[0] < vol_size / 2:
                point1[0] = vol_size - point1[0]
            if point2[0] < vol_size / 2:
                point2[0] = vol_size - point2[0]
        
        point1 = np.clip(point1, 0, vol_size - 1).astype(int)
        point2 = np.clip(point2, 0, vol_size - 1).astype(int)
        
        X[i, 0, point1[0], point1[1], point1[2]] = 1
        X[i, 0, point2[0], point2[1], point2[2]] = 1
        Y[i] = label
    
    return X, Y

class BalancedDataset(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
        self.pos_idx = np.where(Y == 1)[0]
        self.neg_idx = np.where(Y == 0)[0]
        self.count = max(len(self.pos_idx), len(self.neg_idx)) * 2

    def __len__(self): return self.count

    def __getitem__(self, idx):
        i = np.random.choice(self.pos_idx) if idx % 2 == 0 else np.random.choice(self.neg_idx)
        return self.X[i], self.Y[i]
