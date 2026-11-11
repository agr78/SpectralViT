"""Look for demographics in NIfTI headers, extensions and BIDS JSON sidecars.

usage: python nii_meta_probe.py <subject_dir_or_root>
"""
import os, sys, json, glob
import nibabel as nib

# Takes the subject directory as an argument rather than defaulting to one: the default
# used to name a real HCP-Aging subject, and this repository is public.
if len(sys.argv) < 2:
    sys.exit(__doc__.strip())
root = sys.argv[1]
print(f'probing: {root}\n')

nii = []
for dp, _, fs in os.walk(root):
    for f in fs:
        if f.endswith(('.nii', '.nii.gz')):
            nii.append(os.path.join(dp, f))
    if len(nii) > 12: break
print(f'{len(nii)} NIfTI files found (showing first few)\n')

for p in nii[:3]:
    print(f'--- {os.path.relpath(p, root)}')
    try:
        h = nib.load(p).header
        print(f'    descrip  : {bytes(h["descrip"]).decode("utf-8","replace").strip(chr(0))!r}')
        print(f'    aux_file : {bytes(h["aux_file"]).decode("utf-8","replace").strip(chr(0))!r}')
        img = nib.load(p)
        ext = getattr(img.header, 'extensions', [])
        print(f'    extensions: {len(ext)}')
        for e in list(ext)[:2]:
            s = str(e.get_content())[:200]
            print(f'      code={e.get_code()} content={s!r}')
    except Exception as e:
        print(f'    (header unreadable: {type(e).__name__}: {e})')
    # sidecar
    base = p[:-7] if p.endswith('.nii.gz') else p[:-4]
    for j in [base + '.json'] + glob.glob(os.path.join(os.path.dirname(p), '*.json')):
        if os.path.exists(j):
            try:
                d = json.load(open(j))
                keys = [k for k in d if any(t in k.lower() for t in
                        ('age','sex','gender','birth','patient','subject'))]
                print(f'    sidecar {os.path.basename(j)}: {len(d)} keys; demographic keys: {keys}')
                for k in keys[:6]: print(f'      {k} = {d[k]!r}')
            except Exception:
                print(f'    sidecar {os.path.basename(j)}: unreadable')
            break
    print()
