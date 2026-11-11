"""Execute a notebook in process and store each cell's stdout as its output.

Written because the notebook has no other generator: cells share one namespace and the
later statistics depend on the order of the model list, so they must run in sequence.

usage: run_notebook.py <notebook.ipynb>
"""
import sys, os, json, io, contextlib, traceback, time

# Absolute, because the notebook's first cell does os.chdir(REPO): a relative path
# here writes the executed copy into whatever directory that leaves us in, which
# silently strands the outputs away from the notebook they belong to.
nbf = os.path.abspath(sys.argv[1])
nb = json.load(open(nbf))
ns, n = {}, 0
for c in nb['cells']:
    if c['cell_type'] != 'code':
        continue
    n += 1
    src = ''.join(c['source'])
    buf = io.StringIO(); t = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, f'cell{n}', 'exec'), ns)
        err = None
    except Exception:
        err = traceback.format_exc()
    txt = buf.getvalue()
    outs = []
    if txt:
        outs.append({'output_type': 'stream', 'name': 'stdout',
                     'text': [l + '\n' for l in txt.split('\n')[:-1]]})
    if err:
        outs.append({'output_type': 'stream', 'name': 'stderr', 'text': [err]})
    c['outputs'] = outs
    c['execution_count'] = n
    print(f'--- cell {n} ({time.time()-t:.0f}s) ---', flush=True)
    print(txt.rstrip(), flush=True)
    if err:
        print('!! ' + err, flush=True)
    json.dump(nb, open(nbf, 'w'), indent=1)
json.dump(nb, open(nbf, 'w'), indent=1)
print(f'\nwrote {nbf} ({n} code cells)')
