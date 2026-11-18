"""Where the scripts find data and write results.

SPECTRALVIT_DATA  datasets and caches (default: data/ in this repository)
SPECTRALVIT_RAW   original cohort downloads in BIDS layout (default: data/raw)
Results are written to results/ in this repository.
"""
import os, sys

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.expanduser(os.environ.get('SPECTRALVIT_DATA', os.path.join(REPO, 'data')))
RAW = os.path.expanduser(os.environ.get('SPECTRALVIT_RAW', os.path.join(DATA, 'raw')))
RESULTS = os.path.join(REPO, 'results')
PYTHON = sys.executable
