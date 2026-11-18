# ML-Guided Reverse Engineering of Mobile Residential Proxy APKs

Artifact for the paper: the result file behind every measured number, a sample of call graphs that runs the
attribution end to end, and the experiment scripts.

The corpus manifest and WL vectors come from the dataset paper's artifact
(https://anonymous.4open.science/r/static_artifacts_eurosp-C707/). Place it next to this directory;
`verify.sh` finds the sibling that holds its manifest, or set `STATIC_ARTIFACT=/path/to/it`.

## Run

    ./install.sh                 # checks for Python 3.9+, installs dependencies into ./.venv
    ./verify.sh                  # recomputes the paper's numbers, then runs the attribution on the sample
    ./verify.sh --numbers-only   # numbers only

## Output

Pairs of a command (`$ ...`) and its result, in the paper's order, also written to `verify_output.txt`.
Nothing stores the paper's values: compare each result with the paper.

- `scripts/query.py` (shared by all three artifacts) reads values: a SPEC is a file path followed by keys,
  joined by `::` (`*` expands to every key). `.csv` and `.jsonl` files are lists of rows.
- `scripts/derive.py` holds the computations specific to this paper (aggregation over sampling seeds,
  best-ranked hits, SDK isolation medians).
- `scripts/run_attribution.py` re-extracts each sample's vector from its call graph, checks it against the
  released vector, and ranks the application's methods.

`--help` on each program documents every option.

## Layout

    data/results/   result files
    data/samples/   two call graphs per provider (zstd GML), listed in samples.json
    scripts/        query.py, derive.py, run_attribution.py and the modules it imports; experiments/ holds
                    the experiment scripts

## Not included

APKs (retrieve them by SHA-256 from AndroZoo, or from VirusTotal for the few not in AndroZoo), numbers that
come from manual reverse engineering of the applications, cited figures, and experiment settings, which are
set in `scripts/experiments/`.
