# ML-Guided Reverse Engineering of Mobile Residential Proxy APKs

Artifact for the paper of the same name.

## Reproducing the paper's numbers

The corpus manifest, the WL vectors and the classification baselines are released once, with the dataset
paper's artifact (https://github.com/sarsaeroth/static_artifacts_eurosp), and are not duplicated here. Clone it next to this
repository; `verify.sh` finds it as `../static_artifacts_eurosp` (or `../static`), or point `STATIC_ARTIFACT` at it:

    export STATIC_ARTIFACT=/path/to/static_artifacts_eurosp   # default: the sibling clone
    ./install.sh                             # checks for Python 3.9+, installs the dependencies into ./.venv
    ./verify.sh                              # numbers, then the attribution on the shipped sample
    ./verify.sh --numbers-only

`verify.sh` is a flat list of commands. It prints each command line, prefixed with `$`, and then its output:
one number, or a plain-text table. Nothing else is printed: no labels, no expected values, no pass or fail. To
check the paper, read each command to see what it computes, and compare its output with the number the paper
prints. The same output is written to `verify_output.txt`. Every command runs even if an earlier one fails;
`verify.sh` exits non-zero if any command failed (a missing file or key), never because of a value. With the
full run, `scripts/run_attribution.py` then runs the attribution end to end on the sample.

Two programs compute everything; the subcommand names the operation, and `--help` (on the program and on each
subcommand) documents every option:

- `scripts/query.py` is the generic tool. It is the same file, byte for byte, in all three artifacts of this
  repository. Values: `value SPEC`, `stat {min,max,median,mean,sum,count,max-abs} SPEC...`, `diff`/`ratio A B`,
  `count-equal A B` (positions at which two lists of values agree), `log {2,10,e} SPEC`, `one-over-count SPEC`
  (chance level for as many classes as SPEC names values). Records: `count RECORDS` with filters `--where`,
  `--where-in COL=V1,V2`, `--above FIELD=X`, `--nonempty FIELD`; `group-stat STAT RECORDS --by FIELD --field FIELD`
  (the statistic within each group, as a table, or reduced with `--over-groups STAT` or `--count-groups-at-max`).
  Tables: `table SPEC --field FIELD...` (one row per record, one column per field), `grid-table` and
  `value-at-grid-point` (lists indexed by a grid such as the deletion fractions).
- `scripts/derive.py` holds the computations specific to this paper's experiments: `over-seeds STAT` (per
  provider, a statistic over the seeds of the ranking runs), `best-median-hits` (per provider, how many of the
  top K methods the best ranking gets right) and `sdk-isolation-median` (per provider, the median isolated share
  of each application's own SDK tokens).

A SPEC names values inside a result file: the file path, then the keys (or list indices) to follow, joined by
`::`. A key `*` expands to every key of a dict (or every item of a list), so one SPEC can name many values:
`data/results/mincut_v3.json::summary::*::analysed` is the `analysed` count of every provider. `.csv` and
`.jsonl` files are the list of their rows. `--skip-key KEY` leaves out the entries named KEY when `*` expands a
dict (the multiple-kernel tables leave out `beta`, the view weights stored beside the providers). A FIELD is such
a path inside one record, and a table's rows are labeled by the keys the `*` matched. Paths starting with
the dataset artifact (`STATIC_ARTIFACT`, or the sibling clone) read its files.

Precision is set only by format flags: `--digits N` rounds half up to N decimals, `--percent` multiplies by 100
first (the `%` sign is not printed), `--sep` adds thousands separators, and `--in-millions` divides by
one million. Without `--digits`, numbers print in full. The paper omits the leading zero of values below one in
its tables (`.5`); the output prints it (`0.5`).

## What is shipped, and what that supports

    data/results/                    the result file behind every number verify.sh prints not already released with the dataset artifact
    data/samples/                    two call graphs per provider (zstd GML), listed in samples.json
    scripts/                         query.py (generic, shared) and derive.py (this paper's computations), used by
                                     verify.sh; run_attribution.py and the modules
                                     it imports (vectorizer, graph reader, attribution code)
    scripts/experiments/             the experiment scripts that produced the results (flattened; logic as run)

From the dataset artifact (`STATIC_ARTIFACT`): `data/manifests/manifest_v3.csv` (every build: SHA-256, provider, label
source, status, DEX and lineage group), `data/vectors/` (identifier WL vectors, CFG and FCG, at the attributed
model's dimension, every classified application) and `data/results/proxy_v3*` (classification, renaming and
open-world baselines).

What that supports: `run_attribution.py` trains the attributed model (identifier WL over FCGs, linear) from the
dataset artifact's vectors, re-extracts each sample application's vector with provenance from its call graph,
prints its cosine to the released vector (identical vectors give a cosine of one), and ranks the
application's methods by the exact split of each bucket's SHAP value over the methods that produced it, the
split every attribution experiment in the paper uses. Extraction, attribution and ranking can therefore be
checked directly. The corpus-wide figures come from the full corpus and are reported from the released result
files; the sample demonstrates the method rather than reproducing those numbers sample for sample. The
experiment scripts expect the full research tree and are shipped to document the protocol.

Result files:

| file | paper |
|---|---|
| `dex_reuse_v3.json` | DEX reuse per provider (corpus table) |
| `label_validity_dex.json`, `label_validity_neardup.json` | label-unsupported builds and classification without them |
| `analyst_heuristics.json` | method rankings against analyst heuristics, 5 seeds (attribution table) |
| `mkldroid.json` | multiple-kernel localization baseline |
| `attribution_robustness_v3.json` | background-allocation rules and out-of-fold attribution |
| `deletion_v3.json` | deletion of attributed and of random methods |
| `confound_v3.json` | masking of advertising code and of SDK code |
| `masked_attribution_v3.json` | attribution with advertising code masked; rank of the first SDK method |
| `route_compare_v3.json` | explanation route against marker route to the SDK's network core |
| `highdim_reach_v3.json` | reach to the network core from the rankings at the smallest and largest hash dimension |
| `explanation_diagnostic_v3.json` | per-provider predictors of where the explanation succeeds |
| `bootstrap_chain_v3.json` | explanation, first SDK method, derived marker, network core, end to end |
| `mincut_v3.json` | host-to-SDK minimum cut, SDK topology, reflection and native loading |
| `manifest_v3.json` | manifest-declared SDK components |
| `search_space_v3.json` | ranked methods, entry points and slices |
| `exp_tier1_identifiability_truth_v2.json(l)` | token isolation against the e^{-M/d} model (summary, and per application) |
| `highdim_localization.json` | localization against hash dimension |
| `sweep_v3.json` | dimension and holdout sensitivity |

Needs Python 3.9+, numpy, scikit-learn and networkx; `.zst` graphs are read through the standard library on Python 3.14+ and through the `zstandard` package before that (`./install.sh` installs what is needed).

## What verify.sh does not cover

The reverse-engineering section's counts (populations per provider, endpoints, ports, consent behavior, packing)
come from byte scans and manual analysis of the applications and are not recomputable without them; the
literature figures in the introduction and background are cited, not measured. Corpus-wide totals that need the
full graph store (DEX-file occurrences, raw and library-filtered method-node counts) have no command. Comments
in `verify.sh` (not printed) name the paper section each group of commands belongs to.

## Settings (not printed)

`verify.sh` prints results recorded in the shipped files and values computed from them. Protocol settings that
no result file records are not printed; they are set in the experiment scripts:

| setting | where it is set |
|---|---|
| seeds of the ranking runs | `SEEDS` in `scripts/experiments/exp_analyst_heuristics.py` |
| applications sampled per provider | `PER_FAM` in `exp_deletion_v3.py`, `exp_mkldroid.py`, `exp_mincut_v3.py`, `exp_confound_v3.py`, `highdim_localization.py` |
| sampling seed | `SEED` in the same scripts |
| multiple-kernel views, WL iterations and optimization rounds | `VIEWS`, `ITERS`, `ROUNDS` in `exp_mkldroid.py` |
| masking repeats and hash dimension | `REPEATS`, `DIM` in `exp_confound_v3.py` |
| call-graph search limit of the route comparison | `MAX_VISIT` in `exp_route_compare_v3.py` |
| cosine threshold of the lineage grouping | `NEARDUP_COS` in `exp_harness.py` |

## Corpus availability

We do not redistribute the APKs. Most came from AndroZoo, whose user agreement does not permit redistribution;
the dataset artifact's `data/manifests/manifest_v3.csv` identifies every sample by SHA-256 so the same binaries can be
retrieved with an AndroZoo API key, and its `source` column names where each build can be retrieved. Some
samples are currently retrievable only via VirusTotal.

The full graph store and the vectors at larger hash dimensions are not included; the scripts regenerate them
from the APKs.
