#!/usr/bin/env python3
"""Computations specific to this paper's experiments that the generic scripts/query.py does not do. Each call
prints one number, or a plain-text table with one row per provider.

RUNS_SPEC names the per-seed runs of the ranking experiment (data/results/analyst_heuristics.json::runs): a dict
from seed key to provider to that seed's result record. A FIELD is a '::' path inside a record.

    over-seeds STAT RUNS_SPEC --seed-prefix P --field FIELD [--field ...] [--divide-by FIELD]
               [--provider NAME | --over-providers {min,max}]
        for each provider, STAT (median, min, max, count) over the seeds (the keys of RUNS_SPEC starting with P)
        of FIELD, divided by --divide-by within each seed; a table, or one provider's value, or the min/max over
        providers
    best-median-hits RUNS_SPEC --seed-prefix P --field FIELD [--field ...] --top K [--hits-at-most N]
        for each provider, round(K x the largest seed median among the FIELDs): how many of the top K methods
        the best ranking gets right; a table, or with --hits-at-most the number of providers with at most N
    sdk-isolation-median JSONL --colouring C --field FIELD [--field ...] [--log2]
                         [--provider NAME | --over-providers {min,max} | --count-below X]
        per provider, the median over its applications of FIELD inside the record's own-SDK target
        (targets::SDK:<provider>::colourings::C); a FIELD ending in '::*' gives one column per key; --log2 takes
        log2 of each median; a table, one provider, the min/max over providers, or the number of providers whose
        median is below X

Output format (precision only), as in query.py: --digits N, --rounding, --percent, --permille, --in-millions,
--sep, --significant N, --sci N. Without these, whole numbers print without decimals and other numbers in
full.
"""
import argparse
import json
import math
import statistics
import sys
from decimal import ROUND_CEILING, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal


def field(record, path):
    """The value at a '::' path inside a record, or None if any key is missing."""
    node = record
    for key in path.split("::"):
        if isinstance(node, dict) and key in node:
            node = node[key]
        elif isinstance(node, list) and key.isdigit() and int(key) < len(node):
            node = node[int(key)]
        else:
            return None
    return node


def fmt(value, a):
    """A number at the requested precision, as query.py prints it; a missing value prints as '--'."""
    if value is None:
        return "--"
    if a.in_millions:
        value = value / 1e6
    if a.percent:
        value = value * 100
    if a.permille:
        value = value * 1000
    if a.sci is not None:
        return f"{value:.{a.sci}e}"
    if a.significant is not None:
        return f"{value:.{a.significant}g}"
    if a.digits is not None:
        mode = ROUND_HALF_UP
        if a.rounding == "half-even":
            mode = ROUND_HALF_EVEN
        elif a.rounding == "ceiling":
            mode = ROUND_CEILING
        d = Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-a.digits), rounding=mode)
        if a.digits < 0:
            d = d.quantize(Decimal(1))
        # no negative zero ("-0.00") for a value that rounds to zero
        if d == 0:
            d = d.copy_abs()
        if a.sep:
            return f"{d:,}"
        return str(d)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int) and a.sep:
        return f"{value:,}"
    return str(value)


def print_table(header, rows):
    """First column left-aligned, the others right-aligned, a rule under the header (as query.py)."""
    widths = []
    for i in range(len(header)):
        w = len(header[i])
        for row in rows:
            if len(row[i]) > w:
                w = len(row[i])
        widths.append(w)
    lines = [header]
    for row in rows:
        lines.append(row)
    for n in range(len(lines)):
        cells = []
        for i in range(len(lines[n])):
            if i == 0:
                cells.append(lines[n][i].ljust(widths[i]))
            else:
                cells.append(lines[n][i].rjust(widths[i]))
        print("  ".join(cells).rstrip())
        if n == 0:
            print("-" * (sum(widths) + 2 * (len(widths) - 1)))


# parse arguments: output format, then one subparser per computation
fmtp = argparse.ArgumentParser(add_help=False)
fmtp.add_argument("--digits", type=int, metavar="N")
fmtp.add_argument("--rounding", choices=["half-up", "half-even", "ceiling"], default="half-up")
fmtp.add_argument("--percent", action="store_true")
fmtp.add_argument("--permille", action="store_true")
fmtp.add_argument("--in-millions", action="store_true")
fmtp.add_argument("--sep", action="store_true")
fmtp.add_argument("--significant", type=int, metavar="N")
fmtp.add_argument("--sci", type=int, metavar="N")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="op", metavar="COMPUTATION")
sub.required = True
p = sub.add_parser("over-seeds", parents=[fmtp])
p.add_argument("stat", choices=["median", "min", "max", "count"])
p.add_argument("runs")
p.add_argument("--seed-prefix", required=True)
p.add_argument("--field", action="append", required=True, metavar="FIELD")
p.add_argument("--divide-by", metavar="FIELD")
p.add_argument("--provider")
p.add_argument("--over-providers", choices=["min", "max"])
p = sub.add_parser("best-median-hits", parents=[fmtp])
p.add_argument("runs")
p.add_argument("--seed-prefix", required=True)
p.add_argument("--field", action="append", required=True, metavar="FIELD")
p.add_argument("--top", type=int, required=True)
p.add_argument("--hits-at-most", type=int)
p = sub.add_parser("sdk-isolation-median", parents=[fmtp])
p.add_argument("jsonl")
p.add_argument("--colouring", required=True)
p.add_argument("--field", action="append", required=True, metavar="FIELD")
p.add_argument("--log2", action="store_true")
p.add_argument("--provider")
p.add_argument("--over-providers", choices=["min", "max"])
p.add_argument("--count-below", type=float)
a = ap.parse_args()

if a.op == "over-seeds" or a.op == "best-median-hits":
    which = "median"
    if a.op == "over-seeds":
        which = a.stat
    # the dict of runs at RUNS_SPEC
    parts = a.runs.split("::")
    with open(parts[0]) as f:
        runs = json.load(f)
    for key in parts[1:]:
        if key not in runs:
            sys.exit(f"derive.py: key {key!r} not found in {a.runs}")
        runs = runs[key]
    seeds = []
    for k in runs:
        if k.startswith(a.seed_prefix):
            seeds.append(k)
    if not seeds:
        sys.exit(f"derive.py: no seed key starts with {a.seed_prefix!r}")
    providers = list(runs[seeds[0]].keys())
    stat_of = {}
    for p in providers:
        stat_of[p] = []
        for path in a.field:
            vals = []
            for s in seeds:
                v = field(runs[s][p], path)
                if v is None:
                    sys.exit(f"derive.py: {path} not found for {s} {p}")
                if a.op == "over-seeds" and a.divide_by:
                    v = v / field(runs[s][p], a.divide_by)
                vals.append(v)
            if which == "median":
                stat_of[p].append(statistics.median(vals))
            elif which == "count":
                stat_of[p].append(len(vals))
            elif which == "min":
                stat_of[p].append(min(vals))
            else:
                stat_of[p].append(max(vals))
    if a.op == "best-median-hits":
        hits = {}
        for p in providers:
            hits[p] = int(round(max(stat_of[p]) * a.top))
        if a.hits_at_most is not None:
            n = 0
            for p in providers:
                if hits[p] <= a.hits_at_most:
                    n += 1
            print(fmt(n, a))
        else:
            rows = []
            for p in providers:
                rows.append([p, fmt(hits[p], a)])
            print_table(["provider", "hits in top " + str(a.top)], rows)
    elif a.provider:
        print(fmt(stat_of[a.provider][0], a))
    elif a.over_providers:
        vals = []
        for p in providers:
            vals.append(stat_of[p][0])
        if a.over_providers == "min":
            print(fmt(min(vals), a))
        else:
            print(fmt(max(vals), a))
    else:
        header = ["provider"]
        for path in a.field:
            if a.divide_by:
                header.append(which + " " + path + " / " + a.divide_by)
            else:
                header.append(which + " " + path)
        rows = []
        for p in providers:
            row = [p]
            for v in stat_of[p]:
                row.append(fmt(v, a))
            rows.append(row)
        print_table(header, rows)

else:
    # sdk-isolation-median: each application's own-SDK target, grouped by provider
    per_provider = {}
    with open(a.jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            target = record["targets"].get("SDK:" + record["family"])
            if not target:
                continue
            if record["family"] not in per_provider:
                per_provider[record["family"]] = []
            per_provider[record["family"]].append(target["colourings"][a.colouring])
    # expand '*' in a field into one column per key
    columns = []
    for path in a.field:
        if not path.endswith("::*"):
            columns.append(path)
            continue
        first = per_provider[list(per_provider)[0]][0]
        for k in field(first, path[:-3]):
            columns.append(path[:-3] + "::" + str(k))
    medians = {}
    for p in per_provider:
        medians[p] = []
        for path in columns:
            vals = []
            for x in per_provider[p]:
                vals.append(field(x, path))
            m = statistics.median(vals)
            if a.log2:
                m = math.log2(m)
            medians[p].append(m)
    if a.provider:
        print(fmt(medians[a.provider][0], a))
    elif a.over_providers or a.count_below is not None:
        vals = []
        for p in medians:
            vals.append(medians[p][0])
        if a.count_below is not None:
            n = 0
            for v in vals:
                if v < a.count_below:
                    n += 1
            print(n)
        elif a.over_providers == "min":
            print(fmt(min(vals), a))
        else:
            print(fmt(max(vals), a))
    else:
        header = ["provider"]
        for path in columns:
            if a.log2:
                header.append("log2 " + path)
            else:
                header.append(path)
        rows = []
        for p in medians:
            row = [p]
            for v in medians[p]:
                row.append(fmt(v, a))
            rows.append(row)
        print_table(header, rows)
