# Evaluation results log

A record of assembly evaluation runs: what was run, how to reproduce it, and
what came out. Newest run first.

Results are only comparable within a single run block — they depend on the
config, the dataset and the commit, all of which are recorded per block.

---

## 2026-09-13 — ablation pilot on 10-piece synthetic puzzles

**Purpose.** First use of the evaluation harness on real ground-truth data: check
that the metrics, the repeat/seed machinery and the ablation switches all work
end to end, and get a first quantitative reading on what each optional part of
the algorithm contributes.

This is a **pilot, not a result**. Six puzzles is far too small a sample to
support any of the comparisons below, and the caveats section explains why
several of the numbers cannot be taken at face value.

### Setup

| | |
|---|---|
| Commit | `4d99771` |
| Python | 3.11.16 |
| Machine | Apple M4 Pro, 12 cores |
| Dataset | `synth_artworks/test`, the six alphabetically-first 10-piece puzzles |
| Runs | 6 puzzles × 3 seeds (0, 1, 2) = 18 runs per configuration |
| Configurations | 5 (baseline + 4 ablations) = 90 runs total |
| Wall-clock | roughly 25 minutes for the whole sweep |

The dataset comes from the CNN successor project (`cnn-piece-assemble`,
generated from National Gallery of Art open-access images). It is kept outside
this repository — it is large and third-party — at
`~/piece-assemble-eval-data/`. Only 10-piece puzzles were used; larger ones
take too long for a pilot.

Config: `sample_config.yaml` values throughout, except `scale: 1`,
`n_iters: 15`, `n_processes: 4`. Each ablation changes exactly one setting:

| Label | Change from baseline |
|---|---|
| `baseline` | — |
| `no_icp` | `clustering.use_icp: false` |
| `no_trusted_clusters` | `clustering.use_trusted_clusters: false` |
| `no_match_reuse` | `clustering.use_match_reuse: false` |
| `shape_only` | `descriptor.color_dist_w: 0` |

### Results

Mean ± standard deviation over all 18 runs of each configuration.
`largest_cluster_ratio` is the fraction of pieces in the largest correctly
assembled cluster (1.0 = puzzle fully solved); `num_of_clusters` is how many
separate clusters the 10 pieces ended up in (1 = fully solved, 10 = nothing
assembled); errors are over correctly matched neighbour pairs.

| Configuration | largest_cluster_ratio | num_of_clusters | translation_error | angle_error | runtime (s) |
|---|---|---|---|---|---|
| `baseline` | 0.254 ± 0.128 | 7.33 ± 2.03 | 1.341 ± 0.777 | 0.00050 ± 0.0004 | 16.5 ± 3.0 |
| `no_icp` | 0.252 ± 0.219 | 7.83 ± 2.26 | 3.404 ± 3.552 | 0.00370 ± 0.0038 | 7.5 ± 0.8 |
| `no_trusted_clusters` | 0.254 ± 0.220 | 7.67 ± 2.70 | 0.620 ± 0.648 | 0.00040 ± 0.0004 | 17.5 ± 5.6 |
| `no_match_reuse` | 0.204 ± 0.084 | 8.50 ± 1.30 | 1.344 ± 1.206 | 0.00040 ± 0.0006 | 14.3 ± 1.4 |
| `shape_only` | 0.270 ± 0.210 | 7.33 ± 2.50 | 1.039 ± 0.802 | 0.00050 ± 0.0005 | 15.1 ± 2.1 |

Paired per-run differences against the baseline (same puzzle, same seed), with
the ratio of the mean difference to its standard error as a rough indication of
whether the difference is worth taking seriously at all — anything below about 2
is indistinguishable from noise at this sample size:

| Configuration | Δ largest_cluster_ratio | \|mean\|/se | Δ translation_error | \|mean\|/se |
|---|---|---|---|---|
| `no_icp` | −0.002 | 0.07 | +2.063 | 2.45 |
| `no_trusted_clusters` | −0.000 | 0.00 | −0.721 | 2.83 |
| `no_match_reuse` | −0.050 | 2.70 | +0.002 | 0.01 |
| `shape_only` | +0.017 | 0.77 | −0.302 | 1.37 |

### What this suggests (tentatively)

- **ICP buys precision, not coverage.** Turning it off leaves the amount
  assembled essentially unchanged (Δ ratio −0.002) but makes translation error
  2.5× worse and angle error ~7× worse, while halving runtime. So its value is
  in how accurately pieces are placed, not in how many get placed — which is
  consistent with what ICP is there to do, but worth stating that way
  rather than implying it improves assembly success.
- **Match reuse is the only switch that visibly affects coverage**, and only
  slightly (ratio 0.204 vs 0.254).
- **Colour did not help on this data.** `shape_only` scored marginally *better*
  on both coverage and error, which runs against the expectation that the colour
  descriptor helps. Before reading anything into it, note the difference is well
  inside the noise and the descriptor weights were never tuned for this dataset.
  It does mean the contribution of colour needs an experiment designed to
  isolate it — same image, colour term on versus off — rather than an assumption.
- **Turning trusted clusters off lowered registration error.** Counterintuitive
  and unexplained; most likely an artefact of which pairs end up counted as
  correctly matched rather than a real improvement. Worth understanding before
  it is ever reported.

### Caveats — read before quoting any of the above

1. **Six puzzles.** Nothing here is established. The paired column is a sanity
   indicator, not a significance test, and no correction for multiple
   comparisons has been applied.
2. **These runs are not reproducible as configured.** They used
   `n_processes: 4` without a fixed `PYTHONHASHSEED`, so re-running the sweep
   will give somewhat different numbers (see the reproducibility section below).
   The tables are one sample.
3. **The seeds did nothing here.** Within the sweep, all three seeds produced
   identical per-puzzle results (per-puzzle standard deviation is exactly 0).
   The spread in the tables is entirely *between puzzles*, not between repeats.
   So the repeat machinery was exercised but contributed no variability at this
   configuration, and "± " above should not be read as run-to-run variability.
4. **Baseline coverage is poor.** The method assembles only about a quarter of
   each 10-piece puzzle on average, and fully solved none of the six. These
   puzzles are not the ones the default parameters were tuned on, so this says
   more about transfer to unfamiliar data than about the method's best case.
5. Runtimes are wall-clock on a loaded laptop and should be treated as rough.

---

## Reproducing these runs

### Prerequisites

Python 3.11, `pip install -r requirements.txt`, and the puzzle data. The
datasets used here are not in this repository; they were taken from
`synth_artworks` (see the CNN project's README for the download link) and staged
locally:

```bash
mkdir -p ~/piece-assemble-eval-data/synth_artworks/test
# copy the six alphabetically-first 10-piece test puzzles
cd <synth_artworks>/test
for d in $(ls | grep '^10_' | head -6); do
    cp -r "$d" ~/piece-assemble-eval-data/synth_artworks/test/
done
ls ~/piece-assemble-eval-data/synth_artworks/test > /tmp/puzzles.txt
```

Each puzzle directory needs the piece images/masks plus the `pieces.json`
ground truth that `puzzle_generator/generate.py` writes.

### Running one configuration

```bash
python src/piece_assemble/tools/eval_assembly.py \
    config.yaml /tmp/puzzles.txt ~/piece-assemble-eval-data/synth_artworks/test \
    --repeats 3 --seed 0 \
    --output-dir ~/piece-assemble-eval-data/results --label baseline
```

This writes `baseline_runs.csv` (one row per run) and `baseline_summary.json`
(aggregate plus per-puzzle breakdown) into the output directory.

For the ablations, copy the config and change the one setting named in the
table above, then rerun with a matching `--label`.

### Reproducibility: what you have to fix

Three separate things have to be pinned before two runs agree, and the seed
alone is **not** enough:

| | |
|---|---|
| `--seed N` | Seeds the search's own randomness (this instance's `Random` and numpy's global one). |
| `n_processes: 1` | With several worker processes, verified matches arrive over a queue in whatever order the workers happen to finish, and that order changes the result. |
| `PYTHONHASHSEED=0` | Python randomises string hashing per process, which changes the iteration order of the sets of piece ids the algorithm walks over, which changes the result. |

Verified on commit `4d99771`: with all three pinned, two separate invocations of

```bash
PYTHONHASHSEED=0 python src/piece_assemble/tools/eval_assembly.py \
    serial_config.yaml /tmp/one_puzzle.txt <dataset> --repeats 1 --seed 42
```

produced bit-identical metrics. With `PYTHONHASHSEED` left alone they did not
(`largest_cluster_ratio` 0.7 vs 0.5 on the same puzzle and seed), and neither
did two parallel runs with four processes.

Serial mode costs roughly 2.5× the wall-clock time of four processes (≈42 s vs
≈17 s per 10-piece puzzle here), so it is the right setting for anything whose
exact numbers need to be reproducible, and the wrong one for a large sweep.
