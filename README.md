# Piece Assembly

This is a Python package for assembly of images broken into pieces (e.g. jigsaw puzzle).

## Setup
Requires Python 3.11 or later.

```
pip install -r requirements.txt
```

## Usage
Images needs to be already preprocessed prior to using this project.
Each piece is represented as a pair of images
`<id>.jpg` and `<id>_mask.png`, see the `data` directory for reference.

```
python src/piece_assemble/tools/run_assembly.py /path/to/config
```

This repo contains one sample configuration `sample_config.yaml`, which also contains
explanation of parameters and can
to run the assembly of `data/owl_101pcs` as

```
python src/piece_assemble/tools/run_assembly.py sample_config.yaml
```

`sample_config.yaml`'s parameters are tuned for `data/owl_101pcs`. Some of them
do not transfer well to datasets whose pieces have differently shaped borders;
`synth_artworks_10pcs_config.yaml` is a variant tuned for smaller synthetic
puzzles with straighter cuts. See `docs/evaluation-results.md` for the
measurements behind it.

## Evaluating Assembly

The assembly pipeline can be evaluated against ground truth puzzles using
`src/piece_assemble/tools/eval_assembly.py`:

```
python src/piece_assemble/tools/eval_assembly.py CONFIG PUZZLES_PATH DATASET_PATH
    [--repeats N] [--seed N] [--output-dir DIR] [--label NAME]
```

- `CONFIG`: a configuration file with the same structure as used for
  `run_assembly.py`. Only `img_path` is overridden per evaluated puzzle.
- `PUZZLES_PATH`: a text file listing puzzle directory names (one per line,
  relative to `DATASET_PATH`) to evaluate.
- `DATASET_PATH`: directory containing those puzzle directories. Each one must
  hold piece images/masks plus a ground truth `pieces.json`, in the same format
  produced by `puzzle_generator/generate.py` (a `transformed_pieces` list of
  per-piece ground truth transformations, and a `neighbors` list of adjacent
  piece-id pairs).

For every listed puzzle, the pipeline is run and its best assembly is compared
to the ground truth via `piece_assemble.evaluation.evaluate`, which reports
mean rotation/translation registration error over correctly-matched neighbor
pairs, the fraction of pieces in the largest correctly-assembled cluster, and
how many distinct clusters the pieces ended up split across. Per-puzzle and
aggregated metrics are printed as CSV.

The assembly search is stochastic, so `--repeats N` assembles each puzzle N
times using consecutive seeds starting from `--seed`, and metrics are reported
as mean and standard deviation. `--output-dir` additionally writes the
individual runs and an aggregated summary to disk.

Note that `--seed` alone does not make a run repeatable: `n_processes: 1` and
`PYTHONHASHSEED=0` are needed as well, for the reasons given in the script's
own documentation. Results from past runs are recorded in
`docs/evaluation-results.md`.
