# PSMAReg test-phase container

Learn2Reg 2026 / PSMAReg submission for the LapIRN method in this repository.

## Design: no code duplication

The container does **not** hold a copy of the method. The build context is the
**repository root**, so `Code/` is copied straight out of the working tree at
build time and put on `PYTHONPATH`:

```
COPY Code /app/lapirn/Code
ENV PYTHONPATH=/app/lapirn/Code
```

`Code/` is a flat set of sibling modules (`import Functions`, `import my_data`,
…), not a package, so nothing needs restructuring and there is no
`pip install -e .` step. Consequences:

* Every build reflects the current commit — there is nothing to keep in sync.
* During development, bind-mounting `Code/` over the baked copy (`DEV=1`, see
  below) applies edits with no rebuild at all.
* A given commit fully determines the image, which is what the challenge's
  reproducibility requirement asks for.

Only `submission/infer.py` is container-specific: a thin adapter from the
organizers' five-argument interface to the method.

## Method in the container

Mirrors `Code/inference.py` with `chase_flag = "chase_best_model"` and
`use_io = False`:

1. body mask + CT/PET normalisation (`my_data.get_body_mask`, `norm_ct`, `norm_pet`)
2. ANTs affine prereg on half-resolution windowed CT (`affine_reg.compute_affine_dvf`)
3. LapIRN level-3 deformable registration on the affine-warped moving image
4. affine composed **outside** the network field into one total transform
5. saved as full-resolution voxel displacements, channel-first `(3, 192, 192, 288)`

Instance optimisation and PET/AutoPET segmentation are intentionally **not**
included yet — they are the next step, once the baseline per-pair runtime is
measured.

## Usage

```bash
bash submission/build.sh                 # docker build from the repo root
DEV=1 LIMIT=1 bash submission/test.sh    # one pair, live code, timing
bash submission/test.sh                  # all 20 validation pairs
bash submission/export.sh                # psmareg_lapirn.tar.gz
```

`test.sh` invokes the container exactly as the organizers do (§4 of the
instructions: `--network=none`, `--user`, read-only input mount) and prints the
per-pair time plus the extrapolation to the 200-pair / 5-hour budget.

Override with environment variables: `IMAGE`, `DATA_DIR`, `OUTPUT_DIR`,
`DATASET_JSON`, `LIMIT`, `DEV`.

## Checking the output makes sense

`evaluate_disp.py` runs on the host (repo venv, not the container) and scores a
directory of container-produced fields with the same metric code as
`Code/inference.py` -- dice and HD95 delegate to `hd95_official` /
`multilabel_dice`, MTV and TLG to `utils.*_bias_loss`, NDV to `ndv_official`:

```bash
python submission/evaluate_disp.py submission/validation_predictions \
    --compare auspicious-sloth-39469081_combined
```

`--compare` prints the corresponding row from
`submission_results/csvs/chase_leaderboard/` next to the new numbers with a
delta. That reference was produced from a half-resolution field, so exact
agreement is not expected; a small positive dice delta is the plausible outcome,
and a large gap in either direction means the field convention is wrong.

Sub-resolution fields are upsampled first, exactly as the organizers' scorer
does, so old submissions and baselines can be scored through the same path.

## Weights

`weights/model.pth` is
`PSMAReg_LapIRN_auspicious-sloth-39469081_stagelvl3_best_combined.pth` (3.6 MB).
It is baked into the image because evaluation runs offline. Replace the file and
rebuild to submit a different checkpoint; the hyper-parameters in
`infer.py:build_config` must match it.
