# scripts/

One-off / utility scripts that orchestrate `src/car_damage` modules but
aren't part of the installable package itself (e.g. bulk data download,
dataset sanity checks, generating split files, exporting a checkpoint).

TODO: add scripts as needed. Scripts should call into `src/car_damage/`
rather than duplicating logic, and should not hardcode hyperparameters that
belong in `configs/*.yaml`.
