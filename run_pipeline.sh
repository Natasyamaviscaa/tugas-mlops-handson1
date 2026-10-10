#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"
python -m src.basic_tracking
python -m src.autologging
python -m src.deep_learning --epochs "${EPOCHS:-15}"
python -m src.tune --trials "${TRIALS:-20}"
python -m src.compare_runs
python -m src.register
