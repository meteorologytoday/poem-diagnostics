#!/bin/bash

python3 run_diagnostics.py --config config/minimal_setup_CTL.yaml \
                           --modes annual \
                           --components atmos land \
                           --diag-types "map_2d" "timeseries" "zonal_section" \
                           --year-range 2107 2307
