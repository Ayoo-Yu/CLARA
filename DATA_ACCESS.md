# Data access and reuse

The MIT license applies to the authors' software and accompanying software documentation. It does not grant rights in third-party datasets, the commercial-farm outputs, or the journal manuscript. Dependencies retain their own licenses.

## GEFCom 2014

Obtain the wind-track data from the original GEFCom 2014 distribution associated with [Hong et al. (2016)](https://doi.org/10.1016/j.ijforecast.2016.02.001). The [IEEE PES dataset index](https://ieee-pes-data-sharing.org/datasets/detail/0e87366e-2e91-4024-b658-43f6b22faa69) identifies the dataset and its access source. Original benchmark files are not redistributed here. Follow the source's applicable access and reuse terms, and cite the dataset paper.

The repository contains derived condition-level performance summaries. The separate GEFCom policy-evidence archive contains the fitted candidate evidence and selections used to verify CLARA's final state decisions. Neither should be described as raw weather or power measurements.

## Commercial wind farms

The reproduction archive contains anonymized candidate-level outputs, including per-unit observations, median forecasts, interval endpoints and saved choices. It preserves all evaluated farms, seeds, prices and forecasting conditions. It omits raw SCADA/NWP records, model-training records, site and device identifiers, physical capacities, calendar dates and the private mapping back to the farms. The anonymization does not perturb the numerical values used by the evaluation.

Commercial outputs remain subject to the data owners' permission. This package does not assign a general data license or authorize redistribution on their behalf. Public distribution must follow the permission obtained for these specific anonymized outputs. A software license alone is not data-release permission.

See `data/commercial/DATA_DICTIONARY.md` for units, joins, relative time indices and the distinction between model-power-base normalization and nameplate capacity. Do not attempt to infer or reconstruct the farm identities or original calendar alignment.
