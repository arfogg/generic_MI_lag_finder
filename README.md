# generic_MI_lag_finder

This package is intended to determine coupling timescales between two sets of data or 'timeseries'. The Mutual Information (MI) content between the two timeseries is calculated at a range of different temporal lags. MI is a measure of the shared information content between two variables, independent of the order or direction of their relationship. A piecewise linear and quadratic curve are fit to the MI data as a function of temporal lag; the peak of these curves can be interpreted as the coupling timescale between the two measured phenomena.

**License:** CC0-1.0

**Support:** please [create an issue](https://github.com/arfogg/generic_MI_lag_finder/issues) or contact [arfogg](https://github.com/arfogg) directly. Any input on the code / issues found are greatly appreciated and will help to improve the software.

## Required Packages

scipy, matplotlib, numpy, pandas, datetime, sklearn, aaft

[Install aaft from its github here](https://github.com/lneisenman/aaft)


## Running the code

First, the code must be downloaded using `git pull`

Then, from the terminal:
`import generic_mutual_information_routines`

To ensure all packages are installed, and the code is working correctly run:
`generic_mutual_information_routines.test_mi_lag_finder()`

This testing function will generate and plot two example signals, timeseries A and B:
![alt text](https://github.com/arfogg/generic_MI_lag_finder/test_example_timeseries.png "Timeseries A and B")

![alt text][logo]
[logo]:https://github.com/arfogg/generic_MI_lag_finder/test_example_timeseries.png "Timeseries A and B"

It will then run `mi_lag_finder`, and plot out the MI content as a function of applied lag:
![alt text](https://github.com/arfogg/generic_MI_lag_finder/test_example_MI.png "MI as a function of lag")

## Acknowledgements

ARF gratefully acknowledges the support of Science Foundation Ireland Grant 18/FRL/6199 and Irish Research Council Government of Ireland Postdoctoral Fellowship GOIPD/2022/782.


