# MID-LTP-3348
Scripts and jupyter notebooks for running xpcs analysis and plotting the analysed runs at MID experiment at the European XFEL.


### Script for checking the analysed runs
The script checks what runs in the given range are anaysed by the `fast_xpcs` pipeline.
Run the script with the following command, where the first and last run number (`-rf` and `-rl`) are given as arguments.

`python scripts/check_analysed.py -rf {run1} -rl {run2}`
