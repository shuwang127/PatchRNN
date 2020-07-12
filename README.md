# PatchClassificationRNN

PatchClassificationRNN: Security Patch Classification using RNN model.

Developer: Shu Wang

Date: 2020-07-11

File Structure:

PatchClearance

    |-- analysis                        # found samples need to be judged.
    |-- data
            |-- negatives
            |-- positives
            |-- security_patch
    |-- temp                            # temporary stored variables.
            |-- data.npy
            |-- props.npy
            |-- ...
    |-- PatchClassificationRNN.ipynb    # extract features for random_commit and security_patch.
    |-- PatchClassificationRNN.py       # extract features for random_commit and security_patch.
    |-- README.md                       # this file.

Usage:
    python PatchClassificationRNN.py

```shell script
pip install clang
``` 