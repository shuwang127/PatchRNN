# PatchClassificationRNN

PatchClassificationRNN: Security Patch Classification using RNN model.

Developer: Shu Wang

Date: 2020-07-11

File Structure:

    |-- PatchClearance
        |-- analysis                        # task analysis.
        |-- data                            # data storage.
                |-- negatives                   # negative samples.
                |-- positives                   # positive samples.
                |-- security_patch              # positive samples. (official)
        |-- temp                            # temporary stored variables.
                |-- data.npy                    # raw data. (important)
                |-- props.npy                   # properties of diff code. (important)
                |-- ...                         # other temporary files. (trivial)
        |-- PatchClassificationRNN.ipynb    # main entrance. (Google Colaboratory)
        |-- PatchClassificationRNN.py       # main entrance. (Local)

Usage:
    python PatchClassificationRNN.py

```shell script
pip install clang
``` 