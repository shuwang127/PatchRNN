# SecurityPatchIdentificationRNN

**Task**: Security Patch Identification using RNN model.

**Developer**: Shu Wang

**Date**: 2020-07-11

**Version**: S2020.07.22-V2

**Description**: patch identification using commit messages.

**File Structure**:

    |-- SecurityPatchIdentificationRNN
        |-- analysis                                # task analysis.
        |-- data                                    # data storage.
                |-- negatives                           # negative samples.
                |-- positives                           # positive samples.
                |-- security_patch                      # positive samples. (official)
        |-- temp                                    # temporary stored variables.
                |-- data.npy                            # raw data. (important)
                |-- props.npy                           # properties of diff code. (important)
                |-- ...                                 # other temporary files. (trivial)
        |-- SecurityPatchIdentificationRNN.ipynb    # main entrance. (Google Colaboratory)
        |-- SecurityPatchIdentificationRNN.py       # main entrance. (Local)

**Dependencies**:
```shell script
pip install clang == 6.0.0.2
pip install torch == 1.2.0 torchvision == 0.4.0
pip install nltk  == 3.3
```

**Usage**:
```shell script
python SecurityPatchIdentificationRNN.py
```
