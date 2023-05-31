## Dataset

If you want to use the `llf_dataset(Dataset)` in the `utils`, the dataset directory should be oraganized as following:
```
- data_dir
    - train
        - 1 (GT)
        - 1_100
        - 1_50
        - 1_20
    - test
        - 1 (GT)
        - 1_100
        - 1_50
        - 1_20
```
We use the `.npy` format file when loading the data and the range of pixel value is from 0 to 255.





## How to run?
- prepare the dataset
- exec the testing command.
```
python test.py
```

> The `weights` contains the pre-trained weights of MSP-3 which is trained with with 5 $\times$ 5 LF. If you want to try 7 $\times$ 7 or other angular resolutions, please change the `n_view` and re-train the model.