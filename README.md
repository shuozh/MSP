# MSP
Multi-Stream Progressive Restoration for Low-Light Light Field Enhancement and Denoising

CONTACT: [Shuo Zhang](http://faculty.bjtu.edu.cn/9278/, https://shuozh.github.io/)  
(zhangshuo@bjtu.edu.cn)

Any scientific work that makes use of our code should appropriately mention this in the text and cite our TCI 2023 paper. For commercial use, please contact us.

### PAPER TO CITE:

Xianglang Wang, Youfang Lin, Shuo Zhang*.  
Multi-Stream Progressive Restoration for Low-Light Light Field Enhancement and Denoising, 
IEEE Transactions on Computational Imaging (TCI), 2023

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
