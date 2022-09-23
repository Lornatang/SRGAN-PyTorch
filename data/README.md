# Usage

## Step1: Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

## Step2: Prepare the dataset in the following format

```text
# Train dataset struct
- ImageNet
    - original
        - image001.JPEG
        - image002.bmp
        ...

# Test dataset struct
- Set5
    - GTmod12
        - baby.png
        - bird.png
        - ...
    - LRbicx4
        - baby.png
        - bird.png
        - ...
```

## Step3: Preprocess the train dataset

```bash
cd <SRGAN-PyTorch-main>/scripts
python run.py
```

## Step4: Check that the final dataset directory schema is completely correct

```text
# Train dataset
- ImageNet
    - SRGAN
        - train
        - original

# Test dataset
- Set5
    - GTmod12
        - baby.png
        - bird.png
        - ...
    - LRbicx4
        - baby.png
        - bird.png
        - ...

```

