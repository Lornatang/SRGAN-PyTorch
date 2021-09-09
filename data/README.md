# Usage

```bash
# Default use DIV2K dataset.
$ bash download_dataset.sh
```

## Training Dataset directory structure

When training DIV2K, the dataset should be placed in the following directory.

```text
- DIV2K
    - train
        - 0000.png
        - 0001.png
        - 0002.png
        - ...
    - valid 
        - 1000.png
        - 1001.png
        - 1002.png
        - ...
```

## Testing Dataset directory structure

If you need to test Set5, Set14, place the dataset in the following way.

### Set 5

```text
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
