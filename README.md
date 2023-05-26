# This is a simple text image super-resolution package.
[![PyPI](https://img.shields.io/pypi/v/textbsr)](https://pypi.org/project/textbsr/)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=csxmli2016/textbsr) 
[![Citation](https://img.shields.io/badge/Citation-bibtex-green)](https://github.com/csxmli2016/textbsr/blob/main/README.md#bookmark_tabs-citation)

This package can post-process the text region with a simple command, i.e., 
```
textbsr -i [LR_TEXT_PATH] -b [BACKGROUND_SR_PATH]
```

> - [LR_TEXT_PATH] is the LR image path.
> - [BACKGROUND_SR_PATH] stores the results from any blind image super-resolution methods.
> - If the text image is degraded severely, this method may still fail to obtain a plausible result.

### Dependencies and Installation
- numpy
- cnstd
- opencv-python
- torch>=1.8.1
- torchvision>=0.9

``` 
# Install with pip
pip install textbsr
```


### Basic Usage

```
# On the terminal command
textbsr -i [LR_TEXT_PATH]
```
or
```
# On the python environment
from textbsr import textbsr
textbsr.bsr(input_path='./testsets/LQs')
```

Parameter details:

| parameter name | default | description  |
| :-----  | :-----:  | :-----  |
| <span style="white-space:nowrap">-i, --input_path </span>|  | The lr text image path. It can store full images or text layouts only. |
| <span style="white-space:nowrap">-b, --bg_path</span> | None | The background sr path from other methods. If None, we only restore the text region detected by cnstd.|
| <span style="white-space:nowrap">-o, --output_path</span> | None | The save path for text sr result. If None, we save the results on the same path with the format of [input_path]\_TIMESTAMP.|
| <span style="white-space:nowrap">-a, --aligned </span>| False | action='store_true'. If True, the input text image contains only text region. If False, we use CnSTD to detect text regions and then restore them.|
| <span style="white-space:nowrap">-s, --save_text </span>| False | action='store_true'. If True, save the LR and SR text layout.|
| <span style="white-space:nowrap">-d, --device</span> | None | Device, use 'gpu' or 'cpu'. If None, we use torch.cuda.is_available to select the device. |

### Example for post-processing the text region
```
# On the terminal command
textbsr -i [LR_TEXT_PATH] -b [BACKGROUND_SR_PATH] -s
```

or
```
# On the python environment
from textbsr import textbsr
textbsr.bsr(input_path='./testsets/LQs', bg_path='./testsets/RealESRGANResults', save_text=True)
```
> When [BACKGROUND_SR_PATH] is None, we only restore the text region and paste it back to the LR input, with the background region unchanged.




| Real-world LR Text Image | Real-ESRGAN | Post-process using our textbsr | 
| :-----:  | :-----:  | :-----:  |
|<img src="./GitImgs/LR/test1.jpg" width="250px"> | <img src="./GitImgs/RealESRGAN/test1.jpg" width="250px">|<img src="./GitImgs/Ours/test1_BSRGANText.png" width="250px"> |
|<img src="./GitImgs/LR/test42.png" width="250px"> | <img src="./GitImgs/RealESRGAN/test42.png" width="250px">|<img src="./GitImgs/Ours/test4_BSRGANText2.png" width="250px"> |
|<img src="./GitImgs/LR/00426.png" width="250px"> | <img src="./GitImgs/RealESRGAN/00426_out.png" width="250px">|<img src="./GitImgs/Ours/00426_BSRGANText.png" width="250px"> |

From top to bottom: text regions from LR input, RealESRGAN, and post-process using Ours:
<img src="./GitImgs/LR/00426_patch_0i.png" width="750px"> 
<img src="./GitImgs/RealESRGAN/00426_out_crop0.png" width="750px">
<img src="./GitImgs/Ours/00426_patch_0o.png" width="750px">

---

### Example for restoring the aligned text region
```
# On the terminal command
textbsr -i [LR_TEXT_PATH] -a
```
or
```
# On the python environment
from textbsr import textbsr
textbsr.bsr(input_path='./testsets/LQs', aligned=True)
```


| Aligned LR Text Image | Our result |
| :-----:  | :-----:  |
| <img src="./GitImgs/Ours/test5_patch_5i.png" width="250px"> | <img src="./GitImgs/Ours/test5_patch_5o.png" width="250px"> |


## Acknowledgement
This project is built based on [BSRGAN](https://github.com/cszn/BSRGAN) and [cnstd](https://github.com/breezedeus/CnSTD).

## :bookmark_tabs: Citation
If you find this package helpful, please kindly consider citing our paper:
```
@InProceedings{li2023marconet,
author = {Li, Xiaoming and Zuo, Wangmeng and Loy, Chen Change},
title = {Learning Generative Structure Prior for Blind Text Image Super-resolution},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year = {2023}
}
```

## :scroll: License

This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>
