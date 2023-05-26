## This is a simple text image super-resolution package.

---

## Quick Start
### Dependencies and Installation
- numpy
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
| <span style="white-space:nowrap">-i, --input_path </span>| - | The lr text image path. It can be a full image or a text region only |
| <span style="white-space:nowrap">-b, --bg_path</span> | None | The background sr path from other methods. If None, we only super-resolve the text region.|
| <span style="white-space:nowrap">-o, --output_path</span> | None | The save path for text sr result. If None, we save the results on the same path with [input_path]_TIMESTAMP|
| <span style="white-space:nowrap">-a, --aligned </span>| False | action='store_true'. If True, the input text image contains only text region. If False, we use CnSTD to detect and restore the text region.|
| <span style="white-space:nowrap">-s, --save_text </span>| False | action='store_true'. If True, save the LR and SR text layout.|
| <span style="white-space:nowrap">-d, --device</span> | None | Device, use 'gpu' or 'cpu'. If None, we use torch.cuda.is_available to select the device. |
| <span style="white-space:nowrap">-t, --box_score_thresh </span>  |0.35|threshold for CnSTD detecting the text region.|

### Example


> If you find this package helpful, please kindly consider to cite our paper:
```
@InProceedings{li2023marconet,
author = {Li, Xiaoming and Zuo, Wangmeng and Loy, Chen Change},
title = {Learning Generative Structure Prior for Blind Text Image Super-resolution},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year = {2023}
}
```
