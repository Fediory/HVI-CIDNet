&nbsp;
# You Only Need One Color Space: An Efficient Network for Low-light Image Enhancement

**Yixu Feng<sup>∗ </sup>, Cheng Zhang<sup>∗ </sup>**, Pei Wang , Peng Wu , Qingsen Yan , Yanning Zhang

<div align="center">

[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/abs/2402.05809v1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-need-one-color-space-an-efficient/low-light-image-enhancement-on-lime)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lime?p=you-only-need-one-color-space-an-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-need-one-color-space-an-efficient/low-light-image-enhancement-on-lol)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol?p=you-only-need-one-color-space-an-efficient)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-need-one-color-space-an-efficient/low-light-image-enhancement-on-lol-v2)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol-v2?p=you-only-need-one-color-space-an-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-need-one-color-space-an-efficient/low-light-image-enhancement-on-lol-v2-1)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol-v2-1?p=you-only-need-one-color-space-an-efficient)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-need-one-color-space-an-efficient/low-light-image-enhancement-on-lolv2-1)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lolv2-1?p=you-only-need-one-color-space-an-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-need-one-color-space-an-efficient/low-light-image-enhancement-on-npe)](https://paperswithcode.com/sota/low-light-image-enhancement-on-npe?p=you-only-need-one-color-space-an-efficient)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-need-one-color-space-an-efficient/image-enhancement-on-sice-grad)](https://paperswithcode.com/sota/image-enhancement-on-sice-grad?p=you-only-need-one-color-space-an-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-need-one-color-space-an-efficient/image-enhancement-on-sice-mix)](https://paperswithcode.com/sota/image-enhancement-on-sice-mix?p=you-only-need-one-color-space-an-efficient)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-need-one-color-space-an-efficient/low-light-image-enhancement-on-sony-total)](https://paperswithcode.com/sota/low-light-image-enhancement-on-sony-total?p=you-only-need-one-color-space-an-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-need-one-color-space-an-efficient/low-light-image-enhancement-on-vv)](https://paperswithcode.com/sota/low-light-image-enhancement-on-vv?p=you-only-need-one-color-space-an-efficient)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-need-one-color-space-an-efficient/low-light-image-deblurring-and-enhancement-on)](https://paperswithcode.com/sota/low-light-image-deblurring-and-enhancement-on?p=you-only-need-one-color-space-an-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-need-one-color-space-an-efficient/low-light-image-enhancement-on-mef)](https://paperswithcode.com/sota/low-light-image-enhancement-on-mef?p=you-only-need-one-color-space-an-efficient)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-need-one-color-space-an-efficient/low-light-image-enhancement-on-dicm)](https://paperswithcode.com/sota/low-light-image-enhancement-on-dicm?p=you-only-need-one-color-space-an-efficient)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/you-only-need-one-color-space-an-efficient/low-light-image-enhancement-on-lolv2)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lolv2?p=you-only-need-one-color-space-an-efficient)

</div>

&nbsp;

## 💡News

- **2024.03.04** Update five unpaired datasets (DICM, LIME, MEF, NPE, VV) visual results. ✨

- **2024.03.03** Update pre-trained weights and output results on our HVI-CIDNet using Baidu Pan. 🧾

- **2024.02.08**  Update HVI-CIDNet in [Arxiv](https://arxiv.org/abs/2402.05809v1). The new code, models and results will be uploaded. 🎈



## 🧾Weights and Results
All the weights that we trained on different datasets is available at  [[Baidu Pan](https://pan.baidu.com/s/1rvQcQPwsYbtLIYwB3XgjaA?pwd=yixu)] (code: `yixu`).  Results on  DICM, LIME, MEF, NPE, and VV datasets can be downloaded from [[Baidu Pan](https://pan.baidu.com/s/1ApI5B1q2GPBHWdh8AafjlQ?pwd=yixu)] (code: `yixu`). 
**Bolded** fonts represent impressive metrics.
- The metrics of HVI-CIDNet on paired datasets are shown in the following table: 

| Folder (test datasets)                        | PSNR        | SSIM       | LPIPS      | GT Mean | Results                                                      | Weights Path             |
| --------------------------------------------- | ----------- | ---------- | ---------- | ------- | ------------------------------------------------------------ | ------------------------ |
| (LOLv1)<br />v1 w perc loss/ wo gt mean       | 23.8091     | 0.8574     | **0.0856** |         | [Baidu Pan](https://pan.baidu.com/s/1k1_oHDLh8oR47r7RTfB4Hw?pwd=yixu) | LOLv1/w_perc.pth         |
| (LOLv1)<br />v1 w perc loss/ w gt mean        | 27.7146     | 0.8760     | **0.0791** | √       | ditto                                                        | LOLv1/w_perc.pth         |
| (LOLv1)<br />v1 wo perc loss/ wo gt mean      | 23.5000     | **0.8703** | 0.1053     |         | [Baidu Pan](https://pan.baidu.com/s/1hMMh8NNqTLJRSZJ6GxI3rw?pwd=yixu) | LOLv1/wo_perc.pth        |
| (LOLv1)<br />v1 wo perc loss/ w gt mean       | **28.1405** | **0.8887** | 0.0988     | √       | ditto                                                        | LOLv1/wo_perc.pth        |
| (LOLv2_real)<br />v2 wo perc loss/ wo gt mean | 23.4269     | 0.8622     | 0.1691     |         | [Baidu Pan](https://pan.baidu.com/s/1Lo19WOrFY3_3wsuJ9gIYnw?pwd=yixu) | (lost)                   |
| (LOLv2_real)<br />v2 wo perc loss/ w gt mean  | 27.7619     | 0.8812     | 0.1649     | √       | ditto                                                        | (lost)                   |
| (LOLv2_real)<br />v2 best gt mean             | **28.1387** | **0.8920** | **0.1008** | √       | [Baidu Pan](https://pan.baidu.com/s/1qewb6u5w1VUaaEzEjFXllQ?pwd=yixu) | LOLv2_real/w_prec.pth    |
| (LOLv2_real)<br />v2 best Normal              | **24.1106** | 0.8675     | 0.1162     |         | [Baidu Pan](https://pan.baidu.com/s/1V9aMZWEU2D0bVRDmPeNzMQ?pwd=yixu) | (lost)                   |
| (LOLv2_real)<br />v2 best PSNR                | 23.9040     | 0.8656     | 0.1219     |         | [Baidu Pan](https://pan.baidu.com/s/1PFQ49oa_n7ywTGLl3TUb3A?pwd=yixu) | LOLv2_real/best_PSNR.pth |
| (LOLv2_real)<br />v2 best SSIM                | 23.8975     | **0.8705** | 0.1185     |         | [Baidu Pan](https://pan.baidu.com/s/1zeBPFJS3HxQ9zZZnYGMn4g?pwd=yixu) | LOLv2_real/best_SSIM.pth |
| (LOLv2_real)<br />v2 best SSIM/ w gt mean     | **28.3926** | 0.8873     | 0.1136     | √       | None                                                         | LOLv2_real/best_SSIM.pth |
| (LOLv2_syn)<br />syn wo perc loss/ wo gt mean | **25.7048** | **0.9419** | 0.0471     |         | [Baidu Pan](https://pan.baidu.com/s/1ZZtalO3vxqSJOJ58BnfUXw?pwd=yixu) | LOLv2_syn/wo_perc.pth    |
| (LOLv2_syn)<br />syn wo perc loss/ w gt mean  | **29.5663** | **0.9497** | 0.0437     | √       | ditto                                                        | LOLv2_syn/wo_perc.pth    |
| (LOLv2_syn)<br />syn w perc loss/ wo gt mean  | 25.1294     | 0.9388     | **0.0450** |         | [Baidu Pan](https://pan.baidu.com/s/1R_ltvaWHJ_sY-unHAEGunw?pwd=yixu) | LOLv2_syn/w_perc.pth     |
| (LOLv2_syn)<br />syn w perc loss/ w gt mean   | 29.3666     | **0.9500** | **0.0403** | √       | ditto                                                        | LOLv2_syn/w_perc.pth     |
| Sony_Total_Dark                               | **22.9039** | **0.6763** | **0.4109** |         | [Baidu Pan](https://pan.baidu.com/s/15w3oMuF3hOtJK29v_xjX3g?pwd=yixu) | SID.pth                  |
| LOL-Blur                                      | **26.5719** | **0.8903** | **0.1203** |         | [Baidu Pan](https://pan.baidu.com/s/11zTPd3xrJe0GbEXF_lYHvQ?pwd=yixu) | LOL-Blur.pth             |
| SICE-Mix                                      | **13.4235** | 0.6360     | 0.3624     | √       | [Baidu Pan](https://pan.baidu.com/s/11x4oJuIKE0iJqdqagG1RhQ?pwd=yixu) | SICE.pth                 |
| SICE-Grad                                     | **13.4453** | 0.6477     | 0.3181     | √       | [Baidu Pan](https://pan.baidu.com/s/1IICeonyuUHcUfTapa4GKxw?pwd=yixu) | SICE.pth                 |

- Performance on five unpaired datasets are shown in the following table: 

| metrics | DICM  | LIME  | MEF   | NPE   | VV    |
| ------- | ----- | ----- | ----- | ----- | ----- |
| NIQE    | 3.79  | 4.13  | 3.56  | 3.74  | 3.21  |
| BRISQUE | 21.47 | 16.25 | 13.77 | 18.92 | 30.63 |

