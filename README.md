## GLoG-CSUnet: Enhancing Vision Transformers with Adaptable Radiomic Features for Medical Image Segmentation


---
This repository contains the benchmarks, results, and all the code required to reproduce the results, tables, and figures presented in our paper.


------
**1. Prepare  dataset.**

create a folder named "data" in the root directory of the project and download the datasets from the following links:
- Synapse dataset can be found at [the repo of TransUnet](https://github.com/Beckschen/TransUNet). 

- ACDC dataset can be found at [the repo of MT-Unet](https://drive.google.com/file/d/13qYHNIWTIBzwyFgScORL2RFd002vrPF2/view?usp=sharing)

place the downloaded datasets in the `data\ACDC` or `data\Synapse` folder.

**2. Clone the code**

- First, clone our code with:
```
git clone git@github.com:HAAIL/GLoG-CSUnet.git
cd GLoG-CSUnet
```

**3. Install requirements**
- Install the required packages with:
```
pip3 install -r requirements.txt
```
**4. Start training**

- After that, you can start training with:
```
python3  train_CSUnet_ACDC.py 
```
or 
```
python3  train_CSUnet_Synapse.py 
```

The weights will be saved to "./checkpoint/" 

### Bibtex

```
@inproceedings{eghbali2025conformaldqn,
  title={GLoG-CSUnet: Enhancing Vision Transformers with Adaptable Radiomic Features for Medical Image Segmentation},
  author={Eghbali, Niloufar and Bagher-Ebadian, Hassan and  Alhanai, Tuka and Ghassemi, Mohammad M},
  booktitle={IEEE international conference on acoustics, speech and signal processing (ICASSP)},
  year={2025}
}
```
## References
[CS-Unet](https://github.com/kathyliu579/CS-Unet/)<br>
[Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)<br>
[TransUnet](https://github.com/Beckschen/TransUNet)
