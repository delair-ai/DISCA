
<img src="https://github.com/delair-ai/disca/blob/master/imgs/logo-delair.png" alt="drawing" width="200" align="left"/>

<img src="https://github.com/delair-ai/disca/blob/master/imgs/logo-onera.png" alt="drawing" width="200"  align="right"/>

<br />

# Presentation
This repository contains the code of **DISCA** from our [paper](https://arxiv.org/pdf/2009.11250.pdf): Interactive Learning for Semantic Segmentation in Earth Observation. In a nutshell, it consists in neural networks trained to perform semantic segmentation with human guidance. This builds on our previous work [DISIR](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-2-2020/877/2020/isprs-annals-V-2-2020-877-2020.pdf).

 This repository is divided into two parts:
 - `train` which contains the training code of the networks ([README](./train/README.md))
 - `qgs_plugin` which contains the code of the QGIS plugin used to perform the interactive segmentation ([README](./qgis_plugin/README.md))

# Install Python dependencies

```
conda create -n disca python=3.7 rtree gdal=2.4 opencv scipy shapely -c 'conda-forge' 
conda activate disca
pip install -r requirements.txt
```

 # To use
 Please note that this repository has been tested on Ubuntu 18.4, QGIS 3.8 and python 3.7 only.

1. Download a segmentation dataset such as [ISPRS Potsdam](http://www2.isprs.org/commissions/comm3/wg4/data-request-form2.html) or [INRIA dataset](https://project.inria.fr/aerialimagelabeling/download/).
2. Prepare this dataset according to `Dataset preprocessing` in `train/README.md`.
3. Train a modelstill following `train/README.md`.
4. Install the [QGIS](https://www.qgis.org/en/site/) plugin following `qgs_plugin/README.md`.
5. Follow `How to start` in `qgs_plugin/README.md` and start segmenting your data !

 # References

If you use this work for your projects, please take the time to cite our [ECML-PKDD MACLEAN Workshop paper](https://drive.google.com/file/d/11DzAwKGPvGvC7kOtN3FiqVZVAAFGU4-X/viewf):

```
@inproceedings{lenczner2020interactive,
author = {Lenczner, G. and Chan-Hon-Tong, A. and Luminari, N. and Le Saux, B. and Le Besnerais, G.},
title = {Interactive Learning for Semantic Segmentation in Earth Observation},
booktitle = {ECML-PKDD MACLEAN Workshop},
year = {2020}
}
```

 
 # Licence

Code is released under the MIT license for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

See [LICENSE](./LICENSE) for more details.

# Authors

See [AUTHORS.md](./AUTHORS.md)

# Acknowledgements

This work has been jointly conducted at [Delair](https://delair.aero/)  and [ONERA-DTIS](https://www.onera.fr/en/dtis).