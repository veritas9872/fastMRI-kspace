# fastMRI-kspace
###Code for cracking the fastMRI challenge.



**Note**: This code is tested on Python 3.6 and 3.7 with Pytorch 1.1 on Ubuntu 16.04 and 18.04 
  
There is no Python 2.7 or 3.5 compatibility. 

Pytorch 1.0 and below is also not supported.


# UPDATE #
We won 6th place on the fastMRI Challenge as measured by SSIM. See https://fastmri.org/leaderboards/challenge for our results.

Despite being heavily outgunned by our GPU rich competitors, 
our submission is highly competitive with the top performers on SSIM, the main metric of the challenge.

All of our models were trained on a *single* GTX1080Ti or RTX2080Ti device *using the entire dataset*.

Training took approximately 1 week for 100 epochs.

Our work has been accepted for publication at ISBI 2020.

Title: "Deep Learning Fast MRI using Channel Attention in Magnitude Domain".

Our paper is available at IEEE Xplore [here](https://ieeexplore.ieee.org/document/9098416).

Please cite our work as follows:

`
@INPROCEEDINGS{9098416,  author={J. {Lee} and H. {Kim} and H. {Chung} and J. C. {Ye}},  
booktitle={2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI)},   
title={Deep Learning Fast MRI Using Channel Attention in Magnitude Domain},   
year={2020},  volume={},  number={},  pages={917-920},}
`

---

Sorry if the project appears a bit messy. 

This project contains all the code that we used during the entire challenge. 

Each branch has different ideas from each of the members.

Please contact the authors if you have any questions.
