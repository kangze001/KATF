# KATF

**A Method for Malicious Bot Traffic Behavior Recognition**

## Dependencies

This project is implemented using PyTorch and has been tested on the following hardware and software configuration:

* **GPU:** NVIDIA GeForce A5000
* **CUDA Version:** 11.7
* **PyTorch Version:** 2.5.1
* **Environment Management:** Anaconda3

## Datasets

We will publicly release our self-simulated **Bot Traffic Dataset** after the paper is published.

## Experiments

For offline testing or real-time deployment using our self-built Bot Traffic Dataset, please use the **MSTF (Multi-Scale Traffic Temporal Fingerprinting)** features for training and inference. This is necessary to validate the real-time testing capabilities of our KATF model.

However, for public datasets such as **CW, OW, WTF-PAD, Front, Walkie-Talkie, and TrafficSliver**, you **must not** use the MSTF features, as they will not be suitable. For these public datasets, only the **raw time-series features** should be used.

## Contact

If you have any questions or suggestions, feel free to contact:

Ze Kang (kangzekangze@gmail.com or ise\_kangze@stu.ujn.edu.cn)
