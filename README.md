# MultiBaselineDepth
This repository contains the source code of the paper "Unsupervised Monocular Depth Estimation with Multi-Baseline Stereo", BMVC 2020.
If you use this code in your projects, please cite our paper:

```
@inproceedings{imran2020unsupervised,
  title={Unsupervised Monocular Depth Estimation with Multi-Baseline Stereo},
  author={Imran, Saad and Kyung, Chong-Min and Mukaram, Sikander and Karim Khan, Muhammad Umar},
  booktitle={The 31st British Machine Vision Conference},
  year={2020},
  organization={British Machine Vision Virtual Conference}
}
```

For more details:
[paper](https://www.bmvc2020-conference.com/assets/papers/0975.pdf)


## Usage


For training, run the following command
<pre><code>python3 multimain.py --data_path "path to images folder" --filenames_file "path to text file containing images names"</code></pre>
Our code is based on Monodepth. Please see [monodepth](https://github.com/mrharicot/monodepth) for requirements and more details.
## Dataset
The CARLA and small objects datasets can be downloaded from [link](https://drive.google.com/drive/u/1/folders/1xoXOQn3126eArmoUCz3N0VEUzqarqDQo).

## Contact
If you have any questions about the code and dataset, feel free to contact at saadimran@kaist.ac.kr
