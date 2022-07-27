# Inverse Optimal Control Adapted to the Noise Characteristics of the Human Sensorimotor System

![agent_environment010](https://user-images.githubusercontent.com/23743923/136335919-64f8a3cc-078e-4d82-aaad-5a90035c942b.png)

This repository contains code for the paper

Schultheis, M., Straub, D., & Rothkopf, C. A. (2021). Inverse Optimal Control Adapted to the Noise Characteristics of the Human Sensorimotor System. 35th Conference on Neural Information Processing Systems (NeurIPS 2021).

## Install requirements

```bash
python -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
```

## Running the examples

```bash
python reaching_example.py
python random_example.py
python belief_tracking.py
python eye_example.py
```

## Citation
If you use our method in your research, please cite our paper:

```
@article{schultheis2021inverse,
  title={Inverse optimal control adapted to the noise characteristics of the human sensorimotor system},
  author={Schultheis, Matthias and Straub, Dominik and Rothkopf, Constantin A},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={9429--9442},
  year={2021}
}
```
