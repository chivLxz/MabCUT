# Multi-attention bidirectional contrastive learning method for unpaired image-to-image translation (MabCUT)

![framework](.\imgs\framework.png)



## ![result](.\imgs\result.png)rerequisites

Python 3.6 or above.

For packages, see requirements.txt.

### Getting started

- Install PyTorch 1.6 or above and other dependencies (e.g., torchvision, visdom, dominate, gputil).

  For pip users, please type the command `pip install -r requirements.txt`.

  For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.


### MabCUT Training and Test

- Download the `grumpifycat` dataset 
```bash
bash ./datasets/download_cut_dataset.sh grumpifycat
```
The dataset is downloaded and unzipped at `./datasets/grumpifycat/`.

- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

Train the MabCUT model:
```bash
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra_MabCUT 
```

The checkpoints will be stored at `./checkpoints/grumpycat_MabCUT/`.

- Test the MabCUT model:
```bash
python test.py --dataroot ./datasets/horse2zebra --name horse2zebra_MabCUT
```

The test results will be saved to an html file here: `./results/horse2zebra_MabCUT/latest_test/`.

### [Datasets](./docs/datasets.md)
Download CUT/CycleGAN/pix2pix datasets and learn how to create your own datasets.

When preparing the CityScape dataset, please use Pillow=5.0.0 to run prepare_dataset.py for consistency. 

