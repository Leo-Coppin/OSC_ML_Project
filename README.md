# OSC_ML_Project


to create environment run
```bash
conda create -n {env name} python=3.11
conda activate {env name}
conda install numpy
conda install -c conda-forge rdkit
conda install -c conda-forge mordred
conda install scikit-learn
conda install tensorflow
python patch_mordred.py
```

python patch_mordred.py is because rdkit and mordred use two different versions of Numpy