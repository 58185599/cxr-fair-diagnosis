# A Fairer Medical Model for Diagnosis on Chest X-ray

This repository contains the implementation of our models described in the work **"A Fairer Medical Model for Diagnosis on Chest X-ray"**. We explore fairness-aware deep learning techniques using the MIMIC-CXR dataset with image embeddings, comparing empirical risk minimization (ERM) with Group Distributionally Robust Optimization (GroupDRO) to mitigate bias in medical diagnosis tasks.

##  Project Structure

- `ERM-MLP.ipynb`: Implements the ERM-based baseline model using an MLP architecture.
- `debias.ipynb`: Implements a debiased model using GroupDRO for fairness-aware training.
- `Convert-F1.py`: Script to compute the optimal F1 threshold and convert prediction probabilities to binary 0/1 outcomes.
- `example.csv`: A sample input file illustrating the format of patient embedding vectors and metadata.
- `latest_mlp_model.pth`: A trained model that can be loaded in `debias.ipynb` for direct inference.
- `convert.ipynb`: Converts raw .tfrecord files from the MIMIC-CXR embedding dataset into .npy files using provided path information. Can be parallelized using multiprocessing.
##  Requirements

- Python 3.8+
- PyTorch >= 1.10
- numpy
- pandas
- scikit-learn
- tqdm
- matplotlib
- seaborn

Install all required packages via:

```bash
pip install -r requirements.txt
