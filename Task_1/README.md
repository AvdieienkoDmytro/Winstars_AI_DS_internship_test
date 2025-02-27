# Winstars Task 1

## Overview

For this task, I implemented 3 models that achieved sufficient results. The RF model achieved 96-97% accuracy. NN got around 97%. And as expected, the best result came from CNN - 99%. For DNN and CNN, I drew a learning graph and noticed that the model overfits after 5-10 epochs. Before I added the Dropout layer, the model was overfitting after just 2 epochs. Therefore, 10 epochs are quite sufficient for this model.

## Project Structure
```
task1/
├── .venv/                     # Virtual environment
├── notebooks/                 # Jupyter notebooks
│   └── task_1.ipynb          # Demo notebook
├── requirements.txt          # Project dependencies
└── README.md                # Documentation
```
## Installation and Setup
### Requirements
Python 3.10.10

### How to run

1. Change directory
    ```cd ...```
2. Create virtual environment
    ```bash
   py -3.10 -m venv .venv
   ```
3. Activate virtual environment:

    for Linux/MacOS:
    ```bash
   . .venv/bin/activate
   ```
   for Windows(Powershell):
   ```shell
   .venv\Scripts\Activate.ps1
   ```
4. Install dependencies
    ```bash
   pip install -r requirements.txt
   ```
5. Start jupyter server
    ```bash
   jupyter notebook
   ```