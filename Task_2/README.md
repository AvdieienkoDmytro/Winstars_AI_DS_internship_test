# Winstars Task 2

## Project Structure
```
task2/
├── .venv/                     # Virtual environment
├── notebooks/                 # Jupyter notebooks
│   └── task_2.ipynb          # Demo notebook
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