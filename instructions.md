# NSTSC - Instructions

This document provides step-by-step instructions for setting up and running the code in this repository.

## Setup Instructions

1. Install `uv` (a Python package installer and environment manager):
    ```
    pip install uv
    ```

2. Clone the repository:
    ```
    git clone https://github.com/AdityaKharbanda-eaton/NSTSC
    ```

3. Get open source data zip file from [here](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) and extract it to the NSTSC repo. Pssword is 'someone' (without the single quotes.)

4. Change directory to the Codes/ folder within NSTSC:
    ```
    cd NSTSC/Codes
    ```

5. Initialize Python 3.13 (install Python 3.13 if you haven't already):
    ```
    uv init --python 3.13
    ```

6. Create a virtual environment:
    ```
    uv venv
    ```

7. Install required packages:
    ```
    uv add -r requirements.txt
    ```

8. Activate the virtual environment:
    - On Windows:
      ```
      source .venv/Scripts/activate
      ```
    - On Linux/Mac:
      ```
      source .venv/bin/activate
      ```

9. Pull the latest changes:
    ```
    git pull
    ```

10. Switch to the CPU branch:
     ```
     git checkout CPU
     ```

## Running the Code

Once you have set up the environment, you can run the main script with:

```
uv run NSTSC_main.py dataset_name epochs
```

Where:
- `dataset_name`: The name of the dataset to use (enter the exact name as it is written in the UCR Archive folder)
- `epochs`: The number of training epochs

## Example

```
uv run NSTSC_main.py Coffee 100
```