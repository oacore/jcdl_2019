# Do Authors Deposit on Time? Tracking Open Access Policy Compliance

This repository contains supporting data for our JCDL 2019 paper titled "[Do Authors Deposit on Time? Tracking Open Access Policy Compliance](https://ieeexplore.ieee.org/document/8791195)."

The dataset used in our analysis is available at [http://dx.doi.org/10.5281/zenodo.2605409](http://dx.doi.org/10.5281/zenodo.2605409).

All code in this repository uses **Python 3+** (it was tested using Python 3.7.2).

## Repository organization

- **`notebooks/`:** jupyter notebooks
- **`output/`:** generated analysis (figures, etc.)
- **`src/`:** Python source codes which are intended to run locally
- **`config.json.example`:** project configuration such as paths, instructions on how to use this file are provided in the Setup section below
- **`README.md`:** this file
- **`requirements.txt`:** project dependencies
- **`run_server.py`:** script for running an API service for checking compliance

## Setup

1. Install dependencies using ``pip install --upgrade pip && pip install -r requirements.txt``
2. Copy `config.json.example` to `config.json` and edit as needed
3. Run the application (any of the notebooks in the ``notebooks/`` directory)

## Compliance Checker

To get publication and deposit dates and check compliance with the REF 2021 OA Policy for your own DOIs, you can use our compliance checker tool. An example how to use it directly from Python is provided in [this notebook](https://nbviewer.jupyter.org/github/oacore/jcdl_2019/blob/master/notebooks/03_compliance_checker.ipynb).

Alternatively, you can run the tool as a simple API service. Use `python run_server.py` to run it.

To test the functionality of the API, try the following two commands:

```bash
curl -i -X GET http://localhost:8124/check_compliance?doi=10.1002/14651858.CD012515
```

```bash
curl -i -X POST -H 'Content-Type: application/json' \
-d '["10.1002/14651858.CD012515", "10.1007/s11192-018-2669-y", "10.0000/abcd-efgh-ijkl", "10.1145/3057148.3057154", 12345]' \
http://localhost:8124/check_compliance
```
