# Medical Insurance Cost Prediction

This repository contains a Python-based project for predicting medical insurance costs using various machine learning models. The goal of this project is to build and evaluate models that can predict the insurance costs of individuals based on their features such as age, gender, BMI, number of children, smoking status, and region.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)

## Project Overview

The project aims to explore and analyze a dataset related to medical insurance costs. By understanding the relationships between different features and insurance costs, we can create models that accurately predict costs. The project utilizes various machine learning algorithms, including linear regression, decision trees, and random forests, among others.

## Dataset

The dataset used in this project is a mock medical insurance dataset containing the following columns:

- **age**: Age of the individual
- **sex**: Gender of the individual (`male`, `female`)
- **bmi**: Body Mass Index
- **children**: Number of children/dependents
- **smoker**: Smoking status (`yes`, `no`)
- **region**: Residential area (`northeast`, `northwest`, `southeast`, `southwest`)
- **charges**: Medical insurance cost

The dataset is included in the repository as `insurance.csv`.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/GustaDNS/Med_Insurance.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Med_Insurance
    ```
3. Create a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the analysis and modeling scripts, simply execute:

```bash
python main.py
