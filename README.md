# Mosaic AI Model Training Sample Repository

This repository contains a collection of sample scripts and notebooks for training models using Mosaic AI's platform. The primary focus of this repository is to demonstrate how to leverage the Mosaic CLI (MCLI) through its Python SDK to perform various machine learning operations directly within Databricks Notebooks.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Sample Scripts](#sample-scripts)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Mosaic AI Model Training is a powerful platform that provides tools for training and fine-tuning AI models at scale. While the MCLI is typically used to interact with the platform, the MCLI Python SDK offers an alternative that allows you to integrate Mosaic's capabilities directly into your Python scripts or Jupyter Notebooks. This repository provides several examples demonstrating how to use the Python SDK to manage model training processes within Databricks Notebooks.

## Getting Started

### Prerequisites

Before using the contents of this repository, ensure you have the following:

- An active Mosaic AI account.
- Access to a Databricks workspace.
- Serverless Notebook

### Installation

1. **Clone the Repository on your workspace on Databricks:**

   Follow the instruction [here](https://docs.databricks.com/en/repos/git-operations-with-repos.html).

2. **Install the Required Python Packages:**

   On the Databricks Notebook, attach a serverless compute to the notebook and install the MCLI Python SDK as below:

   ```bash
   %pip install --upgrade mosaicml-cli
   ```

## Usage

### Sample Scripts

This repository includes several sample scripts that illustrate how to use the Mosaic AI Python SDK. Each script is designed to be run within a Databricks Notebook and demonstrates different aspects of model training.

- **Sample 1:** Getting Start
  - This is a sample code to run a model training job which is implemented in LLM-Foundry.
- **Sample 2:** Run GPT-Neox on the Mosaic AI Model Training
  - Mosaic AI Model Training recommends using LLM-Foundry or Composer as the framework. However, it is generally possible to run code implemented with other libraries as well. This notebook is a sample for training a GPT-Neox-based model.

## Documentation

For more detailed information on the Mosaic AI Python SDK, refer to the [official documentation](https://docs.mosaicml.com). The documentation includes comprehensive guides, API references, and examples to help you get the most out of the SDK.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to the Mosaic AI and Databricks communities for their continuous support and contributions. This repository is inspired by the collective efforts of these communities to advance AI model training and deployment at scale.


