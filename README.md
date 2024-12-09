# Dog Breed Classifier deployed to "production"

This project focuses on deploying a ML model to classify dog breeds using Amazon SageMaker, pytTorch model in EC2 instance. The model is trained on a dataset of labeled dog images, where each label corresponds to a specific breed. 

# Project Set Up
A local development environment was configured using a Docker container with Amazon SageMaker client libraries, leveraging an official image downloaded from AWS. This approach enabled seamless development and execution of code within a Jupyter Notebook environment, ensuring compatibility with SageMaker's APIs and frameworks.

Using a local container allowed for efficient resource utilization by minimizing reliance on cloud-based instances during the initial stages of development. This setup also provided the flexibility to iterate quickly on the code and test functionality locally before deploying models to SageMaker. As a result, more time was spent refining the project rather than managing cloud resources.

Due to limited access to the existing AWS account(issue reported to Udacity) at the moment of developing, EC2 instance was successfully replaced by local Docker container. Please note that 

# SageMaker Traning / Deployment
- Hyperparamter tuning
instance_type="ml.g4dn.xlarge",


- Hyperparamter tuning Creating an estimator

# EC2 Training

version 

Managing computing resources efficiently
Training models with large datasets using multi-instance training
Setting up high-throughput, low-latency pipelines
AWS security