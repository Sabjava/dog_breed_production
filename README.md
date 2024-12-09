# Dog Breed Classifier deployed to "production"

This project focuses on deploying a ML model to classify dog breeds using Amazon SageMaker, pytTorch model in EC2 instance. The model is trained on a dataset of labeled dog images, where each label corresponds to a specific breed. 

# Project Set Up
A local development environment was configured using a Docker container with Amazon SageMaker client libraries, leveraging an official image downloaded from AWS. This approach enabled seamless development and execution of code within a Jupyter Notebook environment, ensuring compatibility with SageMaker's APIs and frameworks.

Using a local container allowed for efficient resource utilization by minimizing reliance on cloud-based instances during the initial stages of development. This setup also provided the flexibility to iterate quickly on the code and test functionality locally before deploying models to SageMaker. As a result, more time was spent refining the project rather than managing cloud resources.

Due to limited access to the existing AWS account(issue reported to Udacity) at the moment of developing, EC2 instance was successfully replaced by local Docker container. Please note that 

# SageMaker Traning / Deployment
- Hyperparamter tuning
  Here is the list of Hyperparameter tuning jobs thare we run during SageMaker phase:
  
instance_type="ml.g4dn.xlarge",
<img width="1280" alt="Screen Shot 2024-12-09 at 09 48 27" src="https://github.com/user-attachments/assets/f4bc2efa-a1f0-4126-b6a3-e5e00cd096ac">


-  Creating an estimator. Here is the latest training job that was later deployed
<img width="1127" alt="SM_training_job" src="https://github.com/user-attachments/assets/f3c010ea-f3a4-4f61-847d-89ce0275c6de">

-  Deplying endpoint
  Here is screenshot of deployed endpoint
<img width="863" alt="SM_endpoint_deployed" src="https://github.com/user-attachments/assets/20d3bc80-7f23-4fd6-be65-20ff4cae45ea">




# EC2 Training

version 

Managing computing resources efficiently
Training models with large datasets using multi-instance training
Setting up high-throughput, low-latency pipelines
AWS security
