# Dog Breed Classifier deployed to "production"

This project focuses on deploying a ML model to classify dog breeds using Amazon SageMaker, pytTorch model in EC2 instance. The model is trained on a dataset of labeled dog images, where each label corresponds to a specific breed. 

# Project Set Up
A local development environment was configured using a Docker container with Amazon SageMaker client libraries, leveraging an official image downloaded from AWS. This approach enabled seamless development and execution of code within a Jupyter Notebook environment, ensuring compatibility with SageMaker's APIs and frameworks.

Using a local container allowed for efficient resource utilization by minimizing reliance on cloud-based instances during the initial stages of development. This setup also provided the flexibility to iterate quickly on the code and test functionality locally before deploying models to SageMaker. As a result, more time was spent refining the project rather than managing cloud resources.

Due to limited access to the existing AWS account(issue reported to Udacity) at the moment of developing, EC2 instance was successfully replaced by local Docker container. 

# SageMaker Traning / Deployment
- Hyperparamter tuning. The list of Hyperparameter tuning jobs thare we run during SageMaker phase obtained by `aws cli`:
  
<img width="1280" alt="Screen Shot 2024-12-09 at 09 48 27" src="https://github.com/user-attachments/assets/f4bc2efa-a1f0-4126-b6a3-e5e00cd096ac">


-  Creating an estimator. The latest training job that was later deployed
<img width="1127" alt="SM_training_job" src="https://github.com/user-attachments/assets/f3c010ea-f3a4-4f61-847d-89ce0275c6de">


-  Deploying endpoint. Screenshot of deployed endpoint
<img width="863" alt="SM_endpoint_deployed" src="https://github.com/user-attachments/assets/20d3bc80-7f23-4fd6-be65-20ff4cae45ea">


We used 2 different instances for tuning and training with best estimator `ml.g4dn.xlarge` and `ml.m5.xlarge`:

## Instance Comparison: `ml.g4dn.xlarge` vs `ml.m5.xlarge`


| Feature               | ml.g4dn.xlarge                          | ml.m5.xlarge                          |
|-----------------------|-----------------------------------------|----------------------------------------|
| **vCPUs**             | 4                                       | 4                                      |
| **Memory**            | 16 GiB                                  | 16 GiB                                 |
| **GPU**               | 1 NVIDIA T4 Tensor Core GPU with 16 GiB VRAM | None                              |
| **Instance Storage**  | 125 GB NVMe SSD                         | EBS Only                               |
| **Network Performance** | Up to 25 Gbps                         | Up to 10 Gbps                          |
| **Price (On-Demand)** | ~$0.526 per hour                        | ~$0.192 per hour                       |

### Key differences: 
- Use `ml.g4dn.xlarge` for GPU-accelerated tasks like deep learning training or inference.
- Use `ml.m5.xlarge` for general-purpose CPU-bound tasks - most heavy lifting was done at the tuning phase, so it made sense to use cheaper instance for training


# EC2 Training

Before deciding which image to use to run the training job inside EC2 instance, I analyzed the difference in the code
There are several differences. 

1. epoch_loss = running_loss / len(image_dataset[phase]) in ec2train.py
epoch_loss = running_loss // len(image_dataset[phase]) in hpo.py
I would suggest using float valaue  as in ec2train. 

2. Different hyperparameters are used in hpo and ec2 ,

   ## Hyperparameter Comparison

This table shows the differences in hyperparameters and their implications between the `ec2train1.py` and `hpo.py` configurations.

| **Parameter**      | **ec2train1.py**   | **hpo.py**              | **Implications**                                                                                 |
|---------------------|-------------------|-------------------------|--------------------------------------------------------------------------------------------------|
| **Batch Size**      | 2                 | 32                      | Smaller batch size gives detailed gradients; larger batch size leads to faster and smoother training. |
| **Learning Rate**   | 0.0001            | 0.020308291420289847    | Smaller rate slows training for stability; larger rate accelerates but may risk overshooting.   |
| **Epochs**          | 5                 | 50                      | Fewer epochs risk underfitting; more epochs reduce underfitting risk but may overfit.           |

---

### Summary

1. **Batch Size**:
   - `ec2train.py` uses a smaller batch size of 2, suitable for limited memory systems but slower and noisier in training.
   - `hpo.py` employs a batch size of 32, enabling faster and smoother training at the cost of increased memory usage.

2. **Learning Rate**:
   - `ec2train.py` adopts a small learning rate of 0.0001 for stable but slower convergence.
   - `hpo.py` uses a larger learning rate of ~0.0203, which accelerates training but requires careful tuning to avoid instability.

3. **Epochs**:
   - `ec2train.py` limits training to 5 epochs, which may result in underfitting.
   - `hpo.py` trains for 50 epochs, allowing for thorough learning but with a potential risk of overfitting if not monitored.

These differences highlight the resource-optimized nature of `ec2train.py` versus the hyperparameter-optimized approach in `hpo.py`.
Based on the analysis it  would make sense to conside ec2 ttrained model to be POC rather than optimized. I decided to use the least expensive approach and to train it in my local environment in a docker container.
Docker container used the following image:
`public.ecr.aws/sagemaker/sagemaker-distribution   1.11-gpu   5c13a56bf735   2 months ago   15.2GB` 

Execution took around 20 min and resulted in model being saved:

<img width="1118" alt="Screen Shot 2024-12-09 at 14 43 13" src="https://github.com/user-attachments/assets/9366f919-03df-4265-a4d6-731669f18ba8">

# Using Lambda to make predictions
Lambda function serves as a bridge between a client request and the SageMaker endpoint `pytorch-inference-2024-12-09-02-08-34-307`. It expects a JSON payload from the client containing the key url, which specifies the image URL to be sent for prediction. The function uses the boto3 library to invoke the SageMaker endpoint with the provided input, sending the data in JSON format. The prediction results are read from the endpoint’s response, parsed from JSON, and returned to the client along with appropriate HTTP headers. This design enables seamless integration with SageMaker for real-time inference on image data.

<img width="1280" alt="Screen Shot 2024-12-09 at 17 35 07" src="https://github.com/user-attachments/assets/3913f57f-f893-43d6-ba98-6887d498098c">


![Screenshot 2024-11-22 at 11 24 59 AM (2)](https://github.com/user-attachments/assets/5a28b85b-66e9-4e1c-9d4e-254337991e27)

## Security
Amazon Web Services (AWS) provides a highly secure environment, with robust security features such as data encryption, network firewalls, and identity and access management (IAM). However, the security of an AWS workspace largely depends on how it is configured and used. To maintain a secure workspace, it is essential to follow best practices such as granting the minimum necessary permissions (principle of least privilege), restricting access to only the required services and resources, enabling multi-factor authentication (MFA), and monitoring activity through tools like AWS CloudTrail and CloudWatch. By leveraging these features effectively, AWS workspaces can be both secure and resilient to potential threats.

