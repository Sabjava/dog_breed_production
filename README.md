# Dog Breed Classifier deployed to "production"

This project focuses on deploying a ML pyTorch model to classify dog breeds using Amazon SageMaker and directly in EC2 instance. The model was trained on a dataset of labeled dog images, where each label corresponds to a specific breed. 

# Project Set Up
A local development environment was configured using a Docker container with Amazon SageMaker client libraries, leveraging an official image downloaded from AWS. This approach enabled seamless development and execution of code within a Jupyter Notebook environment, ensuring compatibility with SageMaker's APIs and frameworks.

Using a local container allowed for efficient resource utilization by minimizing reliance on cloud-based instances during the initial stages of development. This setup also provided the flexibility to iterate quickly on the code and test functionality locally before deploying models to SageMaker. As a result, more time was spent refining the project rather than managing cloud resources.

Due to limited access to the existing AWS account at the moment of writing the report, EC2 instance was successfully replaced by local Docker container. The issue is reported to Udacity.

# Dataset
Training / Validateion / Test dataset contained multiple images of 133 breeds and was downloaded from AWS by
```
wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
```
It was downloded to local machine and later uploaded using command line aws and stored in S3
```
aws s3 sync
```
<img width="1110" alt="Screen Shot 2024-12-07 at 22 34 32" src="https://github.com/user-attachments/assets/67278ad8-5b5e-4dc4-814f-b2ef0f9ac136">




# SageMaker Traning / Deployment
## Hyperparameter tuning.
- The list of Hyperparameter tuning jobs that we run during SageMaker phase obtained by `aws cli`:
  
<img width="1280" alt="Screen Shot 2024-12-09 at 09 48 27" src="https://github.com/user-attachments/assets/f4bc2efa-a1f0-4126-b6a3-e5e00cd096ac">


## Creating an estimator. The latest training job that was later deployed
<img width="1127" alt="SM_training_job" src="https://github.com/user-attachments/assets/f3c010ea-f3a4-4f61-847d-89ce0275c6de">



## Deploying endpoint. Screenshot of deployed endpoint

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
- Use `ml.m5.xlarge` for general-purpose CPU-bound tasks - most heavy lifting was done at the tuning phase on ml.g4dn.xlarge, so it made sense to use cheaper instance for training (ml.m5.xlarge )


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
Based on the analysis it would make sense to consider ec2 trained model to be POC rather than optimized. I decided to use the least expensive approach and to train it in my local environment in a docker container.
Docker container used the following image:
`public.ecr.aws/sagemaker/sagemaker-distribution   1.11-gpu   5c13a56bf735   2 months ago   15.2GB` 

Execution took around 20 min and resulted in model being saved:

<img width="1118" alt="Screen Shot 2024-12-09 at 14 43 13" src="https://github.com/user-attachments/assets/9366f919-03df-4265-a4d6-731669f18ba8">

# Using Lambda to make predictions
Lambda function serves as a bridge between a client request and the SageMaker endpoint `pytorch-inference-2024-12-09-02-08-34-307`. It expects a JSON payload from the client containing the key url, which specifies the image URL to be sent for prediction. The function uses the boto3 library to invoke the SageMaker endpoint with the provided input, sending the data in JSON format. The prediction results are read from the endpoint’s response, parsed from JSON, and returned to the client along with appropriate HTTP headers. This design enables seamless integration with SageMaker for real-time inference on image data.

<img width="1280" alt="Screen Shot 2024-12-09 at 17 41 05" src="https://github.com/user-attachments/assets/8e66e41a-8ce8-4ec2-96b4-ec578498739d">

# Security
To maintain a secure workspace, I followed best practices such as granting the minimum necessary permissions (principle of least privilege), restricting access to only the required services and resources through IAM Roles and monitoring lambda logs through  AWS CloudTrail and CloudWatch. By leveraging these features effectively, AWS workspaces are both secure and resilient to potential threats.

# Scalability and Concurrency 
This ML project leverages AWS SageMaker for model deployment and Lambda for prediction requests, using auto-scaling to efficiently handle varying traffic loads. SageMaker instances scale automatically based on demand, ensuring resources are optimized and costs are minimized by adjusting the number of instances based on CPU or memory usage. Lambda handles high concurrency, processing individual prediction requests, with costs based on the number of invocations and execution duration. By optimizing inference models, minimizing Lambda execution time, and using auto-scaling policies, the system maintains high availability and cost-efficiency while serving machine learning predictions at scale.
Here is adjustment in estimater to allow scalability
```
estimator = PyTorch(
    entry_point='hpo.py',
    base_job_name='dog-pytorch',
    role=role,
    instance_count=2,
    instance_type='ml.m5.xlarge',
    framework_version='1.4.0',
    py_version='py3',
    hyperparameters=hyperparameters,
    ## Debugger and Profiler parameters
    rules = rules,
    debugger_hook_config=hook_config,
    profiler_config=profiler_config,
)
```
There are several limits we can use in lambda configuration:
```
aws lambda put-function-concurrency \
    --function-name dog_breed_predictior \
    --reserved-concurrent-executions 100
```
This will ensure that no more than 100 instances of the Lambda function will run concurrently.

# Acknowledgement

I thank all the teachers who helped me by providing interesting assignments and learning materials.  Kudos to chatGPT (and all developers whose code is used by the tool), I  can imagine how much harder this project was before. 

![French_bulldog_04821](https://github.com/user-attachments/assets/5efc76a5-7b61-43d9-95b2-16f48136384a)



