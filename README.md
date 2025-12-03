# 0. PGD Attack Success Rate (ASR) Metric 
# 1. Metric Name
Attack Success Rate

Link to the **GitHub repository** for the implementation of my metric: : https://github.com/teofoli-matteo/A4S-TEOFOLI-MATTEO_PGD_ASR

# 2. Trustworthiness aspect
Category : Security / Robustness\
Description :
The Attack Success Rate (ASR) metric measures how easily an AI model's predictions can be changed by adversarial perturbations. It evalutes the orbustness of AI models by quantifying the proportion of successful attack that cause misclassification.

In other words, a higher ASR indicates that the model is more vulnerable to adversarial attacks; while a lower ASR indicates greater robustness.

# 3. Class of models it applies to
  - Primarly designed for neural network classifiers (PyTorch-based in this project)
  - Can generalize to models exposing:
      - ```.predict_class()``` : deterministric class predictions as np.array of shape (N,)
      - ```.predict_proba()``` : predicted class probabilities
      - ```.predict_proba_grad()``` : returns raw logits with gradients enabled, used by the PGD attack to

Data types supported: 
  - Image data (RGB tensors ```[C,H,W]``` : main focus in this project
  
  - Models tested in this Project:
  We evauluted the ASR metric on 4 pretrained ImageNet models

  | Model | Source | Type |
|-----:|-----------|-----------|
|     ResNet-18 | torchvision| CNN|
|     MobileNet-V2| torchvision    |Lightweight CNN|
|     VGG-16| torchvision       |Deep CNN|
|     DenseNet-121| torchvision       |CNN whith skip connections|





# 4. Working assumptions
- The task is a classification problem.
- Input features are numerically perturbable
- Access to model gradients is assumed
- Attack parameters (perturbation size ```eps```, step size ```alpha```, number of iterations ```iters```)

Defaylt attack parameters in my implementation:
  - ```eps = 0.01``` (maximum perturbation per pixel)
  - ```alpha = 0.005``` (step size)
  - ```iters = 7 ``` (number of pgd steps)

# 5. Datasets and Examples
Dataset used in the final evaluation
 - We evaluate ASR on Tiny-ImageNet-200:
    - 200 classes
    - 64x64 images
    - In the project: we sample 500 images for faster evaluation
    - Labels extracted via ```wnids.txt```

How to test: 
## Using the Tiny-ImageNet dataset
In implementing this metric, I use the `tiny-imagenet-200` dataset. To import this dataset into the project, you need to execute this command:
### Instructions
1. ```python3 download_tiny.py```
This script will download and extract the dataset we need to test my metric.
I had to use this script because I couldn't upload my .zip file with the tiny-imagenet-200 folder or the tiny-imagenet-200 zip file because my folder was too large for Moodle.

#### Prerequisite: 
1. Go to the following path: `/a4s/A4S-TEOFOLi-MATTEO`
2. Run the ```uv sync``` command 

To launch the test suite, simply execute this command: 
```uv run pytest``` -> This command will run **all** tests contained in A4S.

```uv run pytest tests/metrics/model_metrics/test_pgd_asr.py``` -> This command will **only** run the **PGD ASR** metric test that I have implemented.

**Another test on a single image** I also implemented a test directly in test_execute.py (here : tests/metrics/model_metrics/test_execute.py), but it only acts on one image (located in the /tests/data/ folder, the image is named labrador.png).

**Note**: In the test I implemented, we set the number of images to 500 in the variable “N_IMAGES” in the file: `test_pgd_asr.py`. You can of course change this value to another (i.e., 10) to have a test that will run quickly.

**Note2** : The measurements I generated with my test_pgd_asr.py are already available in the /tests/data/measures folder so that you can launch the notebook without any problems to view the various results.

## Proof that all tests pass (tested with N_IMAGES = 500) : 
![Tests](tests/data/test_success.png)


# 6. References
- Performance Evaluation of Adversarial Attacks: Discrepancies and Solutions
-> https://arxiv.org/pdf/2104.11103

- Combining Attack Success Rate and Detection Rate for effective Universal Adversarial Attacks
-> https://www.esann.org/sites/default/files/proceedings/2021/ES2021-160.pdf

- How to Estimate the Success Rate of Higher-Order Side-Channel Attacks
-> https://cyber.gouv.fr/sites/default/files/IMG/pdf/How_to_Estimate_the_Success_Rate_of_Higher-Order_-_CHES2014-anssi.pdf

- Understanding Attack Success Rate (ASR) in Adversarial AI
-> https://vtiya.medium.com/understanding-attack-success-rate-asr-in-adversarial-ai-e4a1764c4e49

- Github implementation
-> https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/metrics/metrics.py


