# 0. PGD Attack Success Rate (ASR) Metric 
# 1. Metric Name
Attack Success Rate

# 2. Trustworthiness aspect
Category : Security / Robustness\
Description :
The Attack Success Rate (ASR) metric measures how easily an AI model's predictions can be changed by adversarial perturbations. It evalutes the orbustness of AI models by quantifying the proportion of successful attack that cause misclassification.

In other words, a higher ASR indicates that the model is more vulnerable to adversarial attacks; while a lower ASR indicates greater robustness.

# 3. Class of models it applies to
  - Primarly designed for neural network classifiers (PyTorch-based in this project)
  - Can generalize to models exposing:
      - ```.predict()``` : deterministric class predictions
      - ```.predict.proba()``` : predicted class probabilities
      - ```.predict_with_grad()``` : optionally provides gradients for attack calculation

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
The **Tiny-ImageNet** dataset is provided as a ZIP file (`tiny-imagenet-200.zip`) to avoid having thousands of files in the repo. Before you can use the metrics or run the tests, you must
**unzip the dataset**.
### Instructions
1. Check that the `tiny-imagenet-200.zip` file is present at the root of the repository.
2. Unzip it into the same folder as the repository:
```unzip tiny-imagenet-200.zip```

To launch the test suite, simply execute this command: ```uv run pytest```

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


