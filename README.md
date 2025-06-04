# Simple Breast Cancer Classification Neural Network
A comprehensive, from-scratch implementation of a feedforward neural network for breast cancer diagnosis. This project demonstrates systematic hyperparameter evaluation and optimization by providing deep insights into the network’s design principles.

## Project Overview
In this project, a multi-layer perceptron (MLP) is built from scratch using only NumPy, for the purpose of classifying breast cancer tumors as benign or malignant. Through rigorous experimentation across 16 different configurations, the model achieves a highly satisfactory test accuracy of 98.25% along with excellent generalization properties.

## Dataset Overview
This project is built upon the Wisconsin Breast Cancer Dataset that is best described by the following:
- **Total Samples**: 569 patient records
- **Feature Dimension**: 30 numerical features
- **Class Distribution**:
  - *Malignant (M)*: 212 cases (37.3%)
  - *Benign (B)*: 357 cases (62.7%)
  - *Imbalance Ratio*: ~1.68:1 (moderate imbalance) – model might be biased toward predicting Benign (majority class), recall/specificity will be later tested to check model’s immunity to imbalance
- **Feature Categories**:
  - *Mean Values (10 features)*: radius_mean, texture_mean, perimeter_mean, etc.
  - *Standard Error (10 features)*: radius_se, texture_se, perimeter_se, etc.
  - *Worst Values (10 features)*: radius_worst, texture_worst, perimeter_worst, etc.
 
## Data Preprocessing
- Normalization: min-max scaling to [0,1] range to ensure feature equality
- Target Encoding: Malignant (M) -> 1/Benign (B) -> 0
- Train-Test Split: 80 – 20 stratified split

## Neural Network Implementation
### Neural Network Architecture
Input Layer (30 neurons for 30 input features)  
              |  
Hidden Layer(s) (size and configuration to be tested for optimization, Sigmoid)  
              |  
Output Layer (1 neuron, Sigmoid)

### Learned Patterns (Cancer Signatures) 
**1.	Input Layer – Raw Medical Measurements**
-	radius_mean, texture_mean, perimeter_mean, area_mean, etc.

**2.	First (or Only) Hidden Layer(s) – Combinations of Raw Features into Meaningful Patterns**
-	Example Patterns: “Large size + irregular texture”, “High compactness + Concave points”
-	Each neuron acts as a feature detector for specific B/M signatures
-	Number of patterns learned = number of neurons

**3.	Additional Hidden Layers – Higher-Level Abstractions**
-	Complex representations from previous layer’s patterns
-	May learn meta-patterns (e.g., “multiple malignant indicators present”)

### Why Sigmoid Activation?
- **Output Layer**
  - Maps to probability [0,1]
  - Enables stable backpropagartion
  - Direct interpretable output of malignancy

- **Hidden Layer(s)**
  - Enables learning of non-linear, complex feature interactions
  - Prevents explosive activations within a bounded output
  - Stable gradient computation

## Hyperparameter Optimization 
### Experimentation Methodology
Each parameter will be tested at each phase across many iterations and its best value updated for the later phases. Hyperparameter optimization evaluation and final decision can be found in the `hyperparameter_evaluation_results.png` file.

### Results Analysis
The output for the model architecture and parameters adopted as result of the experimentation methodology presented above were as follows:


#### Accuracy Metrics
| Metric | Value |
|--------|-------|
| Training Accuracy | 96.26% |
| Test Accuracy | 98.25% |
| Overfitting Score | -1.98% |

#### Confusion Matrix - Test Set
|           | Predicted Malignant | Predicted Benign |
|-----------|-------------------|------------------|
| **Actual Malignant** | 43 | 0 |
| **Actual Benign** | 2 | 69 |

#### Medical Performance Metrics
| Metric | Value | Clinical Significance |
|--------|-------|---------------------|
| **Sensitivity (Recall)** | 100.0% | Perfect cancer detection |
| **Specificity** | 97.2% | Excellent benign identification |
| **Precision (PPV)** | 95.6% | High positive prediction accuracy |
| **Negative Predictive Value** | 100.0% | Perfect negative prediction |
| **Balanced Accuracy** | 98.6% | Outstanding overall performance |

#### Why This Performance is Exceptional
- **Negative Overfitting**: Test accuracy>Train accuracy indicates excellent generalization
- **Medical Relevance**: 98.25% Training accuracy is clinically exceptional
- **Balanced Performance**: High precision and recall (dataset imbalance did not severely affect results)
- **Robustness**: Consistent performance across multiple runs, even with seeds changed

## Usage
To train and evaluate the model:
1.	Clone the repository
2.	Install dependencies with `pip install -r requirements.txt`
3.	Run `python neural_network.py` 

## Future Improvements
This project aims at highlighting important considerations in Neural Network architecture design, hyperparameter tuning and overfitting detection. Further improvements to performance for large-scale implementation instances would include the following:
-	Regularization
-	Cross-Validation
-	Feature Engineering
-	Ensemble Methods
