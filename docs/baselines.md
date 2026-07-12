# SupportAI Baseline Report

This document reports on the dataset characteristics, classical ML baseline implementations, comparative evaluation metrics, systematic error modes, and the selection of our production baseline.

---

## 1. Dataset Summary

SupportAI implements routing for the **Banking77** intent classification dataset. 
- **Unique Classes**: 77 distinct banking intents (e.g., `card_lost`, `transfer_not_received_by_recipient`, `billing`).
- **Data Splits**:
  - **Train**: 10,458 samples
  - **Validation**: 1,309 samples
  - **Test**: 1,302 samples
- **Token Statistics (Train)**:
  - Average Sentence Length: **11.76 words**
  - Minimum Sentence Length: **2 words**
  - Maximum Sentence Length: **79 words**
- **Imbalance**: Relatively balanced dataset, with most classes containing 100–150 samples in the combined set, except for a few rare classes.

---

## 2. Classical Models Compared

Three classical TF-IDF machine learning pipelines were trained on the preprocessed training split and evaluated on the test split:
1. **TF-IDF + Logistic Regression**: L2-regularized multinomial logistic regression.
2. **TF-IDF + Linear SVM**: Linear Support Vector Classifier (`LinearSVC`).
3. **TF-IDF + Multinomial Naive Bayes**: Generative probabilistic classifier.

Central parameters (default configuration):
- Tokenizer: Word-level normalization, case folded, whitespace stripped.
- TF-IDF Max Features: 5,000.
- N-gram Range: (1, 2) (unigrams and bigrams).

---

## 3. Metrics Table

Below is the performance comparison compiled on the test set:

| Model | Accuracy | F1 (Weighted) | Precision (Weighted) | Recall (Weighted) | Training Time (s) | Latency (ms/sample) | Model Size (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 87.79% | 87.71% | 88.51% | 87.79% | 5.64s | ~0.04 ms | 3.28 MB |
| **Linear SVM** | **90.71%** | **90.70%** | **91.10%** | **90.71%** | 4.19s | ~0.04 ms | 3.28 MB |
| **Multinomial Naive Bayes** | 84.72% | 84.32% | 85.62% | 84.72% | 0.69s | ~0.04 ms | 6.36 MB |

---

## 4. Confusion Matrix and Error Analysis

Systematic error analysis was executed on the best performing model (**Linear SVM**):
- **Total Errors**: 121 errors out of 1,302 samples (Error Rate: **9.29%**).
- **Most Confused Class Pairs**:
  1. `transfer_not_received_by_recipient` mistaken for `balance_not_updated_after_bank_transfer` (4 counts)
  2. `balance_not_updated_after_bank_transfer` mistaken for `pending_transfer` (3 counts)
  3. `balance_not_updated_after_bank_transfer` mistaken for `transfer_not_received_by_recipient` (2 counts)
- **Top Hardest Class Intents**:
  - `pending_transfer` (accuracy: ~58%)
  - `transfer_not_received_by_recipient` (accuracy: ~67%)

The confusion matrix indicates distinct clusters of error density among overlapping semantic concepts (specifically bank transfers, payment settlement timing, and balance reconciliation).

---

## 5. Failure Cases

### Longest Failure Cases (Semantic Overload)
- **Query**: *"I would like to make sure that a bank transfer I sent earlier today has been received. I did a bank transfer to an account earlier today. How can I check to make sure it was received? What details do I need to look for to verify it?"*
  - **True Label**: `transfer_not_received_by_recipient`
  - **Predicted**: `balance_not_updated_after_bank_transfer`
  - **Diagnostic**: The text contains multiple overlapping entities ("bank transfer", "received", "verify"). Because TF-IDF measures keyword frequencies rather than temporal sequence or syntactic dependency, it misallocates attention to balance status keyword associations.

### Shortest Failure Cases (Sparse Context)
- **Query**: *"change address"*
  - **True Label**: `change_personal_details`
  - **Predicted**: `card_delivery_estimate`
  - **Diagnostic**: Extreme brevity provides insufficient bigram vocabulary tokens, forcing the model to guess based on sparse prior distributions.

---

## 6. Why SVM Was Chosen as Baseline

**Linear SVM** was selected as the final production baseline for the classical layer due to:
1. **Superior Accuracy**: Reached **90.71%** accuracy, outperforming Logistic Regression by ~3% and Naive Bayes by ~6%.
2. **Speed & Efficiency**: Reached a low prediction latency of **0.04 ms/sample** on single-core CPU hardware, with a minimal model footprint of **3.28 MB**.
3. **Robustness in High-Dimensional Space**: SVMs are mathematically designed to find the maximum margin hyperplane, making them highly effective when text inputs are represented as sparse high-dimensional TF-IDF vectors.

---

## 7. Limitations of TF-IDF Baselines

1. **Bag-of-Words Limitation**: TF-IDF discards word order and context. Phrases like *"not received"* and *"received, not"* are treated identically.
2. **Out-of-Vocabulary (OOV)**: If a customer uses synonyms or words not present in the training vocabulary, the classical models fail to capture the semantic relationship.
3. **No Semantic Compositionality**: Words like *"fee"*, *"charge"*, and *"toll"* are treated as completely independent dimensions, ignoring their conceptual similarities.

---

## 8. Hypothesis for why DistilBERT should improve performance

We hypothesize that a fine-tuned **DistilBERT** (a compact transformer model) will resolve these limitations:
1. **Contextual Token Embeddings**: DistilBERT uses self-attention mechanisms to generate embeddings that depend on the word's context, resolving syntax nuances and negative modifiers.
2. **Deep Semantic Understanding**: Through pre-training on large text corpora, DistilBERT maps synonyms and semantically related words (e.g., *"cost"*, *"fee"*, *"amount"*) to nearby points in the vector space, handling OOV keywords and semantic variations.
3. **Contextual Sequence Alignment**: By modeling sequence positioning, DistilBERT can distinguish subtle timing intents, such as separating a transfer that is currently "pending" vs. one that has "failed to arrive".
