# Domain-Agnostic Real-Valued Specificity Prediction (DistilRoBERTa + Self-Ensembling)

This repository contains an updated implementation of real-valued sentence specificity prediction, based on:

**Wei-Jen Ko, Greg Durrett, Junyi Jessy Li. "Domain Agnostic Real-Valued Specificity Prediction." AAAI 2019.**

The original model used GloVe embeddings, a BiLSTM encoder, and a Self-Ensembling (teacher–student) architecture for domain adaptation.
This reimplementation replaces the encoder with DistilRoBERTa and retains the teacher–student Self-Ensembling framework as an integral part of training and evaluation.

---

## Model Overview

The updated model consists of:

### 1. Encoder

- DistilRoBERTa (`distilroberta-base`)
- The [CLS] token embedding is used as the sentence representation.

### 2. Regression Head

A two-layer feedforward network:

```
Linear(768 → 256) → ReLU → Linear(256 → 1)
```

Outputs a real-valued specificity score.

### 3. Self-Ensembling Framework

The training loop includes the full Self-Ensembling mechanism:

**Student model**

- Receives gradients.
- Trained using supervised MSE loss on labeled data.

**Teacher model**

- Same architecture as the student.
- Updated every step via exponential moving average (EMA):

  ```
  θ_teacher = α * θ_teacher + (1 - α) * θ_student
  ```

- Does not receive gradients.
- Used during evaluation.

**Consistency loss**

- Encourages similar predictions for student and teacher:

  ```
  L_u = || f_student(x) - f_teacher(x) ||^2
  ```

**Final training objective**

```
L = L_supervised + λ * L_u
```

This corresponds to the "Self-Ensembling (SE) baseline" in Ko et al. (2019), but with a transformer encoder instead of a BiLSTM.

Note: This repository does not include the unlabeled-domain adaptation or distribution-regularization components from the original paper.

---

## Data Format

Training, validation, and test files must be TSV with two columns:

```
sentence <TAB> score
```

Example:

```
A remarkably well-paced film.    0.82
Decent overall.                  0.31
```

The first line is treated as a header and skipped.

The current model only uses the supervised movie data set. prepare_movie_data.py is the script that converts the original dataset into the correct format.

---

## Training

```
python train.py --train_path train.tsv --valid_path valid.tsv
```

Training automatically:

- creates both student and teacher models,
- applies supervised + consistency losses,
- updates the teacher through EMA,
- evaluates on the validation set each epoch,
- saves the best teacher model (based on validation Pearson) to `best_model.pt`.

---

## Testing

```
python test.py --model_path best_model.pt --test_path test.tsv
```

`test.py`:

- loads the teacher model,
- runs evaluation,
- computes:

  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Pearson correlation
  - Spearman correlation
  - Kendall’s Tau

---

## Evaluation Metrics

The evaluation follows the same metrics used in Ko et al. (2019):

- MSE
- MAE
- Pearson r
- Spearman ρ
- Kendall τ

---

## Citation

If you use this code, please cite:

```
@InProceedings{ko2019domain,
  author    = {Ko, Wei-Jen and Durrett, Greg and Li, Junyi Jessy},
  title     = {Domain Agnostic Real-Valued Specificity Prediction},
  booktitle = {AAAI},
  year      = {2019},
}
```
