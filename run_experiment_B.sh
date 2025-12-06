#!/bin/bash

# Script to run Experiment B: Data Efficiency Curve with MixText Alpha = 0.8

# --- Hyperparameters ---
BATCH_SIZE=8
LEARNING_RATE=2e-5
FROZEN_LAYERS=4
MIX_ALPHA=0.8 # Using a stronger alpha for this experiment
EPOCHS=10 # Using 10 epochs for a more robust result per run

# --- Experiment Setup ---
PERCENTAGES=(1 2_5 10 25 100)
RESULTS_FILE="experiment_B_results.csv"

# --- Preamble ---
echo "Starting Experiment B: Data Efficiency Curve (MixText Alpha = 0.8)"
echo "This will run 10 training jobs. This may take a significant amount of time."
echo "-----------------------------------------------------------------"
echo "Hyperparameters:"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Frozen Layers: $FROZEN_LAYERS"
echo "MixText Alpha: $MIX_ALPHA"
echo "Epochs per run: $EPOCHS"
echo "-----------------------------------------------------------------"

# --- Initialize Results File ---
echo "data_percent,model_type,best_f1" > $RESULTS_FILE

# --- Main Loop ---
for PERCENT in "${PERCENTAGES[@]}"; do
  
  # Determine training file
  if [ "$PERCENT" -eq 100 ]; then
    TRAIN_FILE="train_PBSDS.tsv"
    PERCENT_LABEL="100"
  elif [ "$PERCENT" == "2_5" ]; then
    TRAIN_FILE="train_pbsds_2_5.tsv"
    PERCENT_LABEL="2.5"
  else
    TRAIN_FILE="train_pbsds_${PERCENT}.tsv"
    PERCENT_LABEL="$PERCENT"
  fi
  
  echo "--- Processing ${PERCENT_LABEL}% of data (File: ${TRAIN_FILE}) ---"

  # --- 1. Baseline Run (No MixText) ---
  # Note: Baseline is the same as in Experiment A, but we run it again
  # here to keep the script self-contained and results consistent.
  echo "Running Baseline model for ${PERCENT_LABEL}% data..."
  
  BASELINE_OUTPUT=$(TF_CPP_MIN_LOG_LEVEL=2 python train_binary.py \
    --train_path "$TRAIN_FILE" \
    --valid_path "valid_PBSDS.tsv" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --num_frozen_layers "$FROZEN_LAYERS")
  
  BASELINE_F1=$(echo "$BASELINE_OUTPUT" | grep "Training complete! Best F1:" | awk '{print $5}')
  
  if [ -n "$BASELINE_F1" ]; then
    echo "${PERCENT_LABEL},Baseline,${BASELINE_F1}" >> $RESULTS_FILE
    echo "Baseline Best F1 for ${PERCENT_LABEL}%: ${BASELINE_F1}"
  else
    echo "ERROR: Could not parse F1 score for Baseline at ${PERCENT_LABEL}%."
  fi
  
  echo "--------------------------------"

  # --- 2. MixText Run (Alpha = 0.8) ---
  echo "Running MixText (Alpha=0.8) model for ${PERCENT_LABEL}% data..."
  
  MIXTEXT_OUTPUT=$(TF_CPP_MIN_LOG_LEVEL=2 python train_binary.py \
    --train_path "$TRAIN_FILE" \
    --valid_path "valid_PBSDS.tsv" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --num_frozen_layers "$FROZEN_LAYERS" \
    --use_mixtext \
    --mix_alpha "$MIX_ALPHA")
    
  MIXTEXT_F1=$(echo "$MIXTEXT_OUTPUT" | grep "Training complete! Best F1:" | awk '{print $5}')
  
  if [ -n "$MIXTEXT_F1" ]; then
    echo "${PERCENT_LABEL},MixText_Alpha_0.8,${MIXTEXT_F1}" >> $RESULTS_FILE
    echo "MixText (Alpha=0.8) Best F1 for ${PERCENT_LABEL}%: ${MIXTEXT_F1}"
  else
    echo "ERROR: Could not parse F1 score for MixText (Alpha=0.8) at ${PERCENT_LABEL}%."
  fi

  echo "================================================="

done

echo "Experiment B complete."
echo "Results saved to ${RESULTS_FILE}"
