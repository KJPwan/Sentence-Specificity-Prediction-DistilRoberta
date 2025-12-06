#!/bin/bash

# Script to run Experiment A: Data Efficiency Curve

# --- Hyperparameters ---
BATCH_SIZE=8
LEARNING_RATE=2e-5
FROZEN_LAYERS=4
MIX_ALPHA=0.2
EPOCHS=3 # Using 3 epochs for faster iteration

# --- Experiment Setup ---
PERCENTAGES=(1 2_5 10 25 100)
RESULTS_FILE="experiment_A_results.csv"

# --- Preamble ---
echo "Starting Experiment A: Data Efficiency Curve"
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
  echo "Running Baseline model for ${PERCENT_LABEL}% data..."
  
  # Execute and capture output
  BASELINE_OUTPUT=$(TF_CPP_MIN_LOG_LEVEL=2 python train_binary.py \
    --train_path "$TRAIN_FILE" \
    --valid_path "valid_PBSDS.tsv" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --num_frozen_layers "$FROZEN_LAYERS")
  
  # Parse the best F1 score
  BASELINE_F1=$(echo "$BASELINE_OUTPUT" | grep "Training complete! Best F1:" | awk '{print $5}')
  
  # Save result
  if [ -n "$BASELINE_F1" ]; then
    echo "${PERCENT_LABEL},Baseline,${BASELINE_F1}" >> $RESULTS_FILE
    echo "Baseline Best F1 for ${PERCENT_LABEL}%: ${BASELINE_F1}"
  else
    echo "ERROR: Could not parse F1 score for Baseline at ${PERCENT_LABEL}%."
  fi
  
  echo "--------------------------------"

  # --- 2. MixText Run ---
  echo "Running MixText model for ${PERCENT_LABEL}% data..."
  
  # Execute and capture output
  MIXTEXT_OUTPUT=$(TF_CPP_MIN_LOG_LEVEL=2 python train_binary.py \
    --train_path "$TRAIN_FILE" \
    --valid_path "valid_PBSDS.tsv" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --num_frozen_layers "$FROZEN_LAYERS" \
    --use_mixtext \
    --mix_alpha "$MIX_ALPHA")
    
  # Parse the best F1 score
  MIXTEXT_F1=$(echo "$MIXTEXT_OUTPUT" | grep "Training complete! Best F1:" | awk '{print $5}')
  
  # Save result
  if [ -n "$MIXTEXT_F1" ]; then
    echo "${PERCENT_LABEL},MixText,${MIXTEXT_F1}" >> $RESULTS_FILE
    echo "MixText Best F1 for ${PERCENT_LABEL}%: ${MIXTEXT_F1}"
  else
    echo "ERROR: Could not parse F1 score for MixText at ${PERCENT_LABEL}%."
  fi

  echo "================================================="

done

echo "Experiment A complete."
echo "Results saved to ${RESULTS_FILE}"
