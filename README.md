# LLM Training Stability Simulator

## Overview
This project simulates a simplified LLM training pipeline focusing on:

- Early stopping
- Divergence detection
- Evaluation robustness
- API failure handling (inspired by deprecated endpoints)

## Motivation
During the Hugging Face LLM Course, an issue with a deprecated API endpoint (410 error)
highlighted the importance of resilient ML systems.

This project demonstrates how to:
- Detect training instability
- Stop training early
- Handle failing inference APIs

## Features

### Early Stopping
Prevents overfitting and wasted compute.
⛔ Early stopping triggered at step 49

### Divergence Detection
Flags unstable training runs.

### API Fallback System
Handles deprecated endpoints gracefully.

## Run

```bash
python simulation.py
