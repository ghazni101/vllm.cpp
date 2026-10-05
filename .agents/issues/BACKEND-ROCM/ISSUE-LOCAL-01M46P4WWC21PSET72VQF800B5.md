ID: ISSUE-LOCAL-01M46P4WWC21PSET72VQF800B5
Title: rocm exl3: half-integer bit rates (K+0.5, mul1) for OrcaSAQ-class mixed-rate checkpoints
Row: BACKEND-ROCM
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-05
Updated: 2026-10-05
Closed: -

## Problem

Checkpoints whose trellis tensors have last-dim = 16*K+8 (K+0.5 bits per weight, mul1 codebook, e.g. orcarouter/OrcaSAQ-2-27B-EXL3-3.21bpw with 120 tensors at 56 words) fail LoadExl3's words%16==0 gate and have no kernel arm to decode them even if loaded. exllamav3 >= git-679835b7 defines the format: alternating KA/KA+1-bit positions (mask 0xAAAA, period 16), dq8_half decode.

## Resolution

-
