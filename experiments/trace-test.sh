  #!/bin/bash
  cd /Users/kevin/code/flash-moe/metal_infer

  echo "=== BASELINE (cold) ==="
  sudo purge
  ./infer --model ../models/Qwen3.5-35B-A3B-UD-Q4_K_XL-mlx --prompt "Explain how hash tables work" --tokens 300 --k 8 --trace-timing 2>../experiments/trace.tsv 

