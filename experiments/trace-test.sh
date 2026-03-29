  #!/bin/bash
  cd /Users/kevin/code/flash-moe/metal_infer

  echo "=== BASELINE (cold) ==="
  sudo purge
  ./infer --model ../models/Qwen3.5-35B-A3B-4bit --prompt "Explain how hash tables work" --tokens 100 --k 8 --trace-timing 2>../experiments/trace.tsv 

