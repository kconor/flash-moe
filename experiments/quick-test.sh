  #!/bin/bash

  #sudo purge
  #echo "\n\n=== CODER NEXT ==="
  #cmd="../metal_infer/infer --model ../models/Qwen3-Coder-Next-4bit --prompt 'Explain how hash tables work' --tokens 100 --k 10 --timing 2>&1"
  #eval $cmd

  sudo purge
  echo "\n\n=== 35 A3B - uniform quant ==="
  cmd="../metal_infer/infer --model ../models/Qwen3.5-35B-A3B-4bit --prompt 'Explain how hash tables work' --tokens 100 --k 10 --timing 2>&1"
  eval $cmd

  sudo purge
  echo "\n\n=== 35 A3B - variable quant ==="
  cmd="../metal_infer/infer --model ../models/Qwen3.5-35B-A3B-UD-Q4_K_XL-mlx --prompt 'Explain how hash tables work' --tokens 100 --k 10 --timing 2>&1"
  eval $cmd
