  #!/bin/bash
  cd /Users/kevin/code/flash-moe/metal_infer

  echo "=== BASELINE (cold) ==="
  sudo purge
  ./infer --model ../models/Qwen3-Coder-Next-4bit --prompt "Explain how hash tables work" --tokens 100 --k 10 --timing 2>&1 #| grep -E "cmd[123]|cpu_attn|expert_io|total_layer|sum_phases|Generation"
  sudo purge
  ./infer --model ../models/Qwen3.5-35B-A3B-4bit --prompt "Explain how hash tables work" --tokens 100 --k 8 --timing 2>&1 #| grep -E "cmd[123]|cpu_attn|expert_io|total_layer|sum_phases|Generation"

  #echo ""
  #echo "=== MLOCK 8GB (cold) ==="
  #sudo purge
  #./infer --model ../models/Qwen3-Coder-Next-4bit --prompt "Explain how hash tables work" --tokens 100 --k 10 --timing --mlock-cache 12 2>&1 #| grep -E "mlock|cmd[123]|cpu_attn|expert_io|total_layer|sum_phases|Generation"
