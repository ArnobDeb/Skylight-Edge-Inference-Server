#!/usr/bin/env bash
# Poll Prajna (SSH ControlMaster must be open) until each job finishes, then copy its run dir, convert every
# checkpoint to .npz and mark the lowest-validation-loss checkpoint (BEST). Args: "<jobid>:<run name>" ...
#   bash experiments/adapter_training/prajna/collect_when_done.sh 349525:ovblock_n16_qwen3-8b ...
set -u
cd "$(dirname "$0")/../../.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate arnob
LOG=results/eviction_study/logs/collect_adapters.log
pending=("$@")
while [ ${#pending[@]} -gt 0 ]; do
  left=()
  for spec in "${pending[@]}"; do
    job=${spec%%:*}; name=${spec##*:}
    st=$(ssh -o BatchMode=yes -o ConnectTimeout=20 prajna "/opt/slurm/bin/sacct -n -X -j $job -o State" 2>/dev/null | grep -v bashrc | head -1 | tr -d ' ')
    case "$st" in
      COMPLETED)
        echo "$(date) $job $name COMPLETED -> collecting" >> $LOG
        rsync -a prajna:restorekv/runs/$name/ artifacts/adapters/$name/ >> $LOG 2>&1
        rsync -a "prajna:restorekv/logs/train_*_$job.out" artifacts/adapters/$name/train_log.out >> $LOG 2>&1
        env -u PYTHONPATH CUDA_VISIBLE_DEVICES= nice python src/restorekv_ov/checkpoint.py artifacts/adapters/$name/checkpoint_step*.pt >> $LOG 2>&1
        python3 benchmarks/best_ckpt.py artifacts/adapters/$name >> $LOG 2>&1
        echo "$(date) $job $name collected" >> $LOG ;;
      FAILED|CANCELLED*|TIMEOUT|OUT_OF_ME*|NODE_FAIL)
        echo "$(date) $job $name $st -- NOT collected" >> $LOG ;;
      *)
        [ -z "$st" ] && echo "$(date) $job: no answer from prajna (ControlMaster expired?)" >> $LOG
        left+=("$spec") ;;
    esac
  done
  pending=("${left[@]+"${left[@]}"}")
  [ ${#pending[@]} -gt 0 ] && sleep 300
done
echo "$(date) all collected" >> $LOG
