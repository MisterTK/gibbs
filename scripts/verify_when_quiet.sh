#!/bin/bash
# Wait until non-gibbs GPU load stays <= ${QUIET_MAX_PCT:-15}% for 5 consecutive 10 s samples, then run loop_verify.sh.
# After 30 min of waiting a warning is logged and the poll interval backs off exponentially (10 s
# doubling up to 5 min); gives up (exit 2) after 3 h. loop_verify.sh itself aborts with exit 4
# (contention-aborted) when a run shows base_pct < 35 or e2b_tok_s < 100: nothing is recorded then.
cd "$(dirname "$0")/.."
max=${QUIET_MAX_PCT:-15}  # raise only when the idle floor itself is higher (compositor): then the tripwires carry validity
# The tuner uses the same busy threshold: an idle floor above 20% must not make every verify process
# retune from scratch and never persist.
export GIBBS_TUNE_BUSY_PCT=${GIBBS_TUNE_BUSY_PCT:-$max}
ok=0; t0=$(date +%s); sleep_s=10; warned=0
while [ $ok -lt 5 ]; do
  u=$(ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | grep -o '[0-9]*$')
  if [ "${u:-100}" -le "$max" ]; then
    ok=$((ok+1)); sleep_s=10
  else
    ok=0
    echo "quiet-wait: GPU busy (${u:-?}%), waiting..." >&2
  fi
  waited=$(( $(date +%s) - t0 ))
  [ $waited -gt 10800 ] && { echo "never quiet" >&2; exit 2; }
  if [ $waited -gt 1800 ]; then
    [ $warned -eq 0 ] && { echo "quiet-wait: warning, GPU not quiet for 30 min; backing off" >&2; warned=1; }
    [ $ok -eq 0 ] && sleep_s=$(( sleep_s * 2 > 300 ? 300 : sleep_s * 2 ))
  fi
  sleep $sleep_s
done
exec bash scripts/loop_verify.sh
