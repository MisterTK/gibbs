#!/bin/bash
# Wait (max 3 h) until non-gibbs GPU load stays under 15% for 5 consecutive 10 s samples, then run loop_verify.sh.
cd "$(dirname "$0")/.."
ok=0; t0=$(date +%s)
while [ $ok -lt 5 ]; do
  u=$(ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | grep -o '[0-9]*$')
  if [ "${u:-100}" -lt 15 ]; then ok=$((ok+1)); else ok=0; fi
  [ $(( $(date +%s) - t0 )) -gt 10800 ] && { echo "never quiet" >&2; exit 2; }
  sleep 10
done
exec bash scripts/loop_verify.sh
