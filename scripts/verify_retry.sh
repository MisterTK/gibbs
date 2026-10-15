#!/bin/bash
# verify_when_quiet.sh until it yields a verdict: a contention abort (exit 4: another job took the GPU
# mid-run) is retried after 2 min; any other failure ends the loop. Gives up after $VERIFY_RETRIES
# aborts (default 20). Prints the delta_pp of the accepted run.
cd "$(dirname "$0")/.."
for ((i = 1; i <= ${VERIFY_RETRIES:-20}; i++)); do
  scripts/verify_when_quiet.sh; rc=$?
  [ $rc -ne 4 ] && exit $rc
  echo "verify_retry: contention-aborted (attempt $i), retrying in 120 s" >&2
  sleep 120
done
exit 4
