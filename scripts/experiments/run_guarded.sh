#!/bin/bash
# Run a command in its own process group; kill the group if MemAvailable drops below a floor.
# Usage: run_guarded.sh LOG FLOOR_GB cmd args...
LOG="$1"
FLOOR=$(( $2 * 1024 * 1024 ))
shift 2
setsid "$@" >>"$LOG" 2>&1 &
PID=$!
# poll free memory until the command exits
while kill -0 "$PID" 2>/dev/null; do
  AVAIL=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ "$AVAIL" -lt "$FLOOR" ]; then
    kill -TERM -- -"$PID" 2>/dev/null
    sleep 3
    kill -KILL -- -"$PID" 2>/dev/null
    echo "ABORTED: MemAvailable $((AVAIL / 1024)) MB below floor ($*)" >>"$LOG"
    exit 137
  fi
  sleep 3
done
wait "$PID"
