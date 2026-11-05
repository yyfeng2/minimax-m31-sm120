#!/bin/bash
docker exec minimax-m31 bash -c '
for p in /proc/[0-9]*; do
  if grep -q msa_nvfp4_sm120 "$p/maps" 2>/dev/null; then
    echo "LOADED $p $(tr -d "\0" < $p/cmdline | cut -c1-60)"
  fi
done
'
