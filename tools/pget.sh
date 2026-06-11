#!/usr/bin/env bash
# Parallel-range download through a per-connection-throttled proxy.
# Usage: pget.sh <url> <output> [n_parts=16]
set -u
URL=$1; OUT=$2; N=${3:-16}
# take the largest size advertised across redirect hops (HEAD sometimes
# returns an error/redirect page; x-linked-size is HF's authoritative value)
SIZE=$(for i in 1 2 3; do
  curl -sLI "$URL" | grep -iE '^(content-length|x-linked-size):' | tr -dc '0-9\n'
  sleep 2
done | sort -n | tail -1)
[ -z "$SIZE" ] || [ "$SIZE" -lt 1000000 ] && { echo "no plausible size (got '$SIZE')"; exit 1; }
echo "size: $SIZE bytes, $N parts"
CHUNK=$(( (SIZE + N - 1) / N ))
mkdir -p "$OUT.parts"
pids=()
for i in $(seq 0 $((N-1))); do
  (
    s=$((i*CHUNK)); e=$(( (i+1)*CHUNK - 1 )); [ $e -ge $SIZE ] && e=$((SIZE-1))
    part="$OUT.parts/$i"
    want=$((e - s + 1))
    for try in $(seq 1 200); do
      have=0; [ -f "$part" ] && have=$(stat -c%s "$part")
      [ "$have" -ge "$want" ] && break
      curl -sL --range $((s+have))-$e -o - "$URL" >> "$part" || sleep 3
    done
  ) &
  pids+=($!)
done
wait "${pids[@]}"
total=0
for i in $(seq 0 $((N-1))); do total=$((total + $(stat -c%s "$OUT.parts/$i"))); done
if [ "$total" -ne "$SIZE" ]; then echo "INCOMPLETE: $total/$SIZE"; exit 1; fi
cat $(for i in $(seq 0 $((N-1))); do echo "$OUT.parts/$i"; done) > "$OUT"
rm -rf "$OUT.parts"
echo "DONE: $OUT ($(stat -c%s "$OUT") bytes)"
