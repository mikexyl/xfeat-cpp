#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: performance.sh enable|status|unlock

  enable  Select MAXN (mode 0), lock clocks, and set the fan to full speed.
  status  Show the active nvpmodel profile and clock limits.
  unlock  Release the jetson_clocks locks; the MAXN profile remains selected.

Run this script on the Jetson host, not inside the container. Changing from a
mode with a different GPU TPC mask may require a reboot; nvpmodel will say so.
EOF
}

action=${1:-status}
clock_state=/var/tmp/da3-jetson-clocks.conf
cpu_freq_dir=/sys/devices/system/cpu/cpu0/cpufreq
gpu_freq_dir=/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu
clocks_are_locked() {
  [[ -r "$cpu_freq_dir/scaling_min_freq" && -r "$cpu_freq_dir/scaling_max_freq" ]] \
    && [[ -r "$gpu_freq_dir/min_freq" && -r "$gpu_freq_dir/max_freq" ]] \
    && [[ $(<"$cpu_freq_dir/scaling_min_freq") == $(<"$cpu_freq_dir/scaling_max_freq") ]] \
    && [[ $(<"$gpu_freq_dir/min_freq") == $(<"$gpu_freq_dir/max_freq") ]]
}

case "$action" in
  enable)
    sudo nvpmodel -m 0
    if ! nvpmodel -q | head -n 1 | grep -q MAXN; then
      echo "MAXN is not active yet. Reboot the Jetson, then run this command again." >&2
      exit 1
    fi
    if ! sudo test -f "$clock_state"; then
      sudo jetson_clocks --store "$clock_state"
    fi
    sudo jetson_clocks
    sudo jetson_clocks --fan
    if ! clocks_are_locked; then
      echo "jetson_clocks did not pin CPU/GPU minimums to their maxima." >&2
      exit 1
    fi
    ;;
  status)
    ;;
  unlock)
    if ! sudo test -f "$clock_state"; then
      echo "No saved clock state exists at $clock_state" >&2
      exit 1
    fi
    sudo jetson_clocks --restore "$clock_state"
    ;;
  --help|-h)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

nvpmodel -q
sudo jetson_clocks --show
