#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
STATE_FILE="${CLEANINGCAR_PERF_LOCK_STATE:-/dev/shm/cleaningcar_fixed_freq_restore.sh}"

log() {
  echo "[perf-lock] $*"
}

warn() {
  echo "[perf-lock] WARNING: $*" >&2
}

is_false_like() {
  case "${1:-}" in
    0|false|FALSE|False|no|NO|No|off|OFF|Off)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

use_sudo_enabled() {
  if is_false_like "${CLEANINGCAR_PERF_LOCK_USE_SUDO:-1}"; then
    return 1
  fi
  return 0
}

shell_escape() {
  printf '%q' "$1"
}

append_restore_cmd() {
  local tmp_file="$1"
  local path="$2"
  local value="$3"
  printf "printf '%%s' %s > %s 2>/dev/null || true\n" \
    "$(shell_escape "$value")" \
    "$(shell_escape "$path")" >> "$tmp_file"
}

ensure_root_or_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    return 0
  fi
  if use_sudo_enabled && command -v sudo >/dev/null 2>&1; then
    exec sudo -n bash "$0" "$@"
  fi
  warn "need root or passwordless sudo to change frequencies (or set CLEANINGCAR_PERF_LOCK_USE_SUDO=1)"
  exit 2
}

read_file() {
  local path="$1"
  [ -r "$path" ] || return 1
  tr -d '\n' < "$path"
}

write_file() {
  local path="$1"
  local value="$2"
  printf '%s' "$value" > "$path"
}

list_cpu_policies() {
  find /sys/devices/system/cpu/cpufreq -maxdepth 1 -type d -name 'policy*' | sort
}

list_cpuidle_disable_paths() {
  find /sys/devices/system/cpu -path '*/cpuidle/state1/disable' | sort
}

list_devfreq_nodes() {
  local candidates=(
    "/sys/class/devfreq/fdab0000.npu"
    "/sys/class/devfreq/dmc"
    "/sys/class/devfreq/fb000000.gpu"
  )
  local path
  for path in "${candidates[@]}"; do
    [ -d "$path" ] && printf '%s\n' "$path"
  done
}

supports_governor() {
  local path="$1"
  local governor="$2"
  local available
  available="$(read_file "$path/available_governors" 2>/dev/null || true)"
  [[ " $available " == *" $governor "* ]]
}

status_cpu() {
  local policy
  for policy in $(list_cpu_policies); do
    local gov cur maxf
    gov="$(read_file "$policy/scaling_governor" 2>/dev/null || echo '?')"
    cur="$(read_file "$policy/scaling_cur_freq" 2>/dev/null || echo '?')"
    maxf="$(read_file "$policy/cpuinfo_max_freq" 2>/dev/null || echo '?')"
    log "cpu $(basename "$policy") gov=$gov cur=$cur max=$maxf"
  done
}

status_cpuidle() {
  local path
  for path in $(list_cpuidle_disable_paths); do
    local value
    value="$(read_file "$path" 2>/dev/null || echo '?')"
    log "cpuidle ${path#/sys/devices/system/cpu/}=$value"
  done
}

status_devfreq() {
  local node
  for node in $(list_devfreq_nodes); do
    local gov cur maxf minf
    gov="$(read_file "$node/governor" 2>/dev/null || echo '?')"
    cur="$(read_file "$node/cur_freq" 2>/dev/null || echo '?')"
    maxf="$(read_file "$node/max_freq" 2>/dev/null || echo '?')"
    minf="$(read_file "$node/min_freq" 2>/dev/null || echo '?')"
    log "devfreq $(basename "$node") gov=$gov cur=$cur min=$minf max=$maxf"
  done
}

status_all() {
  status_cpu
  status_cpuidle
  status_devfreq
}

create_state_file_if_missing() {
  if [ -f "$STATE_FILE" ]; then
    return 0
  fi
  local tmp_file
  tmp_file="$(mktemp)"
  {
    echo "#!/usr/bin/env bash"
    echo "set +e"
  } > "$tmp_file"

  local policy
  for policy in $(list_cpu_policies); do
    local gov cur
    gov="$(read_file "$policy/scaling_governor" 2>/dev/null || true)"
    cur="$(read_file "$policy/scaling_cur_freq" 2>/dev/null || true)"
    [ -n "$gov" ] && append_restore_cmd "$tmp_file" "$policy/scaling_governor" "$gov"
    if [ -n "$cur" ] && [ -w "$policy/scaling_setspeed" ]; then
      append_restore_cmd "$tmp_file" "$policy/scaling_setspeed" "$cur"
    fi
  done

  local path
  for path in $(list_cpuidle_disable_paths); do
    local value
    value="$(read_file "$path" 2>/dev/null || true)"
    [ -n "$value" ] && append_restore_cmd "$tmp_file" "$path" "$value"
  done

  local node
  for node in $(list_devfreq_nodes); do
    local gov
    gov="$(read_file "$node/governor" 2>/dev/null || true)"
    [ -n "$gov" ] && append_restore_cmd "$tmp_file" "$node/governor" "$gov"
  done

  chmod +x "$tmp_file"
  mv "$tmp_file" "$STATE_FILE"
  log "saved restore state to $STATE_FILE"
}

apply_cpu_lock() {
  local policy
  for policy in $(list_cpu_policies); do
    if supports_governor "$policy" "performance"; then
      write_file "$policy/scaling_governor" "performance" || warn "failed to set performance on $policy"
      continue
    fi
    if supports_governor "$policy" "userspace" && [ -w "$policy/scaling_setspeed" ]; then
      local maxf
      maxf="$(read_file "$policy/cpuinfo_max_freq" 2>/dev/null || true)"
      write_file "$policy/scaling_governor" "userspace" || warn "failed to set userspace on $policy"
      [ -n "$maxf" ] && write_file "$policy/scaling_setspeed" "$maxf" || true
    fi
  done
}

apply_cpuidle_lock() {
  local path
  for path in $(list_cpuidle_disable_paths); do
    write_file "$path" "1" || true
  done
}

apply_devfreq_lock() {
  local node
  for node in $(list_devfreq_nodes); do
    if supports_governor "$node" "performance"; then
      write_file "$node/governor" "performance" || warn "failed to set performance on $node"
      continue
    fi
    if supports_governor "$node" "userspace"; then
      local userspace_target=""
      if [ -w "$node/userspace/set_freq" ]; then
        userspace_target="$node/userspace/set_freq"
      elif [ -w "$node/set_freq" ]; then
        userspace_target="$node/set_freq"
      fi
      local maxf
      maxf="$(read_file "$node/max_freq" 2>/dev/null || true)"
      write_file "$node/governor" "userspace" || warn "failed to set userspace on $node"
      if [ -n "$userspace_target" ] && [ -n "$maxf" ]; then
        write_file "$userspace_target" "$maxf" || warn "failed to set max freq on $node"
      fi
    fi
  done
}

apply_all() {
  create_state_file_if_missing
  apply_cpuidle_lock
  apply_cpu_lock
  apply_devfreq_lock
  log "applied fixed-frequency/performance settings"
  status_all
}

restore_all() {
  if [ ! -f "$STATE_FILE" ]; then
    warn "state file not found: $STATE_FILE"
    exit 0
  fi
  bash "$STATE_FILE" || warn "restore script exited non-zero"
  rm -f "$STATE_FILE"
  log "restored previous settings"
  status_all
}

case "$ACTION" in
  apply)
    ensure_root_or_sudo "$ACTION"
    apply_all
    ;;
  restore)
    ensure_root_or_sudo "$ACTION"
    restore_all
    ;;
  status)
    status_all
    ;;
  *)
    echo "usage: $0 {apply|restore|status}" >&2
    exit 1
    ;;
esac
