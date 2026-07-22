#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  Unified Web Server Launcher
#  - Self-healing environment: checks and fixes dependencies
#  - Single script: no separate install script needed
#  - Idempotent: safe to run every time, installs only what's missing
# ============================================================

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv-gst"
PACKAGES_DIR="$SCRIPT_DIR/packages"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-8000}"
SERVICE_NAME="web_server_${WEB_PORT}"
PID_FILE="$SCRIPT_DIR/${SERVICE_NAME}.pid"
LOG_FILE="$SCRIPT_DIR/${SERVICE_NAME}.log"
ACTION="${1:-start}"

# Redirect all output to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

# Temp dir for apt source workaround, cleaned up on exit
_APT_TMPDIR=""
_cleanup() {
  if [ -n "${_APT_TMPDIR:-}" ] && [ -d "${_APT_TMPDIR:-}" ]; then
    rm -rf "$_APT_TMPDIR"
  fi
}
trap _cleanup EXIT

# ---- Logging ----

log_setup()  { echo "[setup] $*" >&2; }
log_warn()   { echo "[setup] WARNING: $*" >&2; }
log_error()  { echo "[setup] ERROR: $*" >&2; }
log_service(){ echo "[service] $*" >&2; }

# ---- Diagnostic dump: prints system state for remote debugging ----

dump_diagnostics() {
  local stage="${1:-unknown}"
  echo ""
  echo "========== DIAGNOSTICS (stage: $stage) =========="
  echo "--- System ---"
  echo "hostname: $(hostname 2>/dev/null || echo N/A)"
  echo "kernel:  $(uname -a 2>/dev/null || echo N/A)"
  echo "arch:    $(uname -m 2>/dev/null || echo N/A)"
  echo "user:    $(id 2>/dev/null || echo N/A)"
  echo "date:    $(date 2>/dev/null || echo N/A)"
  echo "uptime:  $(uptime 2>/dev/null || echo N/A)"
  echo ""
  echo "--- Disk ---"
  df -h "$SCRIPT_DIR" 2>/dev/null || echo "(df failed)"
  echo ""
  echo "--- Memory ---"
  free -h 2>/dev/null || echo "(free failed)"
  echo ""
  echo "--- Python ---"
  local _py
  for _py in /usr/bin/python3 /usr/bin/python3.{8,9,10,11,12} "$VENV_DIR/bin/python"; do
    if [ -x "$_py" ]; then
      local _ver
      _ver="$("$_py" -V 2>&1)" || _ver="(version check failed)"
      echo "$_py -> $_ver"
    fi
  done
  echo ""
  echo "--- Network ---"
  local _ip
  _ip="$(hostname -I 2>/dev/null | head -c 200)" || _ip="(hostname -I failed)"
  echo "ip: $_ip"
  echo "port $WEB_PORT:"
  ss -ltnp 2>/dev/null | grep ":${WEB_PORT}" || echo "  (not listening)"
  echo ""
  echo "--- Project ---"
  echo "script_dir:  $SCRIPT_DIR"
  echo "venv_dir:    $VENV_DIR (exists: $([ -d "$VENV_DIR" ] && echo YES || echo NO))"
  echo "packages:    $PACKAGES_DIR (exists: $([ -d "$PACKAGES_DIR" ] && echo YES || echo NO))"
  echo "config:      $(resolve_config_path 2>/dev/null || echo NOT-FOUND)"
  echo "requirements: $([ -f "$SCRIPT_DIR/requirements.txt" ] && echo YES || echo NO)"
  if [ -d "$VENV_DIR" ]; then
    echo "venv pip packages count: $( [ -x "$VENV_DIR/bin/pip" ] && "$VENV_DIR/bin/pip" list --format=columns 2>/dev/null | tail -n +3 | wc -l || echo 'N/A' )"
  fi
  echo "==============================================="
  echo ""
}

fail() {
  local stage="$1"; shift
  log_error "[$stage] $*"
  dump_diagnostics "$stage"
  log_error ">>> Script halted at stage: $stage"
  exit 1
}

# ---- Config ----

resolve_config_path() {
  local config_override="${CONFIG_PATH:-}"
  local default_config="$SCRIPT_DIR/configs/config.json"
  local legacy_config="$SCRIPT_DIR/config.json"
  if [ -n "$config_override" ]; then
    if [ -f "$config_override" ]; then
      printf '%s\n' "$config_override"
      return 0
    fi
    echo "config file not found: $config_override" >&2
    return 1
  fi
  if [ -f "$default_config" ]; then
    printf '%s\n' "$default_config"
    return 0
  fi
  if [ -f "$legacy_config" ]; then
    printf '%s\n' "$legacy_config"
    return 0
  fi
  echo "config file not found: $default_config" >&2
  return 1
}

# ============================================================
#  APT helpers
# ============================================================

ensure_sudo_prefix() {
  if [ "$(id -u)" -eq 0 ]; then
    return 0
  fi
  if command -v sudo >/dev/null 2>&1; then
    printf 'sudo'
    return 0
  fi
  log_error "sudo not found and not running as root"
  log_error "  possible cause: running as non-root user on a minimal system without sudo"
  log_error "  fix: run as root (su -) or install sudo (apt-get install sudo)"
  exit 1
}

_APT_OPTIONS=()

_prepare_apt_main_sources_only() {
  local sources_file="/etc/apt/sources.list"
  [ -f "$sources_file" ] || return 1
  grep -Eq '^[[:space:]]*deb([[:space:]]|\[)' "$sources_file" || return 1
  if [ -z "$_APT_TMPDIR" ]; then
    _APT_TMPDIR="$(mktemp -d)"
  fi
  _APT_OPTIONS=(
    -o "Dir::Etc::sourcelist=$sources_file"
    -o "Dir::Etc::sourceparts=$_APT_TMPDIR"
    -o "APT::Get::List-Cleanup=0"
  )
}

_run_apt_cmd() {
  local sudo_prefix="$1"; shift
  if [ -n "$sudo_prefix" ]; then
    "$sudo_prefix" apt-get "${_APT_OPTIONS[@]}" "$@"
  else
    apt-get "${_APT_OPTIONS[@]}" "$@"
  fi
}

install_apt_packages() {
  local sudo_prefix="$1"; shift
  _APT_OPTIONS=()
  if _run_apt_cmd "$sudo_prefix" install -y "$@"; then
    return 0
  fi
  log_setup "apt install failed, refreshing package indexes"
  _APT_OPTIONS=()
  if ! _run_apt_cmd "$sudo_prefix" update; then
    if _prepare_apt_main_sources_only; then
      _run_apt_cmd "$sudo_prefix" update || return 1
    else
      return 1
    fi
  fi
  _APT_OPTIONS=()
  _run_apt_cmd "$sudo_prefix" install -y "$@"
}

# ============================================================
#  Dependency check primitives
# ============================================================

find_system_python() {
  if [ -x /usr/bin/python3 ]; then
    printf '/usr/bin/python3'
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  return 1
}

is_python_version_supported() {
  local python_bin="$1"
  local version
  version="$("$python_bin" -c 'import sys; v=sys.version_info; print(f"{v.major}.{v.minor}")' 2>/dev/null)" || return 1
  case "$version" in
    3.8|3.9|3.10|3.11|3.12) return 0 ;;
    *) return 1 ;;
  esac
}

is_apt_package_installed() {
  dpkg -s "$1" >/dev/null 2>&1
}

is_venv_functional() {
  local python_bin="$VENV_DIR/bin/python"
  [ -x "$python_bin" ] || return 1
  "$python_bin" -V >/dev/null 2>&1 || return 1
}

has_opencv_gstreamer() {
  local python_bin="$1"
  "$python_bin" -c '
import cv2
info = cv2.getBuildInformation()
for line in info.splitlines():
    if "GStreamer:" in line and "YES" in line:
        raise SystemExit(0)
raise SystemExit(1)
' 2>/dev/null
}

is_file_same() {
  [ -f "$1" ] && [ -f "$2" ] && cmp -s "$1" "$2"
}

can_import() {
  local python_bin="$1"
  local module="$2"
  "$python_bin" -c "import $module" >/dev/null 2>&1
}

# ============================================================
#  Self-healing environment ensure functions
# ============================================================

ensure_system_python() {
  log_setup "[python] checking system python3..."
  local python_bin
  if python_bin="$(find_system_python)"; then
    local version
    version="$("$python_bin" -c 'import sys; v=sys.version_info; print(f"{v.major}.{v.minor}")' 2>/dev/null)" || version="unknown"
    if is_python_version_supported "$python_bin"; then
      log_setup "[python] found: $python_bin ($version)"
      printf '%s\n' "$python_bin"
      return 0
    fi
    log_warn "[python] default python3 ($python_bin) is version $version, not in supported range 3.8-3.12"
    log_warn "[python] searching for alternative versions..."
    for ver in 3.8 3.9 3.10 3.11 3.12; do
      if [ -x "/usr/bin/python${ver}" ] && is_python_version_supported "/usr/bin/python${ver}"; then
        log_setup "[python] found alternative: /usr/bin/python${ver}"
        printf '/usr/bin/python%s\n' "$ver"
        return 0
      fi
    done
    log_error "[python] no supported python3 (3.8-3.12) found on system"
    log_error "  possible cause: system only has python 3.13+ or 3.7-, or python3 is not installed"
    log_error "  detected: $python_bin -> $version"
    log_error "  fix: install a supported python3 version, e.g. apt-get install python3.10"
    fail "python" "no supported python3 version found (need 3.8-3.12)"
  fi
  log_error "[python] python3 binary not found anywhere on system"
  log_error "  possible cause: OS installed without python3, or python3 removed accidentally"
  log_error "  fix: apt-get install python3 python3-venv"
  fail "python" "python3 not found"
}

ensure_apt_dependencies() {
  log_setup "[apt] checking system packages..."
  local required_packages=(
    python3-venv
    python3-pip
    python3-opencv
    gstreamer1.0-tools
    gstreamer1.0-plugins-base
    gstreamer1.0-plugins-good
    gstreamer1.0-plugins-bad
    gstreamer1.0-libav
  )

  local missing=()
  for pkg in "${required_packages[@]}"; do
    if ! is_apt_package_installed "$pkg"; then
      missing+=("$pkg")
    fi
  done

  if [ ${#missing[@]} -eq 0 ]; then
    log_setup "[apt] all required packages installed"
    return 0
  fi

  log_setup "[apt] missing packages: ${missing[*]}"
  log_setup "[apt] installing ${#missing[@]} packages..."
  local sudo_prefix
  sudo_prefix="$(ensure_sudo_prefix)" || sudo_prefix=""
  if ! install_apt_packages "$sudo_prefix" "${missing[@]}"; then
    log_error "[apt] apt-get install failed for: ${missing[*]}"
    log_error "  possible cause 1: no network / DNS resolution failure (check: ping baidu.com)"
    log_error "  possible cause 2: apt sources list broken (check: cat /etc/apt/sources.list)"
    log_error "  possible cause 3: disk full (check: df -h)"
    log_error "  possible cause 4: dpkg locked by another process (check: lsof /var/lib/dpkg/lock)"
    log_error "  fix: resolve above issues, then re-run this script"
    fail "apt" "apt-get install failed"
  fi
  log_setup "[apt] packages installed successfully"
}

ensure_venv() {
  local python_sys="$1"

  log_setup "[venv] checking virtual environment: $VENV_DIR"
  if is_venv_functional; then
    log_setup "[venv] ok"
    return 0
  fi

  if [ -d "$VENV_DIR" ]; then
    log_setup "[venv] venv directory exists but python broken, removing: $VENV_DIR"
    log_warn "[venv] possible cause: system python was upgraded/reinstalled, venv symlinks are stale"
    rm -rf "$VENV_DIR"
  fi

  log_setup "[venv] creating venv with: $python_sys -m venv --system-site-packages $VENV_DIR"
  if ! "$python_sys" -m venv --system-site-packages "$VENV_DIR" 2>&1; then
    log_error "[venv] python -m venv failed"
    log_error "  possible cause 1: python3-venv package not installed (should have been installed by apt step)"
    log_error "  possible cause 2: disk full, cannot create venv directory (check: df -h)"
    log_error "  possible cause 3: permission denied on $SCRIPT_DIR (check: ls -la $SCRIPT_DIR)"
    log_error "  fix: ensure python3-venv is installed and directory is writable"
    fail "venv" "venv creation failed"
  fi

  if ! is_venv_functional; then
    log_error "[venv] venv created but python binary still not functional"
    log_error "  possible cause: venv was created but bin/python symlink is broken"
    log_error "  diagnostic: ls -la $VENV_DIR/bin/python*"
    ls -la "$VENV_DIR/bin/python"* 2>/dev/null || true
    fail "venv" "venv python not functional after creation"
  fi

  log_setup "[venv] upgrading pip..."
  if ! "$VENV_DIR/bin/python" -m pip install --upgrade pip 2>&1; then
    log_warn "[venv] pip upgrade failed (non-fatal, continuing)"
    log_warn "[venv] possible cause: no network, but cached pip should work"
  fi
  log_setup "[venv] ready"
}

ensure_opencv_gstreamer() {
  local python_bin="$1"
  log_setup "[opencv] checking cv2 GStreamer support..."
  if has_opencv_gstreamer "$python_bin"; then
    log_setup "[opencv] GStreamer support ok"
    return 0
  fi

  log_error "[opencv] cv2 has no GStreamer support"
  log_error "  possible cause 1: python3-opencv installed but built without GStreamer (common on non-RK boards)"
  log_error "  possible cause 2: opencv was replaced by pip-installed version (pip opencv-python has no GStreamer)"
  log_error "  diagnostic: $python_bin -c 'import cv2; print(cv2.getBuildInformation())' | grep GStreamer"
  log_setup "[opencv] attempting reinstall of python3-opencv..."
  local sudo_prefix
  sudo_prefix="$(ensure_sudo_prefix)" || sudo_prefix=""
  install_apt_packages "$sudo_prefix" --reinstall python3-opencv 2>&1 || true
  if ! has_opencv_gstreamer "$python_bin"; then
    log_error "[opencv] GStreamer support still unavailable after reinstall"
    log_error "  this usually means the OS's python3-opencv package was not compiled with GStreamer"
    log_error "  fix: use an OS image that includes GStreamer-enabled opencv, or build opencv from source"
    fail "opencv" "GStreamer support not available after reinstall"
  fi
  log_setup "[opencv] GStreamer support restored after reinstall"
}

ensure_native_libs() {
  [ -d "$PACKAGES_DIR" ] || return 0

  log_setup "[native-libs] checking native libraries from packages/ ..."
  local sudo_prefix
  sudo_prefix="$(ensure_sudo_prefix)" || sudo_prefix=""
  local need_ldconfig=0

  # librknnrt.so
  if [ -f "$PACKAGES_DIR/librknnrt.so" ]; then
    if ! is_file_same "$PACKAGES_DIR/librknnrt.so" /usr/lib/librknnrt.so; then
      log_setup "[native-libs] installing librknnrt.so -> /usr/lib/"
      if [ -n "$sudo_prefix" ]; then
        "$sudo_prefix" cp -f "$PACKAGES_DIR/librknnrt.so" /usr/lib/librknnrt.so
        "$sudo_prefix" chmod 755 /usr/lib/librknnrt.so || true
      else
        cp -f "$PACKAGES_DIR/librknnrt.so" /usr/lib/librknnrt.so
        chmod 755 /usr/lib/librknnrt.so || true
      fi
      need_ldconfig=1
    fi
  fi

  # librga.so
  if [ -f "$PACKAGES_DIR/librga.so" ]; then
    if ! is_file_same "$PACKAGES_DIR/librga.so" /usr/local/lib/librga.so; then
      log_setup "[native-libs] installing librga.so -> /usr/local/lib/"
      if [ -n "$sudo_prefix" ]; then
        "$sudo_prefix" cp -f "$PACKAGES_DIR/librga.so" /usr/local/lib/librga.so
        "$sudo_prefix" chmod 755 /usr/local/lib/librga.so || true
      else
        cp -f "$PACKAGES_DIR/librga.so" /usr/local/lib/librga.so
        chmod 755 /usr/local/lib/librga.so || true
      fi
      need_ldconfig=1
    fi
  fi

  # im2d.h
  if [ -f "$PACKAGES_DIR/im2d.h" ]; then
    if ! is_file_same "$PACKAGES_DIR/im2d.h" /usr/local/include/rga/im2d.h; then
      log_setup "[native-libs] installing im2d.h -> /usr/local/include/rga/"
      if [ -n "$sudo_prefix" ]; then
        "$sudo_prefix" mkdir -p /usr/local/include/rga
        "$sudo_prefix" cp -f "$PACKAGES_DIR/im2d.h" /usr/local/include/rga/im2d.h
      else
        mkdir -p /usr/local/include/rga
        cp -f "$PACKAGES_DIR/im2d.h" /usr/local/include/rga/im2d.h
      fi
    fi
  fi

  if [ "$need_ldconfig" -eq 1 ]; then
    log_setup "[native-libs] running ldconfig..."
    if [ -n "$sudo_prefix" ]; then
      "$sudo_prefix" ldconfig || log_warn "[native-libs] ldconfig failed (may cause .so loading issues)"
    else
      ldconfig || log_warn "[native-libs] ldconfig failed (may cause .so loading issues)"
    fi
  fi
  log_setup "[native-libs] ok"
}

ensure_rknn_wheel() {
  local python_bin="$1"
  local python_sys="$2"

  log_setup "[rknn] checking RKNNLite..."
  if can_import "$python_bin" "rknnlite.api"; then
    log_setup "[rknn] RKNNLite importable, ok"
    return 0
  fi

  if [ ! -d "$PACKAGES_DIR" ]; then
    log_warn "[rknn] RKNNLite not importable and no packages/ directory"
    log_warn "[rknn] this is expected on non-RK hardware (x86 PC, etc.)"
    return 0
  fi

  local py_tag
  py_tag="$("$python_sys" -c 'import sys; v=sys.version_info; print(f"cp{v.major}{v.minor}")')" || py_tag=""
  if [ -z "$py_tag" ]; then
    log_warn "[rknn] could not determine python ABI tag"
    return 0
  fi
  log_setup "[rknn] searching for wheel with tag: $py_tag"

  local whl_file=""
  whl_file="$(ls "$PACKAGES_DIR"/rknn_toolkit_lite*"$py_tag"*.whl 2>/dev/null | head -n 1)" || true

  if [ -n "$whl_file" ]; then
    log_setup "[rknn] installing: $whl_file"
    if ! "$python_bin" -m pip install "$whl_file" 2>&1; then
      log_error "[rknn] pip install failed for RKNN wheel"
      log_error "  possible cause 1: wheel is compiled for different python version"
      log_error "  possible cause 2: wheel is compiled for different architecture (arm64 vs armhf)"
      log_error "  possible cause 3: missing system library dependency (librknnrt.so)"
      log_error "  diagnostic: file $whl_file"
      log_error "  diagnostic: ldd /usr/lib/librknnrt.so 2>/dev/null"
      fail "rknn" "pip install rknn wheel failed"
    fi
    if ! can_import "$python_bin" "rknnlite.api"; then
      log_error "[rknn] wheel installed but rknnlite.api still not importable"
      log_error "  possible cause: librknnrt.so missing or wrong version"
      log_error "  diagnostic: $python_bin -c 'from rknnlite.api import RKNNLite'"
      fail "rknn" "rknnlite.api not importable after wheel install"
    fi
    log_setup "[rknn] installed successfully"
  else
    log_warn "[rknn] RKNNLite not importable, no matching wheel for $py_tag in packages/"
    log_warn "[rknn] available wheels:"
    ls -la "$PACKAGES_DIR"/rknn_toolkit_lite*.whl 2>/dev/null || log_warn "[rknn]   (none found)"
    log_warn "[rknn] expected tag pattern: *$py_tag*.whl"
    log_warn "[rknn] this is expected on non-RK hardware or if python version does not match the wheel"
  fi
}

ensure_pip_requirements() {
  local python_bin="$1"
  local req_file="$SCRIPT_DIR/requirements.txt"

  [ -f "$req_file" ] || return 0

  log_setup "[pip] checking requirements: $req_file"
  local hash_file="$VENV_DIR/.requirements_hash"
  local current_hash
  current_hash="$(md5sum "$req_file" | cut -d' ' -f1)"

  # Fast path: hash unchanged, dependencies consistent, packages actually present
  if [ -f "$hash_file" ]; then
    local saved_hash=""
    saved_hash="$(cat "$hash_file" 2>/dev/null)" || true
    if [ "$current_hash" = "$saved_hash" ]; then
      local pip_ok=true
      # pip check: dependency conflicts
      if ! "$python_bin" -m pip check >/dev/null 2>&1; then
        pip_ok=false
        log_setup "[pip] pip check failed (dependency conflicts detected)"
      fi
      # Spot-check: first package must actually be installed
      if $pip_ok; then
        local first_pkg
        first_pkg="$(grep -vE '^\s*(#|$)' "$req_file" | head -n1 | sed -E 's/^([a-zA-Z0-9._-]+).*/\1/')" || true
        if [ -n "$first_pkg" ] && ! "$python_bin" -m pip show "$first_pkg" >/dev/null 2>&1; then
          pip_ok=false
          log_setup "[pip] spot-check failed: '$first_pkg' not installed (packages may be missing)"
        fi
      fi
      if $pip_ok; then
        log_setup "[pip] requirements up-to-date (hash + pip check + spot-check passed)"
        return 0
      fi
      log_setup "[pip] environment inconsistent, reinstalling requirements"
    else
      log_setup "[pip] requirements.txt changed (hash mismatch), reinstalling"
    fi
  fi

  log_setup "[pip] running: $python_bin -m pip install -r $req_file"
  if ! "$python_bin" -m pip install -r "$req_file" 2>&1; then
    log_error "[pip] pip install failed"
    log_error "  possible cause 1: no network (pip needs to download packages on first install)"
    log_error "  possible cause 2: DNS failure (check: ping pypi.org)"
    log_error "  possible cause 3: disk full (check: df -h)"
    log_error "  possible cause 4: pip cache corrupted (try: rm -rf ~/.cache/pip)"
    log_error "  possible cause 5: python version incompatible with a package"
    log_error "  diagnostic: $python_bin --version"
    log_error "  diagnostic: cat $req_file"
    fail "pip" "pip install -r requirements.txt failed"
  fi
  printf '%s' "$current_hash" > "$hash_file"
  log_setup "[pip] requirements installed"
}

# ---- Main environment gate ----

ensure_environment() {
  echo "" >&2
  log_setup "========== Environment Check =========="
  local python_sys
  python_sys="$(ensure_system_python)"
  ensure_apt_dependencies
  ensure_venv "$python_sys"
  local python_bin="$VENV_DIR/bin/python"
  ensure_opencv_gstreamer "$python_bin"
  ensure_native_libs
  ensure_rknn_wheel "$python_bin" "$python_sys"
  ensure_pip_requirements "$python_bin"
  log_setup "========== Environment Ready =========="
  log_setup "venv: $VENV_DIR"
  echo "" >&2
}

# ============================================================
#  Service management
# ============================================================

port_owner_pids() {
  local port="$1"
  ss -ltnp 2>/dev/null | awk -v port=":$port" '
    $4 ~ port"$" {
      if (match($0, /pid=[0-9]+/)) {
        pid = substr($0, RSTART + 4, RLENGTH - 4)
        print pid
      }
    }
  ' | sort -u
}

describe_pid() {
  local pid="$1"
  if [ ! -d "/proc/$pid" ]; then
    return 1
  fi
  local cmdline
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  local cwd
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  echo "pid=$pid cwd=$cwd cmd=$cmdline"
}

stop_pid() {
  local pid="$1"
  if [ ! -d "/proc/$pid" ]; then
    return 0
  fi
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 10); do
    if [ ! -d "/proc/$pid" ]; then
      return 0
    fi
    sleep 1
  done
  kill -9 "$pid" 2>/dev/null || true
}

stop_same_project_port_owners() {
  local port="$1"
  local pid
  while read -r pid; do
    [ -z "$pid" ] && continue
    if [ ! -d "/proc/$pid" ]; then
      continue
    fi
    local cmdline
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    local cwd
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
    if [[ "$cmdline" == *"$SCRIPT_DIR"* ]] || [[ "$cwd" == *"$SCRIPT_DIR"* ]] || [[ "$cmdline" == *"web.server"* ]]; then
      log_service "stop existing same-project process on port $port: $(describe_pid "$pid")"
      stop_pid "$pid"
    fi
  done < <(port_owner_pids "$port")
}

ensure_port_available() {
  local port="$1"
  local owners
  owners="$(port_owner_pids "$port" || true)"
  if [ -z "$owners" ]; then
    return 0
  fi
  stop_same_project_port_owners "$port"
  owners="$(port_owner_pids "$port" || true)"
  if [ -z "$owners" ]; then
    return 0
  fi
  log_error "[port] port $port is already in use by unrelated process:"
  local pid
  while read -r pid; do
    [ -z "$pid" ] && continue
    describe_pid "$pid" || true
  done <<< "$owners"
  log_error "  possible cause 1: another web server is running on port $port"
  log_error "  possible cause 2: previous instance did not shut down cleanly"
  log_error "  fix: stop the conflicting process, or change port: WEB_PORT=8001 $0 start"
  fail "port" "port $port occupied by unrelated process"
}

wait_for_server_ready() {
  local pid="$1"
  local port="$2"
  for _ in $(seq 1 30); do
    if [ ! -d "/proc/$pid" ]; then
      return 1
    fi
    local owners
    owners="$(port_owner_pids "$port" || true)"
    if echo "$owners" | grep -qx "$pid"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_cpu_monitor() {
  local pid="$1"
  local cpu_limit="${MONITOR_CPU_LIMIT:-600}"
  local interval="${MONITOR_INTERVAL:-10}"
  (
    while ps -p "$pid" >/dev/null 2>&1; do
      local cpu_raw
      cpu_raw="$(ps -p "$pid" -o %cpu= 2>/dev/null | awk '{print int($1)}')"
      if [ -n "$cpu_raw" ] && [ "$cpu_raw" -gt "$cpu_limit" ]; then
        log_service "pid=$pid cpu=${cpu_raw}% limit=${cpu_limit}%"
        kill "$pid" 2>/dev/null || true
        sleep 5
        if ps -p "$pid" >/dev/null 2>&1; then
          kill -9 "$pid" 2>/dev/null || true
        fi
        log_service "process killed, check log: $LOG_FILE"
        break
      fi
      sleep "$interval"
    done
  ) &
}

do_stop() {
  local stopped=0
  if [ -f "$PID_FILE" ]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "${pid:-}" ]; then
      log_service "stop pid from pid file: $pid"
      stop_pid "$pid"
      stopped=1
    fi
    rm -f "$PID_FILE"
  fi

  local owners
  owners="$(port_owner_pids "$WEB_PORT" || true)"
  if [ -z "$owners" ]; then
    if [ "$stopped" -eq 0 ]; then
      log_service "no listener on port $WEB_PORT"
    fi
    return 0
  fi

  local pid
  while read -r pid; do
    [ -z "$pid" ] && continue
    if [ ! -d "/proc/$pid" ]; then
      continue
    fi
    local cmdline
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    local cwd
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
    if [[ "$cmdline" == *"$SCRIPT_DIR"* ]] || [[ "$cwd" == *"$SCRIPT_DIR"* ]] || [[ "$cmdline" == *"web.server"* ]]; then
      log_service "stop port owner: $(describe_pid "$pid")"
      stop_pid "$pid"
      stopped=1
    else
      log_service "port $WEB_PORT occupied by unrelated process:"
      describe_pid "$pid" || true
    fi
  done <<< "$owners"

  sleep 1
  local remaining
  remaining="$(port_owner_pids "$WEB_PORT" || true)"
  if [ -n "$remaining" ]; then
    log_service "port $WEB_PORT still occupied"
    return 1
  fi
  log_service "port $WEB_PORT released"
}

do_status() {
  local owners
  owners="$(port_owner_pids "$WEB_PORT" || true)"
  if [ -z "$owners" ]; then
    log_service "stopped"
    return 1
  fi
  log_service "listening on port $WEB_PORT"
  local pid
  while read -r pid; do
    [ -z "$pid" ] && continue
    describe_pid "$pid" || true
  done <<< "$owners"
}

do_start() {
  local config_path
  config_path="$(resolve_config_path)" || {
    log_error "[config] failed to locate config file"
    log_error "  expected: $SCRIPT_DIR/configs/config.json or $SCRIPT_DIR/config.json"
    log_error "  or set env: CONFIG_PATH=/path/to/config.json"
    log_error "  possible cause: project files incomplete, config.json missing"
    log_error "  fix: copy config file to expected location"
    fail "config" "config file not found"
  }
  log_setup "[config] using: $config_path"

  ensure_environment
  local python_bin="$VENV_DIR/bin/python"

  if [ -f "$PID_FILE" ]; then
    local old_pid
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$old_pid" ] && ps -p "$old_pid" >/dev/null 2>&1; then
      log_service "stop old pid from pid file: $old_pid"
      stop_pid "$old_pid"
    fi
    rm -f "$PID_FILE"
  fi

  ensure_port_available "$WEB_PORT"

  cd "$SCRIPT_DIR"
  nohup "$python_bin" -m web.server --config "$config_path" --host "$WEB_HOST" --port "$WEB_PORT" >> "$LOG_FILE" 2>&1 &
  local new_pid=$!
  if wait_for_server_ready "$new_pid" "$WEB_PORT"; then
    echo "$new_pid" > "$PID_FILE"
    log_service "started pid=$new_pid host=$WEB_HOST port=$WEB_PORT log=$LOG_FILE config=$config_path"
    start_cpu_monitor "$new_pid"
  else
    log_error "[start] server failed to start within 30 seconds"
    if [ ! -d "/proc/$new_pid" ]; then
      log_error "  possible cause 1: python process crashed immediately (import error, config error)"
      log_error "  possible cause 2: missing module or dependency not detected during setup"
      log_error "  possible cause 3: config file has invalid JSON or wrong values"
    else
      log_error "  possible cause 1: server is running but not listening on port $WEB_PORT"
      log_error "  possible cause 2: server is stuck during initialization (model loading, etc.)"
      log_error "  possible cause 3: port binding delayed by more than 30 seconds"
    fi
    log_error "  check log file for details: $LOG_FILE"
    log_error "  last 50 lines of log:"
    echo "--- LOG BEGIN ---"
    tail -n 50 "$LOG_FILE" 2>/dev/null || echo "(log file not found or empty)"
    echo "--- LOG END ---"
    fail "start" "server did not become ready"
  fi
}

case "$ACTION" in
  start)
    do_start
    ;;
  stop)
    do_stop
    ;;
  restart)
    do_stop || true
    do_start
    ;;
  status)
    do_status
    ;;
  *)
    echo "usage: $0 [start|stop|restart|status]"
    exit 1
    ;;
esac
