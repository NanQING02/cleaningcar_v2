#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv-gst"
PACKAGES_DIR="$SCRIPT_DIR/packages"
APT_OPTIONS=()
APT_SOURCEPARTS_OVERRIDE=""

cleanup_install_runtime() {
  if [ -n "${APT_SOURCEPARTS_OVERRIDE:-}" ] && [ -d "${APT_SOURCEPARTS_OVERRIDE:-}" ]; then
    rm -rf "$APT_SOURCEPARTS_OVERRIDE"
  fi
}

trap cleanup_install_runtime EXIT

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

run_apt() {
  local desc="$1"
  shift
  if ! "$@"; then
    echo "[setup] apt failed: $desc"
    exit 1
  fi
}

prepare_apt_main_sources_only() {
  local sources_file="/etc/apt/sources.list"
  if [ ! -f "$sources_file" ]; then
    return 1
  fi
  if ! grep -Eq '^[[:space:]]*deb([[:space:]]|\[)' "$sources_file"; then
    return 1
  fi
  if [ -z "${APT_SOURCEPARTS_OVERRIDE:-}" ]; then
    APT_SOURCEPARTS_OVERRIDE="$(mktemp -d)"
  fi
  APT_OPTIONS=(
    -o "Dir::Etc::sourcelist=$sources_file"
    -o "Dir::Etc::sourceparts=$APT_SOURCEPARTS_OVERRIDE"
    -o "APT::Get::List-Cleanup=0"
  )
}

run_apt_cmd() {
  local sudo_prefix="$1"
  shift
  if [ -n "$sudo_prefix" ]; then
    "$sudo_prefix" apt-get "${APT_OPTIONS[@]}" "$@"
  else
    apt-get "${APT_OPTIONS[@]}" "$@"
  fi
}

refresh_apt_indexes() {
  local sudo_prefix="$1"
  APT_OPTIONS=()
  if run_apt_cmd "$sudo_prefix" update; then
    return 0
  fi
  if prepare_apt_main_sources_only; then
    echo "[setup] apt-get update failed, retry with /etc/apt/sources.list only"
    if run_apt_cmd "$sudo_prefix" update; then
      return 0
    fi
  fi
  return 1
}

install_system_packages() {
  local sudo_prefix="$1"
  shift
  APT_OPTIONS=()
  if run_apt_cmd "$sudo_prefix" install -y "$@"; then
    return 0
  fi
  echo "[setup] direct apt install failed, refreshing package indexes"
  refresh_apt_indexes "$sudo_prefix" || return 1
  run_apt_cmd "$sudo_prefix" install -y "$@"
}

detect_system_python() {
  local candidates=()
  if [ -x /usr/bin/python3 ]; then
    candidates+=("/usr/bin/python3")
  fi
  local cmd_python=""
  if command -v python3 >/dev/null 2>&1; then
    cmd_python="$(command -v python3)"
    candidates+=("$cmd_python")
  fi
  local ver
  for ver in 3.8 3.9 3.10 3.11 3.12; do
    if [ -x "/usr/bin/python${ver}" ]; then
      candidates+=("/usr/bin/python${ver}")
    fi
  done

  local python_bin
  local version
  local seen=" "
  for python_bin in "${candidates[@]}"; do
    [ -n "$python_bin" ] || continue
    case "$seen" in
      *" $python_bin "*) continue ;;
    esac
    seen="${seen}${python_bin} "
    version="$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    case "$version" in
      3.8|3.9|3.10|3.11|3.12)
        printf '%s\n' "$python_bin"
        return 0
        ;;
    esac
  done

  echo "supported python3 not found; need Python 3.8-3.12" >&2
  exit 1
}

ensure_supported_python_version() {
  local python_bin="$1"
  local version
  version="$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  case "$version" in
    3.8|3.9|3.10|3.11|3.12)
      printf '%s\n' "$version"
      return 0
      ;;
  esac
  echo "unsupported python version: $version" >&2
  exit 1
}

ensure_sudo_prefix() {
  if [ "$(id -u)" -eq 0 ]; then
    printf '%s\n' ""
    return 0
  fi
  if command -v sudo >/dev/null 2>&1; then
    printf '%s\n' "sudo"
    return 0
  fi
  echo "sudo not found and current user is not root" >&2
  exit 1
}

resolve_config_path >/dev/null

PYTHON_SYS="$(detect_system_python)"
PYTHON_VER="$(ensure_supported_python_version "$PYTHON_SYS")"
FORCE_SETUP="${FORCE_SETUP:-}"
PYTHON_BIN=""

if [ -d "$VENV_DIR" ] && [ -x "$VENV_DIR/bin/python" ] && [ -z "$FORCE_SETUP" ]; then
  if "$VENV_DIR/bin/python" -V >/dev/null 2>&1; then
    echo "[setup] use existing venv: $VENV_DIR"
    PYTHON_BIN="$VENV_DIR/bin/python"
  else
    echo "[setup] existing venv is invalid, rebuilding: $VENV_DIR"
    rm -rf "$VENV_DIR"
  fi
fi

if [ -z "$PYTHON_BIN" ]; then
  SUDO_PREFIX="$(ensure_sudo_prefix)"
  APT_PACKAGES=(
    python3-venv
    python3-pip
    python3-opencv
    gstreamer1.0-tools
    gstreamer1.0-plugins-base
    gstreamer1.0-plugins-good
    gstreamer1.0-plugins-bad
    gstreamer1.0-libav
  )

  if [ -n "$SUDO_PREFIX" ]; then
    run_apt "install python/opencv/gstreamer packages" \
      install_system_packages "$SUDO_PREFIX" "${APT_PACKAGES[@]}"
  else
    run_apt "install python/opencv/gstreamer packages" \
      install_system_packages "" "${APT_PACKAGES[@]}"
  fi

  CV2_STATUS="$("$PYTHON_SYS" - <<'EOF'
try:
    import cv2
    info = cv2.getBuildInformation()
    lines = [line for line in info.splitlines() if "GStreamer" in line]
    ok = any(("GStreamer:" in line and "YES" in line) for line in lines)
    print("OK" if ok else "NO_GST")
except Exception:
    print("NO_CV2")
EOF
)"
  if [ "$CV2_STATUS" = "NO_CV2" ]; then
    echo "[setup] cv2 not available in system python"
    exit 1
  fi
  if [ "$CV2_STATUS" = "NO_GST" ]; then
    echo "[setup] system cv2 has no GStreamer support"
    exit 1
  fi

  "$PYTHON_SYS" -m venv --system-site-packages "$VENV_DIR"
  PYTHON_BIN="$VENV_DIR/bin/python"
  "$PYTHON_BIN" -m pip install --upgrade pip

  if [ -d "$PACKAGES_DIR" ]; then
    RKN_LIB_SRC="$PACKAGES_DIR/librknnrt.so"
    if [ -f "$RKN_LIB_SRC" ]; then
      if [ -n "$SUDO_PREFIX" ]; then
        "$SUDO_PREFIX" cp -f "$RKN_LIB_SRC" /usr/lib/librknnrt.so
        "$SUDO_PREFIX" chmod 755 /usr/lib/librknnrt.so || true
        "$SUDO_PREFIX" ldconfig || true
      else
        cp -f "$RKN_LIB_SRC" /usr/lib/librknnrt.so
        chmod 755 /usr/lib/librknnrt.so || true
        ldconfig || true
      fi
    fi

    RGA_SO_SRC="$PACKAGES_DIR/librga.so"
    if [ -f "$RGA_SO_SRC" ]; then
      if [ -n "$SUDO_PREFIX" ]; then
        "$SUDO_PREFIX" cp -f "$RGA_SO_SRC" /usr/local/lib/librga.so
        "$SUDO_PREFIX" chmod 755 /usr/local/lib/librga.so || true
        "$SUDO_PREFIX" ldconfig || true
      else
        cp -f "$RGA_SO_SRC" /usr/local/lib/librga.so
        chmod 755 /usr/local/lib/librga.so || true
        ldconfig || true
      fi
    fi

    RGA_HDR_SRC="$PACKAGES_DIR/im2d.h"
    if [ -f "$RGA_HDR_SRC" ]; then
      if [ -n "$SUDO_PREFIX" ]; then
        "$SUDO_PREFIX" mkdir -p /usr/local/include/rga
        "$SUDO_PREFIX" cp -f "$RGA_HDR_SRC" /usr/local/include/rga/im2d.h
      else
        mkdir -p /usr/local/include/rga
        cp -f "$RGA_HDR_SRC" /usr/local/include/rga/im2d.h
      fi
    fi

    PY_MAJOR="$(echo "$PYTHON_VER" | cut -d. -f1)"
    PY_MINOR="$(echo "$PYTHON_VER" | cut -d. -f2)"
    PY_TAG="cp${PY_MAJOR}${PY_MINOR}"
    RKNN_WHL=""
    if ls "$PACKAGES_DIR"/rknn_toolkit_lite*"$PY_TAG"*.whl >/dev/null 2>&1; then
      RKNN_WHL="$(ls "$PACKAGES_DIR"/rknn_toolkit_lite*"$PY_TAG"*.whl 2>/dev/null | head -n 1)"
    fi
    if [ -n "$RKNN_WHL" ]; then
      "$PYTHON_BIN" -m pip install "$RKNN_WHL"
    fi
  fi

  if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    "$PYTHON_BIN" -m pip install -r "$SCRIPT_DIR/requirements.txt"
  fi
fi

"$PYTHON_BIN" - <<'EOF'
try:
    from rknnlite.api import RKNNLite  # type: ignore
    print("[setup] RKNNLite available.")
except Exception as exc:
    print("[setup] warning: RKNNLite import failed:", repr(exc))
EOF

echo "[setup] runtime environment ready: $VENV_DIR"
