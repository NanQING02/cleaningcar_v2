from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / 'configs' / 'config.json'
LEGACY_CONFIG_PATH = ROOT / 'config.json'
CONFIG_PATH = DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else LEGACY_CONFIG_PATH
TEMPLATE_PATH = ROOT / 'web' / 'templates' / 'zone_editor.html'
RUN_SCRIPT = ROOT / 'run_zone_detect.py'


class FrameCache:
    def __init__(self):
        self.data = None
        self.size = (960, 540)
        self.key = None

    def clear(self):
        self.data = None
        self.key = None


FRAME_CACHE = FrameCache()


def set_config_path(path):
    global CONFIG_PATH
    CONFIG_PATH = Path(path).resolve()
    return CONFIG_PATH
