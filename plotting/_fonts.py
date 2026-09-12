from pathlib import Path
from matplotlib import font_manager

def preferred_font():
    """Use the paper font when installed; do not redistribute proprietary fonts."""
    fonts = Path("C:/Windows/Fonts")
    for filename in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        path = fonts / filename
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
    for family in ("Times New Roman", "Liberation Serif", "DejaVu Serif"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            pass
    return "serif"
