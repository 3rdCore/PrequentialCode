import io

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PIL import Image


def fig2img(fig: Figure) -> Image.Image:
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img
