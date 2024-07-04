import cmap
from PIL import Image
# import matplotlib.cm as cm
import numpy as np

# test data
# np.random.seed(42)
# mask = np.random.randint(low=0, high=6, size=(32, 32))

# color palette - random colors for testing
# colormap = np.zeros((256, 4))
# colormap[:7, :3] = cm.jet(np.linspace(0, 1, 7))[:, :3]
# colormap[:7, 3] = np.linspace(0, 1, 7)

# img = Image.fromarray(mask.astype(np.uint8))
# img = img.convert('P')
# img.putpalette((colormap * 255).astype(np.uint8).flatten(), rawmode='RGBA')

# img.save('image.png')

def colormap_lut(name):
    cm = cmap.Colormap(name)
    return (cm.lut()[:,:3] * 255).astype(np.uint8)

def save_lut(filename, lut):
    lut.shape = (1,-1,3)
    img = Image.fromarray(lut, "RGB")
    img.save(filename)

if __name__ == '__main__':
    import sys
    argv = sys.argv
    argc = len(argv)
    name = argv[1] if argc > 1 else 'inferno'
    filename = argv[2] if argc > 2 else name + '.png'
    save_lut(filename, colormap_lut(name))
