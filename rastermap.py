import numpy as np
import rasterio


class RasterMap(object):
    def __init__(self, tif_file):
        self.src = rasterio.open(tif_file)

        # not interested in values smaller than 0
        band = self.src.read(1)
        band[band <= 0] = 0

        self.band = band
        self.max_val = np.max(band)

    def get_elevation(self, lat, lon):
        vals = self.src.index(lon, lat)
        return self.band[vals]

    def get_cost(self, lat, lon):
        # purposefuly inversing the terrain for hill climb / gradient ascent
        # inversing terrain is easier than working with inversing the gradient
        # or changing algorithms individually.
        return self.get_elevation(lat, lon) * -1 + self.max_val
