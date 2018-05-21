Hill Climbing IRL
=========================

This repository is based on well thought [Physical Gradient Descent repository](https://github.com/chrisfosterelli/physical-gradient-descent) from [Chris Foster's blog post](https://fosterelli.co/executing-gradient-descent-on-the-earth).

Thought it was pretty neat, and wanted to see how other Hill climbing algorithms would fare with each other in a real world (literally) use case. While doing that, refactored the code for my better understanding.

Unfortunately, real world geography is not a good test case for gradient **descent** kind of algorithms since the aim of reaching ocean is very easy to do (just have bigger steps.). Meanwhile hill climbing or gradient ascent fits perfectly.

Luckly, this can be easily solved by turning the world inside out, thus turning it into a great problem for finding global/local minima or just reversing the loss / slope and changing the algorithms adapt for finding the global/local maxima. Inversing the world was the easiest, so I've done that.

In this repository, I've experimented with climbing the mountain my university sits on from a nearby seaside. It works pretty well and can see interesting paths from different algorithms (like mostly momentum based algorithms overshooting at first).

#### You can change the starting location from `params.json`

```json
{
    "_comment": "Starting coordinates",
    "center": {
        "lat": 40.953845,
        "lng": 29.095831
    },
    "_comment": "output path for the CSV files, if changed need to change visualizer.html too",
    "output": "outputs/",
    "_comment": "number of steps to take until convergence",
    "iters": 500,
    "_comment": "elevation map from SRTM Tile Grabber",
    "tif": "tifs/srtm_42_04.tif"
}
```
#### If you only want some of the algorithms visualized, or want a different color scheme, you can edit the `methods` dictionary in `run.py`

```python
"""
fun: function
  Function reference for the algorithm used, gets a custom RasterMap object, start coordinates and hyperparameters
color: string
  Color to use for visualization.
"""
methods = {
    'Gradient Descent': {'fun': gradient_descent, 'color': '#FF0000'},
    'Momentum': {'fun': gradient_descent_w_momentum, 'color': '#009933'},
    'NAG': {'fun': gradient_descent_w_nesterov, 'color': '#9900FF'},
    'Adagrad': {'fun': adagrad, 'color': '#0066FF'},
    'RMSprop': {'fun': RMSprop, 'color': '#000000'},
    'Adam': {'fun': adam, 'color': '#FFFF00'}
}
```

*If you want to add new optimizer algorithms, you can add function reference with a similar signature to methods dictionary.*

## Results:

<p align="center">
  <img src="https://raw.githubusercontent.com/umutto/Hill-Climbing-IRL/master/src/srtm_42_04.gif" alt="Hill Climbing From Bostanci"  height="600" width="600"/>  
  </br>
  <sup><i>Example output of common gradient descent algorithms.</i></sup>
</p>  


#### You can follow the original instructions from Chris Foster below, it should work for this repository as well.  
---


This is code for adapting the gradient descent algorithm to run on earth's 
actual geometry. You can read more about this in the attached [blog post].

## Running gradient descent

You'll need Python 3, and can install the dependencies with:

```bash
> virtualenv -p python3 env
> source env/bin/activate
> pip install -r requirements.txt
```

And run gradient descent like so:

```bash
(env)> python gradientdescent.py 47.801686 -123.709083 ~/Downloads/srtm_12_03/srtm_12_03.tif
```

A number of parameter tweaking options are supported:

```bash
usage: gradientdescent.py [-h] [--output OUTPUT] [--alpha ALPHA]
                          [--gamma GAMMA] [--iters ITERS]
                          lat lon tif
```

You can get TIF files from this [tile grabber]!

## Running the visualizer

The visualizer makes AJAX requests, so you will need to serve it from a web
server instead of just opening the HTML file on the filesystem. You can do that
easily with Python if you prefer, which will serve the local directory:

```bash
> python -m http.server
```

Then access http://localhost:8000 and the visualizer will start. You may have to
update the key used inside the HTML by editing the file to your own Google Maps
API key.

[tile grabber]: http://dwtkns.com/srtm/
[blog post]: https://fosterelli.co/executing-gradient-descent-on-the-earth

