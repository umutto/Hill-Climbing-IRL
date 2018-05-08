Hill Climbing IRL
=========================

This repository is based on well thought [Physical Gradient Descent repository](https://github.com/chrisfosterelli/physical-gradient-descent) from [Chris Foster's blog post](https://fosterelli.co/executing-gradient-descent-on-the-earth).

Thought it was pretty neat, and wanted to see how other Hill climbing algorithms would fare with each other in a real world (literally) use case. While doing that, refactored the code for my better understanding.

Unfortunately, real world geography is not a good test case for gradient **descent** kind of algorithms since the aim of reaching ocean is very easy to do (just have bigger steps.). Meanwhile hill climbing or gradient ascent fits perfectly.

Luckly, this can be easily solved by turning the world inside out, thus turning it into a great problem for finding global/local minima or just reversing the loss / slope and changing the algorithms adapt for finding the global/local maxima. Inversing the world was the easiest, so I've done that.

In this repository, I've experimented with climbing the mountain my university sits on from a nearby seaside. It works pretty well and can see interesting paths from different algorithms (like mostly momentum based algorithms overshooting at first).

To run the code, below you can find the original readme by Chris Foster. All of it is applicable for this repository as well.

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

