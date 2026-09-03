# LittleTestings

A tiny, beginner-friendly Monte Carlo simulation that estimates pi.

The notebook generates 10,000 random points in a square. The share that lands
inside the square's unit circle approximates the circle-to-square area ratio,
so multiplying that share by four gives an estimate of pi.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook monte_carlo_pi.ipynb
```

The random seed is fixed at `42`, making the result reproducible.
