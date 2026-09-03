# Estimating pi with random points

This is a small Monte Carlo experiment I made to see how closely random
sampling can approximate pi.

The idea is simple: scatter 10,000 random points across the square
`[-1, 1] × [-1, 1]` and count how many fall inside the unit circle. The
circle covers `pi / 4` of the square, so multiplying that fraction by four
gives us an estimate of pi.

## Try it yourself

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook monte_carlo_pi.ipynb
```

The notebook uses the fixed seed `42`, so you should get the same result each
time you run it.
