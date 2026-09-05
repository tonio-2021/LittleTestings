# Little Testings

This is a collection of small things I try while going through different
topics from university. Most of them are just short experiments to make the
ideas a bit more concrete.

## Estimating pi with random points

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

## Vertex cover from a maximal matching

This is a small implementation of the 2-approximation for vertex cover that
uses a maximal matching. It takes any remaining edge, adds both endpoints to
the cover, and then removes the edges that are already covered.

It does not always find the smallest possible cover, but it is quite simple
and the result is never more than twice as large as an optimal cover.

Run the example with:

```bash
python3 maximal_matching_vertex_cover.py
```

The tests only use Python's standard library:

```bash
python3 -m unittest test_maximal_matching_vertex_cover.py
```
