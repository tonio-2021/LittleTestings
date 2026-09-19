# Cliques in a complement graph

This is a small follow-up to the independent-set example. A clique has an
edge between every pair of its vertices. When the graph is complemented,
those edges disappear, so the same labelled vertices become an independent
set.

The script builds the complement by checking every pair of vertices. It also
finds a largest clique by trying every subset. That is fine for the five-node
example, but definitely not meant for larger graphs.

From the main folder:

```bash
python3 -m graph_algorithms.clique_complement.checker
python3 -m unittest graph_algorithms.clique_complement.test_checker
```

In the example, `a`, `b`, and `c` make a triangle in the original graph. None
of the three edges is present in the complement, so the same three names form
an independent set there.
