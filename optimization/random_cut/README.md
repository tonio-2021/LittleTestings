# Trying random cuts

This is a small weighted Max-Cut experiment. Each vertex gets put on one of
two sides with a fair random choice, and an edge counts when its endpoints end
up on different sides.

The lecture calculation says the average cut should keep half of the total
edge weight. The script repeats the random choice 5,000 times on a four-node
example and also checks every possible cut. The exhaustive part is only there
as a check because it becomes too slow when the graph grows.

With the fixed example seed, the random average was `6.560`, close to half of
the total weight (`6.500`). The best random try had weight `10`, which was
also the exhaustive answer.

From the main folder:

```bash
python3 -m optimization.random_cut.random_cut
python3 -m unittest optimization.random_cut.test_random_cut
```
