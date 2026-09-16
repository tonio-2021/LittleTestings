# Fractional knapsack

This is the version of knapsack where I can take part of an item. I sort the
items by value per unit of weight, take as much as fits, and stop when the bag
is full.

The example prints the order of the picks and compares the final value with
a small linear-programming solve. With a capacity of 4, it takes all of the
small item and two-thirds of the medium one. Both methods get a value of 22.

From the main folder:

```bash
python3 -m optimization.fractional_knapsack.knapsack
python3 -m unittest optimization.fractional_knapsack.test_knapsack
```

This greedy rule is for *fractional* knapsack. If whole items are required,
the same ordering does not always give the best answer.
