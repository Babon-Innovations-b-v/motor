# motor/solvers/

The sim body's numerical solvers, pure numpy and scipy: `least` (the bounded least squares the static
optimisation shares the forces with, projected Newton), `within_balls` (RRA's capped move, ADMM over a product of
balls), `smoothed` and `filter_matrix` (the path's low-pass, `CUTOFF` 6 Hz, Winter 2009) and `stencils` (the central
differences as matrices). Its test runs in the score venv against answers worked out by hand.

- #70: the fit's solve goes to mink and scipy's `least_squares` (no handwritten iterative solver left in the fit);
  `least` stays for the static optimisation and `within_balls` with RRA, which #70 leaves as it is.
