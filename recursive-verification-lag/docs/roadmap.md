# Research Roadmap

## Highest-value external validation

Run the same mechanism tests with a pretrained small code model.

A minimal study should use:

1. 50–200 finite-domain coding tasks;
2. a fixed bank of sampled candidate programs per task;
3. public tests or a learned public proxy;
4. stronger hidden/exhaustive tests as trusted semantic reward;
5. independent sweeps over selection pressure, verifier refresh cadence, and trusted-data budget;
6. a richer-verifier control;
7. both Best-of-N and soft/exponential selection;
8. pre-registered inclusion criteria and per-task results.

## Key falsification test

If increasing trusted-data budget systematically moves the semantic failure boundary in a fixed verifier class, the current misspecification-threshold interpretation should be weakened.

If richer verifier representation removes the boundary while more data mainly sharpen it, the current mechanism story is strengthened.

## Theoretical target

The next worthwhile theorem is not another divergence definition. It is an observable **adaptive refresh frontier** based on the residual verification geometry after past trusted information has been incorporated.

A useful result would characterize when to refresh without oracle knowledge of the true gain.

If this cannot be done without artificial assumptions or direct access to the true margin, theory expansion should stop and the existing bridge theorem should remain the endpoint.
