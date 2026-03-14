# Benchmark CLI Reference

Run `python canon_benchmark.py` with these flags:

| Flag | Effect |
|------|--------|
| `--quick` | Small problems only (14 problems, ~30s) |
| `--json` | JSON to stdout (timing to stderr) |
| `--backend X` | Single backend: `CPP`, `SCIPY`, or `COO` |
| `--all-backends` | Run CPP + SCIPY + COO, print comparison |
| `--trace` | cProfile per problem (last iteration) |
| `--verify` | Compare A,b,c matrices against SCIPY oracle |
| `--quiet` | Suppress stderr |

## Common Recipes

```bash
# Quick smoke test with verification
python canon_benchmark.py --quick --json --verify

# Full benchmark, all backends, with traces
python canon_benchmark.py --all-backends --json --trace --verify

# Single backend quick check
python canon_benchmark.py --quick --json --verify --backend COO
```

## JSON Output Structure

Single backend:
```json
{"timestamp": "...", "backends": ["COO"], "geomean_ms": 1.23, "total_ms": 45.6,
 "problems": [{"name": "...", "mean_ms": 1.0, "std_ms": 0.1, "min_ms": 0.9,
   "verification": {"pass": true, "max_err": 1e-12, "mismatches": []},
   "trace": [{"func": "...", "tottime_ms": 0.5, "cumtime_ms": 0.8, "calls": 10, "pct": 50.0}]}]}
```

Multi-backend (`--all-backends`):
```json
{"timestamp": "...", "best_backend": "CPP", "geomean_ms": 1.23,
 "backends": {"CPP": {...}, "SCIPY": {...}, "COO": {...}}}
```

## Interpreting Traces

- `tottime_ms` = self time (excludes callees) — **optimize this**
- `cumtime_ms` = inclusive time — shows call tree impact
- `pct` = % of total problem time
- Focus on top 3 entries by `tottime_ms`
