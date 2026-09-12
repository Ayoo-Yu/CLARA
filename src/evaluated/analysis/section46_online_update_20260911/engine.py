"""Incremental equivalent of the existing local evidence estimator."""
import numpy as np
from numba import njit


@njit(cache=True)
def add_rows(state_ids, days, facts, start, end, mapping, counts, guard_counts,
             sums, day_counts, day_sums, nonempty_days, day_mean_sums, day_mean_squares):
    # facts: cost, covered, historical TUWR, historical ARD, all six actions.
    for i in range(start, end):
        s = state_ids[i]
        day = days[i]
        warm = np.isfinite(facts[i, 0, 2])
        for level in range(mapping.shape[1]):
            g = mapping[s, level]
            counts[g] += 1
            if warm:
                guard_counts[g] += 1
            old_n = day_counts[g, day]
            if old_n == 0:
                nonempty_days[g] += 1
            day_counts[g, day] += 1
            for a in range(6):
                cost = facts[i, a, 0]
                sums[g, a, 0] += cost
                sums[g, a, 1] += facts[i, a, 1]
                if warm:
                    sums[g, a, 2] += facts[i, a, 2]
                    sums[g, a, 3] += facts[i, a, 3]
                    sums[g, a, 4] += facts[i, a, 1]
                day_sums[g, day, a] += cost


@njit(cache=True)
def evidence(ids, mapping, nmin, nu, cold, counts, guard_counts, sums,
             nonempty_days, day_counts, day_sums):
    # Columns: raw, parent, shrunk, SE, score, coverage, TUWR, ARD.
    out = np.empty((len(ids), 6, 8))
    levels = np.empty(len(ids), np.int64)
    for row in range(len(ids)):
        s = ids[row]
        chosen = -1
        for level in range(mapping.shape[1]):
            if counts[mapping[s, level]] >= nmin[s]:
                chosen = level
                break
        assert chosen >= 0
        levels[row] = chosen
        g = mapping[s, chosen]
        root = mapping[s, -1]
        for a in range(6):
            parent = sums[root, a, 0] / counts[root]
            raw = parent
            prior = parent
            shrunk = parent
            for level in range(mapping.shape[1] - 2, chosen - 1, -1):
                gl = mapping[s, level]
                raw = sums[gl, a, 0] / counts[gl]
                prior = parent
                shrunk = (sums[gl, a, 0] + nu[s] * parent) / (counts[gl] + nu[s])
                parent = shrunk
            nd = nonempty_days[g]
            se = 0.0
            if nd > 1:
                # Welford over actual daily means avoids cancellation near zero SE.
                mean = 0.0
                residual = 0.0
                seen = 0
                for day in range(day_counts.shape[1]):
                    dn = day_counts[g, day]
                    if dn:
                        value = day_sums[g, day, a] / dn
                        seen += 1
                        delta = value - mean
                        mean += delta / seen
                        residual += delta * (value - mean)
                se = np.sqrt(max(0.0, residual) / ((nd - 1) * nd))
            cov = sums[g, a, 1] / counts[g] if cold[s] else (sums[g, a, 4] / guard_counts[g] if guard_counts[g] else np.nan)
            tw = sums[g, a, 2] / guard_counts[g] if guard_counts[g] else np.nan
            ard = sums[g, a, 3] / guard_counts[g] if guard_counts[g] else np.nan
            out[row, a, 0] = raw
            out[row, a, 1] = prior
            out[row, a, 2] = shrunk
            out[row, a, 3] = se
            out[row, a, 4] = shrunk + se
            out[row, a, 5] = cov
            out[row, a, 6] = tw
            out[row, a, 7] = ard
    return out, levels


@njit(cache=True)
def choose_one(e, c, cold, eps, tau):
    ok = np.empty(6, np.bool_)
    violation = np.empty(6)
    for a in range(6):
        ok[a] = e[a, 5] >= c - eps and (cold or (np.isfinite(e[a, 6]) and e[a, 6] <= tau))
        cv = max(0.0, c - eps - e[a, 5]) / eps
        tw = 0.0 if cold else (max(0.0, e[a, 6] - tau) / tau if np.isfinite(e[a, 6]) else np.inf)
        violation[a] = max(cv, tw)
    empty = not np.any(ok)
    if empty:
        v = np.min(violation)
        for a in range(6):
            ok[a] = violation[a] <= v + 1e-12
    best = np.inf
    for a in range(6):
        if ok[a]:
            best = min(best, e[a, 4])
    for a in range(6):
        ok[a] = ok[a] and e[a, 4] <= best + 1e-12
    best = -np.inf
    for a in range(6):
        if ok[a]:
            best = max(best, e[a, 5])
    for a in range(6):
        ok[a] = ok[a] and e[a, 5] >= best - 1e-12
    for field in (6, 7):
        finite = 0
        total = 0
        best = np.inf
        for a in range(6):
            if ok[a]:
                total += 1
                if np.isfinite(e[a, field]):
                    finite += 1
                    best = min(best, e[a, field])
        assert finite == 0 or finite == total
        if finite:
            for a in range(6):
                ok[a] = ok[a] and np.isfinite(e[a, field]) and e[a, field] <= best + 1e-12
    assert np.any(ok)
    return np.argmax(ok), empty


@njit(cache=True)
def choose_many(ids, estimates, coverage, cold, eps, tau):
    choices = np.empty(len(ids), np.int8)
    empty = np.empty(len(ids), np.bool_)
    for i in range(len(ids)):
        choices[i], empty[i] = choose_one(estimates[i], coverage[ids[i]], cold[ids[i]], eps, tau)
    return choices, empty


@njit(cache=True)
def run_batches(issue, label, available, state_ids, days, facts, feedback_order,
                mapping, nmin, nu, cold, coverage, frozen, eps, tau, stats, stop=-1):
    counts, guard_counts, sums, day_counts, day_sums, nonempty_days, farma, farmas = stats
    n = len(issue) if stop < 0 else stop
    # Original chronological order. 0 frozen, 1 joint, 2 cost only, 3 reliability only.
    selected = np.empty((n, 4), np.int8)
    # Online six-candidate estimates for paired cost/coverage error diagnostics.
    predictions = np.empty((n, 6, 3))  # shrunk cost, coverage, historical TUWR
    trace = np.empty((n, 4))  # level, empty joint, empty frozen, feedback count
    pointer = 0
    start = 0
    while start < n:
        end = start + 1
        while end < n and issue[end] == issue[start]:
            end += 1
        current = issue[start]
        oldptr = pointer
        while pointer < len(feedback_order) and available[feedback_order[pointer]] <= current:
            row = feedback_order[pointer]
            assert label[row] < current and issue[row] < current
            pointer += 1
        if pointer > oldptr:
            indices = feedback_order[oldptr:pointer]
            add_rows(state_ids[indices], days[indices], facts[indices], 0, len(indices), mapping,
                     counts, guard_counts, sums, day_counts, day_sums, nonempty_days, farma, farmas)
        ids = state_ids[start:end]
        ev, levels = evidence(ids, mapping, nmin, nu, cold, counts, guard_counts, sums, nonempty_days, day_counts, day_sums)
        for j in range(end - start):
            row = start + j
            s = ids[j]
            old = frozen[s]
            fresh = ev[j]
            selected[row, 0], eold = choose_one(old, coverage[s], cold[s], eps, tau)
            selected[row, 1], enew = choose_one(fresh, coverage[s], cold[s], eps, tau)
            cost = fresh.copy()
            cost[:, 5:] = old[:, 5:]
            rel = old.copy()
            rel[:, 5:] = fresh[:, 5:]
            selected[row, 2], _ = choose_one(cost, coverage[s], cold[s], eps, tau)
            selected[row, 3], _ = choose_one(rel, coverage[s], cold[s], eps, tau)
            predictions[row, :, 0] = fresh[:, 2]
            predictions[row, :, 1] = fresh[:, 5]
            predictions[row, :, 2] = fresh[:, 6]
            trace[row, 0] = levels[j]
            trace[row, 1] = enew
            trace[row, 2] = eold
            trace[row, 3] = pointer
        start = end
    return selected, predictions, trace, pointer
