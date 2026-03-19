import sympy as sp

# Symbols (assume positive integers where needed)
r, t = sp.symbols('r t', integer=True, positive=True)

def C(n, k):
        return sp.binomial(n, k)

# n = r*t, where r = m-1 in Turán(K_m) setting
n = r*t

# d_m(n) for n = r*t: balanced r-partite with parts all size t
dm_n = C(n, 2) - r * C(t, 2)
dm_n_simplified = sp.simplify(dm_n)

# d_m(n-1) for n-1 = r*t - 1: parts (t-1, t, ..., t)
dm_n_minus_1 = C(n - 1, 2) - (C(t - 1, 2) + (r - 1) * C(t, 2))
dm_n_minus_1_simplified = sp.simplify(dm_n_minus_1)

# d_m^f(n, n-1) = n/(n-2) * d_m(n-1)
dm_f = sp.simplify(n / (n - 2) * dm_n_minus_1_simplified)

print("d_m(n) =", dm_n_simplified)
print("d_m(n-1) =", dm_n_minus_1_simplified)
print("d_m^f(n,n-1) =", dm_f)
print("d_m^f(n,n-1) - d_m(n) simplifies to:", sp.simplify(dm_f - dm_n_simplified))
