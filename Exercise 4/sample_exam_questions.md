# Sample Exam Questions: Explain / Correct–Fix / Order

Practice for the "explain, correct/fix, order provided snippets" part of the exam. Based on numerical optimization (GD, Newton, RMSProp, Adam) and non-convex objectives.

---

## Part A: Explain

### Q1. Explain what the following code does and what optimization method it implements.

```python
def step(self, point):
    gradient = self.compute_gradient(point)
    point = point - self.lr * gradient
    return point
```

**Expected type of answer:** Name the method, write the update rule in one sentence, and state what `lr` and `gradient` represent.

---

### Q2. Explain the purpose of `EPS` in this snippet and what could go wrong without it.

```python
point = point - self.lr * gradient / (np.sqrt(self.v) + EPS)
```

**Expected type of answer:** What is EPS used for (numerical stability), and what failure (e.g. division by zero or NaN) can occur if it is omitted.

---

### Q3. Explain why the following two lines are computed in this order, and what would be wrong if their order were swapped.

```python
self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
m_hat = self.m / (1 - self.beta_1 ** self.i)
```

**Expected type of answer:** What `m` and `m_hat` represent, why we first update `m` then use it in `m_hat`, and what goes wrong if we use `m_hat` before updating `m` (e.g. using stale momentum).

---

### Q4. This code is part of an optimizer step. What method is it, and in one sentence what role the Hessian plays in the update?

```python
gradient = self.compute_gradient(point)
hessian = self.compute_hessian(point)
direction = np.linalg.solve(hessian, gradient)
point = point - self.lr * direction
```

**Expected type of answer:** Newton (or Newton-type) method; Hessian scales the step (second-order information / curvature) so that the step size adapts per direction.

---

## Part B: Correct / Fix

### Q5. The following Newton step is discouraged for numerical reasons. Fix it without changing the mathematical update (same formula, more stable implementation).

```python
gradient = self.compute_gradient(point)
hessian = self.compute_hessian(point)
hessian_inv = np.linalg.inv(hessian)
point = point - self.lr * hessian_inv @ gradient
```

**Expected fix:** Use `np.linalg.solve(hessian, gradient)` instead of forming `hessian_inv` and multiplying (solves \(H d = g\) for \(d\), then `point = point - self.lr * direction`). Explain briefly why this is better (stability, cost, same result).

---

### Q6. The gradient for a 2D function is intended to be \(\nabla f = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}]\). Fix the bug in this gradient implementation.

```python
def gradient(point):
    x, y = point[0], point[1]
    dx = 2 * (x**2 + y - 11) * 2 * x + 2 * (x + y**2 - 7)
    dy = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * 2 * y
    grad = np.array([dy, dx])   # <-- bug here
    return grad
```

**Expected fix:** Use `grad = np.array([dx, dy])` so that the first component is the derivative w.r.t. \(x\) and the second w.r.t. \(y\). Optionally note that the order must match the parameter order \((x, y)\).

---

### Q7. In RMSProp, the update should use the current gradient and the updated \(v\). Fix the snippet so that the order of operations is correct.

```python
def step(self, point):
    self.v = self.beta_2 * self.v + (1 - self.beta_2) * gradient * gradient
    gradient = self.compute_gradient(point)
    point = point - self.lr * gradient / (np.sqrt(self.v) + EPS)
    return point
```

**Expected fix:** Compute `gradient = self.compute_gradient(point)` first, then update `self.v` using that `gradient`, then update `point`. Explain that we need the gradient at the current point before we can update \(v\) and then the parameters.

---

### Q8. The Hessian of a twice-differentiable scalar function must be symmetric: \(H_{ij} = H_{ji}\). The code below returns a 2×2 Hessian. Fix it so that the matrix is symmetric.

```python
h11 = 4 * (x**2 + y - 11) + 8 * x**2 + 2
h12 = 4 * x + 4 * y
h21 = 8 * x + 4 * y   # <-- inconsistent with h12
h22 = 4 * (x + y**2 - 7) + 8 * y**2 + 2
hessian = np.array([[h11, h12], [h21, h22]])
```

**Expected fix:** Set `h21 = h12` (or correct the formula for `h21` so that it equals the correct \(\partial^2 f / \partial y \partial x\)). State that for smooth \(f\), \(\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}\).

---

## Part C: Order

### Q9. The following lines implement one step of Adam, but they are in the wrong order. Number them in the correct execution order (1, 2, 3, …).

- ( ) `point = point - self.lr * m_hat / (np.sqrt(v_hat) + EPS)`
- ( ) `self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient`
- ( ) `gradient = self.compute_gradient(point)`
- ( ) `v_hat = self.v / (1 - self.beta_2 ** self.i)`
- ( ) `self.i += 1`
- ( ) `m_hat = self.m / (1 - self.beta_1 ** self.i)`
- ( ) `self.v = self.beta_2 * self.v + (1 - self.beta_2) * gradient * gradient`
- ( ) `return point`

**Correct order:**  
1. `self.i += 1`  
2. `gradient = self.compute_gradient(point)`  
3. `self.m = ...`  
4. `self.v = ...`  
5. `m_hat = ...`  
6. `v_hat = ...`  
7. `point = point - ...`  
8. `return point`

(Reasoning: increment step counter, get gradient, update moments, bias-correct, then apply update and return.)

---

### Q10. A training loop for an optimizer is given as a list of steps in random order. Put them in the correct order.

- ( ) Evaluate the objective \(f(\theta)\) at the new point (e.g. for logging or stopping).
- ( ) Initialize the parameter vector \(\theta\) (e.g. starting point).
- ( ) Call the optimizer’s `step` with the current \(\theta\) to get the next \(\theta\).
- ( ) Compute the gradient of the objective at the current \(\theta\) (inside `step` or before).
- ( ) Repeat until convergence or max iterations.
- ( ) Check stopping criterion (e.g. \(f < \varepsilon\) or max iter reached).

**Correct order (conceptual):**  
1. Initialize \(\theta\).  
2. Repeat: (a) gradient is computed inside `step` at current \(\theta\); (b) call `step` to get new \(\theta\); (c) evaluate \(f\) at new \(\theta\); (d) check stopping; exit or continue.

So: Initialize → [ (gradient computed in step → step → evaluate f → check stop) ].

---

### Q11. For one step of Gradient Descent, order these operations correctly.

- ( ) `point = point - self.lr * gradient`
- ( ) `gradient = self.compute_gradient(point)`
- ( ) `return point`

**Correct order:** 1. Compute gradient at current point. 2. Update point. 3. Return point.  
(We must use the gradient at the *current* point before updating; otherwise we would be using a stale gradient.)

---

## Part D: Short conceptual (explain in one or two sentences)

### Q12. Why do we use numerical optimization (e.g. gradient descent) instead of a closed-form solution for many machine learning objectives?

**Expected idea:** Closed form requires solving \(\nabla f = 0\) analytically; for non-convex or non-linear objectives there is no tractable closed form, so we iterate with gradient-based updates.

---

### Q13. Why is Newton’s method risky for non-convex objectives?

**Expected idea:** The Hessian can have negative eigenvalues (saddle or local maximum), so \(H^{-1} \nabla f\) may not point downhill and the step can increase the objective or be ill-defined.

---

### Q14. What is the purpose of bias correction in Adam (the \(\hat{m}\) and \(\hat{v}\) terms)?

**Expected idea:** Early in training, the running averages \(m\) and \(v\) are biased toward zero; dividing by \(1 - \beta^t\) corrects this so that early steps are not unnecessarily small.

---

## Answer key (brief)

| Q  | Type    | Main point |
|----|---------|------------|
| 1  | Explain | Gradient descent; \(\theta \leftarrow \theta - \alpha \nabla f\); lr = step size, gradient = \(\nabla f\). |
| 2  | Explain | EPS avoids division by zero in denominator \(\sqrt{v}+\varepsilon\); without it, NaNs or infs when \(v=0\). |
| 3  | Explain | First update \(m\), then use it in \(m_hat\); if swapped, we’d use old \(m\) and wrong bias correction. |
| 4  | Explain | Newton; Hessian scales the step (curvature / second-order). |
| 5  | Fix     | Use `np.linalg.solve(hessian, gradient)` and then `point - lr * direction`; more stable, same math. |
| 6  | Fix     | `grad = np.array([dx, dy])` so component order matches \((x, y)\). |
| 7  | Fix     | Compute `gradient` first, then update `self.v`, then update `point`. |
| 8  | Fix     | Set `h21 = h12` (or fix formula) so Hessian is symmetric. |
| 9  | Order   | i+=1 → gradient → m → v → m_hat → v_hat → point update → return. |
| 10 | Order   | Init θ → loop: step (uses gradient) → evaluate f → check stop. |
| 11 | Order   | gradient → update point → return. |
| 12 | Concept | No closed form for non-convex/non-linear; we iterate. |
| 13 | Concept | H not positive definite → Newton step can ascend or be undefined. |
| 14 | Concept | Correct bias in m and v so early steps are not too small. |

---

*Use this document to practice under exam conditions: cover the answers, do Explain / Fix / Order in your head or on paper, then check.*
