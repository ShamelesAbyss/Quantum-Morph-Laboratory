# COLOSSUS recorded task outputs

These are raw decoded model outputs on the frozen experiment 007 inputs,
not recommended implementations. Incorrect answers remain unchanged. See the
[protocol and oracle](COLOSSUS_REPRODUCTION.md) and
[research report](PROJECT_COLOSSUS.md) for inputs, objective checks, timing,
source identities and limitations. A single fenced block may be extracted for
functional testing; literal format compliance is separately recorded. No
original BF16 task baseline is measured. Fourteen custom tasks do not establish
general coding capability or retained intelligence.

## INT3 GPTQ

11/14 passed the predeclared checks; only 1/14 outputs had literal format
compliance. All 14 stopped on EOS, with no output-limit truncation. This uses
the measured 1024-token context, greedy decoding and seed 42.

### rust-1 — fail

Category: rust_generation. Output SHA-256: `354af30ac09d79da19f9f556fc126e3cceae4f9e49610c4734d8ed0f5567a167`.

````text
```rust
pub fn runs(xs: &[i32]) -> Vec<(i32, usize)> {
    let mut result = Vec::new();
    let mut count = 0;
    if let Some(&first) = xs.first() {
        for &x in xs {
            if x == first {
                count += 1;
            } else {
                result.push((first, count));
                count = 1;
            }
        }
        result.push((first, count));
    }
    result
}
```
````

Failure: The generated run encoding compiles but fails the mixed-run unit assertion; its initial run value is never updated.

### rust-2 — pass

Category: rust_generation. Output SHA-256: `f3fc5c42311a60b7bdec0fe88daad755c607a2431fdc2b47c0c8490f2fb2dea3`.

````text
```rust
pub fn first_unique(s: &str) -> Option<char> {
    let mut char_count = std::collections::HashMap::new();
    let mut char_order = Vec::new();

    for c in s.chars() {
        *char_count.entry(c).or_insert(0) += 1;
        char_order.push(c);
    }

    for &c in &char_order {
        if let Some(&count) = char_count.get(&c) {
            if count == 1 {
                return Some(c);
            }
        }
    }

    None
}
```
````

### python-1 — pass

Category: python_generation. Output SHA-256: `ec857be0739dae41a37f5605d9ceccda364a39a0a0ac32b6a57548e5f7b5fb80`.

````text
```python
def merge_intervals(intervals):
    if not intervals:
        return []

    # Sort intervals by their start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    merged = [sorted_intervals[0]]

    for current in sorted_intervals[1:]:
        last_merged = merged[-1]
        if current[0] <= last_merged[1] + 1:  # Check if intervals touch or overlap
            merged[-1] = (last_merged[0], max(last_merged[1], current[1]))
        else:
            merged.append(current)

    return merged
```
````

### python-2 — pass

Category: python_generation. Output SHA-256: `96d500bde5159723100d9773705397a7f5c49e1cdd034189856f1775b49a0e92`.

````text
```python
def rotate(xs, k):
    if not xs:
        return []
    n = len(xs)
    k = k % n  # Handle negative k by converting it to equivalent positive rotation
    return xs[-k:] + xs[:-k]
```
````

### js-1 — pass

Category: javascript_generation. Output SHA-256: `31e60cc18654b019a3eb9ae1a66542b1f8e4d149090fe0c08d30eeb942fabeae`.

````text
```javascript
function uniqueStable(xs) {
    const seen = new Set();
    const result = [];

    for (const item of xs) {
        if (!seen.has(item)) {
            seen.add(item);
            result.push(item);
        }
    }

    return result;
}
```
````

### js-2 — pass

Category: javascript_generation. Output SHA-256: `dc9800a6e601f10f084db51e2770a139023f3649dabbbb46df9f898354ace04f`.

````text
```javascript
function lowerBound(xs, target) {
    let left = 0;
    let right = xs.length;

    while (left < right) {
        const mid = Math.floor((left + right) / 2);
        if (xs[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}
```
````

### debug-1 — pass

Category: debugging. Output SHA-256: `58136eb5dfb571dfced94c93bd7198db5ecc2d17062240444f8a862e94e2e189`.

````text
```rust
pub fn last_index(xs: &[i32], value: i32) -> Option<usize> {
    for i in (0..xs.len()).rev() {
        if xs[i] == value {
            return Some(i);
        }
    }
    None
}
```
````

### debug-2 — pass

Category: debugging. Output SHA-256: `0a7a79860f593b37b105493c0b29bdc296853630d307d7fead52b148366673d0`.

````text
```python
def append_item(x, items=None):
    if items is None:
        items = []
    items.append(x)
    return items
```
````

### reason-1 — fail

Category: algorithmic_reasoning. Output SHA-256: `074b9e42219fb6c2fef7fcf20fa3594bf81e31f3b2d06811e93f4448458ca613`.

````text
```json
{
  "path": ["A", "C", "B", "D"],
  "weight": 7
}
```
````

Failure: The path is correct but its reported weight is 7; the specified edge weights sum to 4.

### reason-2 — pass

Category: algorithmic_reasoning. Output SHA-256: `3f1006556b695a0b4f998b94c8a0170a99b8d3b4738825a9ed10ac8276e34d60`.

````text
```json
{
  "distance": 3,
  "dp_time": "O(m*n)"
}
```
````

### explain-1 — pass

Category: code_explanation. Output SHA-256: `bcb497149ad3848f7e1d29af46c1c52a6426cb6e80db45434798fbfe252363f9`.

````text
```json
{
  "value": [[0, 1], [0, 1], [0, 1]],
  "shared": true,
  "cause": "list repetition aliases the inner list"
}
```
````

### explain-2 — pass

Category: code_explanation. Output SHA-256: `ca9de46eba50f3ccfd389f3a154c41f7e6ff355630fef589209b900549b59ff5`.

````text
```json
{
  "a": [1, 2, 3],
  "b": [2, 4],
  "mutates": false
}
```
````

### instruction-1 — fail

Category: instruction_following. Output SHA-256: `893e329f36a8112983fa6afadd85a0c1ab16929be3c41a5fa84f6c67da6ddfa0`.

````text
```json
{
  "names": ["JavaScript", "Python", "Rust"],
  "count": 3
}
```
````

Failure: Values are correct after fence extraction, but the explicit JSON-only/no-Markdown requirement is violated.

### instruction-2 — pass

Category: instruction_following. Output SHA-256: `6506c1cafceb6e44967d78d9dfa110cc92864d83612e41f010c7119b7ad6c32e`.

````text
ALPHA
56
omega
````


