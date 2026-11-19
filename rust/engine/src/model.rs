//! CPU reference math for the layers: RMSNorm, SiLU-and-mul, RoPE, causal attention, linear, sampler.
//!
//! Checked against `rust/record/record_cpu_model.py`, which uses plain torch and not flash-attn.

pub fn parse_f32(text: &str) -> f32 {
    text.parse::<f32>().unwrap_or(f32::NAN)
}

pub fn rms_norm(x: &[f32], weight: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mut var = 0.0_f32;
        for v in row {
            var += v * v;
        }
        var /= cols as f32;
        let scale = (var + eps).sqrt().recip();
        for c in 0..cols {
            out[r * cols + c] = row[c] * scale * weight[c];
        }
    }
    out
}

pub fn silu_and_mul(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let half = cols / 2;
    let mut out = vec![0.0; rows * half];
    for r in 0..rows {
        for c in 0..half {
            let a = x[r * cols + c];
            let b = x[r * cols + half + c];
            let sig = 1.0 / (1.0 + (-a).exp());
            out[r * half + c] = a * sig * b;
        }
    }
    out
}

pub fn apply_rotary(x: &[f32], cos: &[f32], sin: &[f32], shape: &[usize]) -> Vec<f32> {
    let n = shape.iter().product();
    let last = *shape.last().unwrap();
    let half = last / 2;
    let mut out = vec![0.0; n];
    let outer = n / last;
    for i in 0..outer {
        for c in 0..half {
            let x1 = x[i * last + c];
            let x2 = x[i * last + half + c];
            let co = cos[i * half + c];
            let sn = sin[i * half + c];
            out[i * last + c] = x1 * co - x2 * sn;
            out[i * last + half + c] = x2 * co + x1 * sn;
        }
    }
    out
}

pub fn rope_cache(head_dim: usize, max_pos: usize, base: f32) -> Vec<f32> {
    let half = head_dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| {
            let exp = (2 * i) as f32 / head_dim as f32;
            base.powf(exp).recip()
        })
        .collect();
    let mut out = vec![0.0; max_pos * head_dim];
    for t in 0..max_pos {
        for i in 0..half {
            let freq = (t as f32) * inv[i];
            out[t * head_dim + i] = freq.cos();
            out[t * head_dim + half + i] = freq.sin();
        }
    }
    out
}

pub fn causal_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    h: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; t * h * d];
    for head in 0..h {
        let mut scores = vec![0.0_f32; t * t];
        for qi in 0..t {
            for ki in 0..t {
                if ki > qi {
                    scores[qi * t + ki] = f32::NEG_INFINITY;
                    continue;
                }
                let mut dot = 0.0_f32;
                for di in 0..d {
                    dot += q[(qi * h + head) * d + di] * k[(ki * h + head) * d + di];
                }
                scores[qi * t + ki] = dot * scale;
            }
        }
        for qi in 0..t {
            let row = &scores[qi * t..(qi + 1) * t];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            let mut probs = vec![0.0_f32; t];
            for (i, s) in row.iter().enumerate() {
                let e = (*s - max).exp();
                probs[i] = e;
                sum += e;
            }
            for p in &mut probs {
                *p /= sum;
            }
            for di in 0..d {
                let mut acc = 0.0_f32;
                for ki in 0..t {
                    acc += probs[ki] * v[(ki * h + head) * d + di];
                }
                out[(qi * h + head) * d + di] = acc;
            }
        }
    }
    out
}

pub fn linear(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols_out];
    for r in 0..rows {
        for o in 0..cols_out {
            let mut acc = bias.map(|b| b[o]).unwrap_or(0.0);
            for c in 0..cols_in {
                acc += x[r * cols_in + c] * weight[o * cols_in + c];
            }
            out[r * cols_out + o] = acc;
        }
    }
    out
}

pub fn sample(
    logits: &[f32],
    noise: &[f32],
    rows: usize,
    cols: usize,
    temperature: f32,
) -> Vec<i64> {
    let mut tokens = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &logits[r * cols..(r + 1) * cols];
        let mut scaled = vec![0.0_f32; cols];
        let mut max = f32::NEG_INFINITY;
        for (i, v) in row.iter().enumerate() {
            scaled[i] = v / temperature;
            max = max.max(scaled[i]);
        }
        let mut sum = 0.0_f32;
        for v in &mut scaled {
            *v = (*v - max).exp();
            sum += *v;
        }
        let mut best_i = 0;
        let mut best = f32::NEG_INFINITY;
        for i in 0..cols {
            let p = scaled[i] / sum;
            let n = noise[r * cols + i].max(1e-10);
            let score = p / n;
            if score > best {
                best = score;
                best_i = i;
            }
        }
        tokens.push(best_i as i64);
    }
    tokens
}

pub fn close_enough(got: &[f32], expect: &[f32]) -> Result<(), String> {
    if got.len() != expect.len() {
        return Err(format!("len {} vs {}", got.len(), expect.len()));
    }
    let mut worst = 0.0_f32;
    let mut at = 0;
    for (i, (g, e)) in got.iter().zip(expect).enumerate() {
        let tol = 1e-5 + 1e-5 * e.abs();
        let err = (g - e).abs();
        if err > worst {
            worst = err;
            at = i;
        }
        if err > tol {
            return Err(format!(
                "index {i}: got {g} expected {e} abs {err} tol {tol}"
            ));
        }
    }
    let _ = (worst, at);
    Ok(())
}
