#pragma once
// CPU double-precision reference implementations (independent of the GPU code) used by the op tests and the e2e test.
#include <cmath>
#include <vector>
typedef std::vector<double> D;

static double gelu_d(double x) { return 0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))); }
static double gelu_grad_d(double x) {
    double t = tanh(0.7978845608028654 * (x + 0.044715 * x * x * x));
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * 0.7978845608028654 * (1 + 3 * 0.044715 * x * x);
}
// C[M,N] = op(A) op(B); A is [M,K] (or [K,M] if ta), B is [K,N] (or [N,K] if tb), all row-major dense.
static D mm_d(const D& A, const D& B, int M, int N, int K, bool ta, bool tb) {
    D C((size_t)M * N, 0.0);
    for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) {
        double a = ta ? A[(size_t)k * M + m] : A[(size_t)m * K + k];
        for (int n = 0; n < N; n++) C[(size_t)m * N + n] += a * (tb ? B[(size_t)n * K + k] : B[(size_t)k * N + n]);
    }
    return C;
}
// norm forward: ln (rms=false) or rms. Returns out; fills mean, rstd.
static D norm_fwd_d(const D& x, const D& g, const D& b, int R, int d, double eps, bool rms, D& mean, D& rstd) {
    D o((size_t)R * d); mean.assign(R, 0); rstd.assign(R, 0);
    for (int r = 0; r < R; r++) {
        double mu = 0, v = 0;
        if (!rms) { for (int c = 0; c < d; c++) mu += x[r * d + c]; mu /= d; }
        for (int c = 0; c < d; c++) v += (x[r * d + c] - mu) * (x[r * d + c] - mu);
        v /= d; double rs = 1 / sqrt(v + eps); mean[r] = mu; rstd[r] = rs;
        for (int c = 0; c < d; c++) o[r * d + c] = (x[r * d + c] - mu) * rs * g[c] + (rms || b.empty() ? 0 : b[c]);
    }
    return o;
}
// norm backward: dx, dgamma, dbeta (recomputes statistics from x).
static void norm_bwd_d(const D& x, const D& dy, const D& g, int R, int d, double eps, bool rms, D& dx, D& dg, D& db) {
    D mean, rstd, dummy; norm_fwd_d(x, g, D(), R, d, eps, rms, mean, rstd);
    dx.assign((size_t)R * d, 0); dg.assign(d, 0); db.assign(d, 0);
    for (int r = 0; r < R; r++) {
        double s1 = 0, s2 = 0;
        for (int c = 0; c < d; c++) { double xh = (x[r * d + c] - mean[r]) * rstd[r], gc = dy[r * d + c] * g[c]; s1 += gc; s2 += gc * xh; dg[c] += dy[r * d + c] * xh; db[c] += dy[r * d + c]; }
        s1 /= d; s2 /= d;
        for (int c = 0; c < d; c++) { double xh = (x[r * d + c] - mean[r]) * rstd[r], gc = dy[r * d + c] * g[c]; dx[r * d + c] = rstd[r] * (gc - (rms ? 0 : s1) - xh * s2); }
    }
}
// Causal GQA attention. q [B*T, H*hd], k,v [B*T, Hkv*hd]. Forward fills o, lse[(b*H+h)*T+t]. If dO is given also dq,dk,dv.
static void attn_d(int B, int T, int H, int Hkv, int hd, double scale, const D& q, const D& k, const D& v, D& o, D& lse,
                   const D* dO, D* dq, D* dk, D* dv) {
    int G = H / Hkv;
    o.assign((size_t)B * T * H * hd, 0); lse.assign((size_t)B * H * T, 0);
    if (dO) { dq->assign(q.size(), 0); dk->assign(k.size(), 0); dv->assign(v.size(), 0); }
    for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
        int kvh = h / G;
        for (int t = 0; t < T; t++) {
            D s(t + 1); double mx = -1e300;
            for (int j = 0; j <= t; j++) {
                double a = 0; for (int d = 0; d < hd; d++) a += q[((size_t)(b * T + t)) * H * hd + h * hd + d] * k[((size_t)(b * T + j)) * Hkv * hd + kvh * hd + d];
                s[j] = a * scale; mx = std::max(mx, s[j]);
            }
            double l = 0; for (int j = 0; j <= t; j++) { s[j] = exp(s[j] - mx); l += s[j]; }
            lse[((size_t)b * H + h) * T + t] = mx + log(l);
            for (int j = 0; j <= t; j++) { s[j] /= l; for (int d = 0; d < hd; d++) o[((size_t)(b * T + t)) * H * hd + h * hd + d] += s[j] * v[((size_t)(b * T + j)) * Hkv * hd + kvh * hd + d]; }
            if (!dO) continue;
            D dp(t + 1); double delta = 0;
            for (int j = 0; j <= t; j++) {
                double a = 0; for (int d = 0; d < hd; d++) a += (*dO)[((size_t)(b * T + t)) * H * hd + h * hd + d] * v[((size_t)(b * T + j)) * Hkv * hd + kvh * hd + d];
                dp[j] = a; delta += s[j] * a;
            }
            for (int j = 0; j <= t; j++) {
                double ds = s[j] * (dp[j] - delta);
                for (int d = 0; d < hd; d++) {
                    (*dv)[((size_t)(b * T + j)) * Hkv * hd + kvh * hd + d] += s[j] * (*dO)[((size_t)(b * T + t)) * H * hd + h * hd + d];
                    (*dq)[((size_t)(b * T + t)) * H * hd + h * hd + d] += scale * ds * k[((size_t)(b * T + j)) * Hkv * hd + kvh * hd + d];
                    (*dk)[((size_t)(b * T + j)) * Hkv * hd + kvh * hd + d] += scale * ds * q[((size_t)(b * T + t)) * H * hd + h * hd + d];
                }
            }
        }
    }
}
// mean cross entropy over non-ignored rows; fills dlogits (softmax-onehot)/count. Returns loss (0 if none).
static double ce_d(const D& lg, const std::vector<int>& tg, int N, int V, int ign, D* dl) {
    double sum = 0; int cnt = 0;
    if (dl) dl->assign((size_t)N * V, 0);
    for (int r = 0; r < N; r++) if (tg[r] != ign) cnt++;
    for (int r = 0; r < N; r++) {
        if (tg[r] == ign) continue;
        double mx = -1e300; for (int c = 0; c < V; c++) mx = std::max(mx, lg[(size_t)r * V + c]);
        double s = 0; for (int c = 0; c < V; c++) s += exp(lg[(size_t)r * V + c] - mx);
        sum += mx + log(s) - lg[(size_t)r * V + tg[r]];
        if (dl) for (int c = 0; c < V; c++) (*dl)[(size_t)r * V + c] = (exp(lg[(size_t)r * V + c] - mx) / s - (c == tg[r])) / cnt;
    }
    return cnt ? sum / cnt : 0;
}
// AdamW (decoupled wd, bias-corrected) one step on arrays; t is 1-based.
static void adamw_d(D& p, const D& g, D& m, D& v, double lr, double b1, double b2, double eps, double wd, int t, double gs) {
    for (size_t i = 0; i < p.size(); i++) {
        double gi = g[i] * gs; m[i] = b1 * m[i] + (1 - b1) * gi; v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        double mh = m[i] / (1 - pow(b1, t)), vh = v[i] / (1 - pow(b2, t));
        p[i] = p[i] - lr * wd * p[i] - lr * mh / (sqrt(vh) + eps);
    }
}
