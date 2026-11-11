#pragma once
// The fused MLP beds' host side (split_bed.cu, grid_bed.cu): the panel's MLP with its data, drawn the same way for every bed, and the float64
// reference of its training steps. The model: embedding V x E, L hidden layers of H units (linear with bias, then tanh GELU), a head H x V without
// bias, softmax cross-entropy over the step's T tokens, SGD. A bed's descriptor names the shape (fields E, V, H, T, L), so the reference reads any.
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace mlp_ref {

constexpr int MAXL = 4;

// The model in float, as the device holds it, and the step's tokens and targets: one generator, one draw order, so two beds given one seed and
// one shape train the same model on the same data.
struct Data {
  std::vector<float> W[MAXL], b[MAXL], Wh, emb;
  std::vector<int> ids, tgt;   // [steps][T]
};
template <class D> Data draw(const D& d, unsigned seed, int steps) {
  Data x;
  std::mt19937 rng(seed); std::uniform_real_distribution<float> u(-1.f, 1.f); std::uniform_int_distribution<int> tok(0, d.V - 1);
  x.Wh.resize((size_t)d.H * d.V); x.emb.resize((size_t)d.V * d.E);
  for (auto& v : x.emb) v = 0.5f * u(rng);
  for (int l = 0; l < d.L; ++l) {
    const int K = l ? d.H : d.E;
    x.W[l].resize((size_t)K * d.H); for (auto& v : x.W[l]) v = u(rng) / std::sqrt((float)K);
    x.b[l].resize(d.H); for (auto& v : x.b[l]) v = 0.1f * u(rng);
  }
  for (auto& v : x.Wh) v = u(rng) / std::sqrt((float)d.H);
  x.ids.resize((size_t)steps * d.T); x.tgt.resize((size_t)steps * d.T);
  for (auto& v : x.ids) v = tok(rng);
  for (auto& v : x.tgt) v = tok(rng);
  return x;
}

struct Model { std::vector<double> W[MAXL], b[MAXL], Wh, emb; };
inline Model model_of(const Data& x, int L) {
  Model m;
  for (int l = 0; l < L; ++l) { m.W[l].assign(x.W[l].begin(), x.W[l].end()); m.b[l].assign(x.b[l].begin(), x.b[l].end()); }
  m.Wh.assign(x.Wh.begin(), x.Wh.end()); m.emb.assign(x.emb.begin(), x.emb.end());
  return m;
}

inline double gelu_d(double x) { const double y = 0.7978845608028654 * (x + 0.044715 * x * x * x); return 0.5 * x * (1.0 + std::tanh(y)); }
inline double gelu_gd(double x) { const double y = 0.7978845608028654 * (x + 0.044715 * x * x * x), t = std::tanh(y); return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x); }

// k training steps of `m` in float64 at learning rate lr; returns each step's loss.
template <class D> std::vector<double> ref_steps(const D& d, Model& m, const std::vector<int>& ids, const std::vector<int>& tgt, int k, double lr) {
  const int E = d.E, V = d.V, H = d.H, T = d.T, L = d.L; std::vector<double> losses;
  for (int step = 0; step < k; ++step) {
    const int* id = &ids[(size_t)step * T]; const int* tg = &tgt[(size_t)step * T];
    std::vector<std::vector<double>> A(L + 1), Z(L);   // A[l]: layer l's input (A[0] the embedding rows, T x E; A[l > 0] T x H); A[L] the head's input
    A[0].assign((size_t)T * E, 0.0);
    for (int r = 0; r < T; ++r) for (int e = 0; e < E; ++e) A[0][(size_t)r * E + e] = m.emb[(size_t)id[r] * E + e];
    for (int l = 0; l < L; ++l) {
      const int K = l ? H : E;
      Z[l].assign((size_t)T * H, 0.0); A[l + 1].assign((size_t)T * H, 0.0);
      for (int r = 0; r < T; ++r) for (int c = 0; c < H; ++c) { double a = m.b[l][c]; for (int kk = 0; kk < K; ++kk) a += A[l][(size_t)r * K + kk] * m.W[l][(size_t)kk * H + c]; Z[l][(size_t)r * H + c] = a; A[l + 1][(size_t)r * H + c] = gelu_d(a); }
    }
    std::vector<double> Lg((size_t)T * V), dL((size_t)T * V);
    double loss = 0.0;
    for (int r = 0; r < T; ++r) {
      double mx = -1e300; for (int v = 0; v < V; ++v) { double a = 0; for (int c = 0; c < H; ++c) a += A[L][(size_t)r * H + c] * m.Wh[(size_t)c * V + v]; Lg[(size_t)r * V + v] = a; mx = std::max(mx, a); }
      double z = 0; for (int v = 0; v < V; ++v) z += std::exp(Lg[(size_t)r * V + v] - mx);
      loss += mx + std::log(z) - Lg[(size_t)r * V + tg[r]];
      for (int v = 0; v < V; ++v) dL[(size_t)r * V + v] = (std::exp(Lg[(size_t)r * V + v] - mx) / z - (v == tg[r] ? 1.0 : 0.0)) / T;
    }
    losses.push_back(loss / T);
    std::vector<double> gWh((size_t)H * V, 0.0), gEmb((size_t)V * E, 0.0), dH((size_t)T * H, 0.0);
    std::vector<std::vector<double>> gW(L), gb(L);
    for (int c = 0; c < H; ++c) for (int v = 0; v < V; ++v) { double a = 0; for (int r = 0; r < T; ++r) a += A[L][(size_t)r * H + c] * dL[(size_t)r * V + v]; gWh[(size_t)c * V + v] = a; }
    for (int r = 0; r < T; ++r) for (int c = 0; c < H; ++c) { double a = 0; for (int v = 0; v < V; ++v) a += dL[(size_t)r * V + v] * m.Wh[(size_t)c * V + v]; dH[(size_t)r * H + c] = a; }
    for (int l = L - 1; l >= 0; --l) {
      const int K = l ? H : E;
      std::vector<double> dZ((size_t)T * H);
      for (size_t i = 0; i < dZ.size(); ++i) dZ[i] = dH[i] * gelu_gd(Z[l][i]);
      gW[l].assign((size_t)K * H, 0.0); gb[l].assign(H, 0.0);
      for (int kk = 0; kk < K; ++kk) for (int c = 0; c < H; ++c) { double a = 0; for (int r = 0; r < T; ++r) a += A[l][(size_t)r * K + kk] * dZ[(size_t)r * H + c]; gW[l][(size_t)kk * H + c] = a; }
      for (int c = 0; c < H; ++c) { double a = 0; for (int r = 0; r < T; ++r) a += dZ[(size_t)r * H + c]; gb[l][c] = a; }
      std::vector<double> dA((size_t)T * K, 0.0);
      for (int r = 0; r < T; ++r) for (int kk = 0; kk < K; ++kk) { double a = 0; for (int c = 0; c < H; ++c) a += dZ[(size_t)r * H + c] * m.W[l][(size_t)kk * H + c]; dA[(size_t)r * K + kk] = a; }
      if (l == 0) { for (int r = 0; r < T; ++r) for (int e = 0; e < E; ++e) gEmb[(size_t)id[r] * E + e] += dA[(size_t)r * E + e]; }
      else dH = dA;
    }
    for (int l = 0; l < L; ++l) { for (size_t i = 0; i < m.W[l].size(); ++i) m.W[l][i] -= lr * gW[l][i]; for (size_t i = 0; i < m.b[l].size(); ++i) m.b[l][i] -= lr * gb[l][i]; }
    for (size_t i = 0; i < m.Wh.size(); ++i) m.Wh[i] -= lr * gWh[i];
    for (size_t i = 0; i < m.emb.size(); ++i) m.emb[i] -= lr * gEmb[i];
  }
  return losses;
}

// The relative L2 error of an update (after minus before) against the reference's update.
inline double rel_delta(const std::vector<float>& gpu, const std::vector<double>& ref, const std::vector<float>& init) {
  double n = 0, dd = 0; for (size_t i = 0; i < ref.size(); ++i) { const double r = ref[i] - init[i], g = (double)gpu[i] - init[i]; n += r * r; dd += (g - r) * (g - r); }
  return std::sqrt(dd / std::max(n, 1e-300));
}

}  // namespace mlp_ref
