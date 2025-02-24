#pragma once


/**
 *
 * \details
 * Q : {N, S, E}
 * K : {N, T, E}
 * V : {N, T, F}
 */
struct MultiHeadAttentionProblemSize {
  int N, S, T, E, F, H;
  float Scale;

 public:
  // CUTLASS_HOST_DEVICE
  MultiHeadAttentionProblemSize() : N(0), S(0), T(0), E(0), F(0), H(0) {}

  /**
   * Multi-head attention.
   *
   * \param n: Batch size (always batch dimension first in VENOM)
   * \param s: Query sequence length.
   * \param t: Key-Value sequence length.
   * \param e: Inner embedding dimension.
   * \param f: Outer embedding dimension.
   * \param h: Number of heads.
   */
  // CUTLASS_HOST_DEVICE
  MultiHeadAttentionProblemSize(int n, int s, int t, int e, int f, int h)
      : N(n), S(s), T(t), E(e), F(f), H(h), Scale(1 / sqrtf(E)) {}

  /**
   * Multi-head attention.
   *
   * \param n: Batch size (always batch dimension first in VENOM)
   * \param s: Query sequence length.
   * \param t: Key-Value sequence length.
   * \param e: Inner embedding dimension.
   * \param f: Outer embedding dimension.
   * \param h: Number of heads.
   */
  // CUTLASS_HOST_DEVICE
  MultiHeadAttentionProblemSize(int n, int s, int t, int e, int f, int h,
                                float scale)
      : N(n), S(s), T(t), E(e), F(f), H(h), Scale(scale) {}

  /**
   * Scaled dot-producted attention.
   *
   * \param n: Batch size (always batch dimension first in VENOM)
   * \param s: Query sequence length.
   * \param t: Key-Value sequence length.
   * \param e: Inner embedding dimension.
   * \param f: Outer embedding dimension.
   */
  // CUTLASS_HOST_DEVICE
  MultiHeadAttentionProblemSize(int n, int s, int t, int e, int f)
      : N(n), S(s), T(t), E(e), F(f), H(1), Scale(1 / sqrtf(E)) {}

  /**
   * Multi-head self-attention.
   *
   * \param n: Batch size (always batch dimension first in VENOM)
   * \param s: Sequence length.
   * \param e: Embedding dimension.
   * \param h: Number of heads.
   */
  // CUTLASS_HOST_DEVICE
  MultiHeadAttentionProblemSize(int n, int s, int e, int h)
      : N(n), S(s), T(s), E(e), F(e), H(h), Scale(1 / sqrtf(E)) {}

  inline bool operator==(
      const MultiHeadAttentionProblemSize& problem_size) const {
    return (N == problem_size.N) && (S == problem_size.S) &&
           (T == problem_size.T) && (E == problem_size.E) &&
           (F == problem_size.F) && (H == problem_size.H);
  }

  inline bool operator!=(
      const MultiHeadAttentionProblemSize& problem_size) const {
    return !(*this == problem_size);
  }

  int QuerySize() const { return N * S * H * E; }

  int KeySize() const { return N * T * H * E; }

  int ValueSize() const { return N * T * H * F; }

  int OutputSize() const { return N * S * H * F; }

  int PSize() const { return S * T; }
};

struct Params {
  const void* query;
  const void* key;
  const void* value;
  void* output;
  const void* mask;
  bool mask_broadcast;

  Params() = default;

  Params(const void* query_, const void* key_, const void* value_,
         void* output_, bool mask_broadcast_, const void* mask_ = nullptr)
      : query(query_),
        key(key_),
        value(value_),
        output(output_),
        mask_broadcast(mask_broadcast_),
        mask(mask_) {}
};

