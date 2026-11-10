// The one X-macro list of every operator registry file. Optional registries (GPT, game) are picked up when present.
// Ids are sparse and globally unique: 0-31 core, 32-127 gpt, 128+ game. Include this wherever X(name, id, sig) is expanded.
#include "kvm_ops.def"
#if __has_include("kvm_ops_comm.def") && !defined(KVM_LEAN)  // KVM_DIST_HOOK: ids 64..95 (-DKVM_LEAN skips the comm and extra packs for fast builds)
#include "kvm_ops_comm.def"
#endif
#include "kvm_ops_fmlp.def"
#if __has_include("kvm_ops_gpt.def")
#include "kvm_ops_gpt.def"
#endif
// Extra generic pack (ids 96-127): activations, SwiGLU, softmax, gather/scatter, MoE routing, bf16, schedules, matmul epilogues.
#if __has_include("kvm_ops_extra.def") && !defined(KVM_LEAN)
#include "kvm_ops_extra.def"
#endif
// The game pack links the Doom engine (nvcc -rdc=true + doom/build/doom_dev.o): opt in with -DKVM_WITH_GAME.
#if defined(KVM_WITH_GAME) && __has_include("kvm_ops_game.def")
#include "kvm_ops_game.def"
#endif
// Test and extension hook: -DKVM_EXTRA_OPS_DEF='"my_ops.def"' (with the implementation via -DKVM_EXTRA_OPS_CUH='"my_ops.cuh"').
#ifdef KVM_EXTRA_OPS_DEF
#include KVM_EXTRA_OPS_DEF
#endif
