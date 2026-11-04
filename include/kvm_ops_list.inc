// The one X-macro list of every operator registry file. Optional registries (GPT, game) are picked up when present.
// Ids are sparse and globally unique: 0-31 core, 32-127 gpt, 128+ game. Include this wherever X(name, id, sig) is expanded.
#include "kvm_ops.def"
#if __has_include("kvm_ops_gpt.def")
#include "kvm_ops_gpt.def"
#endif
#if __has_include("kvm_ops_game.def")
#include "kvm_ops_game.def"
#endif
// Test and extension hook: -DKVM_EXTRA_OPS_DEF='"my_ops.def"' (with the implementation via -DKVM_EXTRA_OPS_CUH='"my_ops.cuh"').
#ifdef KVM_EXTRA_OPS_DEF
#include KVM_EXTRA_OPS_DEF
#endif
