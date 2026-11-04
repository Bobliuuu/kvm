#pragma once
// Minimal KvmCtx for the standalone op harnesses (no dependency on the executor): the same three members the op library uses
// from kvm_exec.cuh's KvmCtx (bufs, nbufs, fail).
#include "kvm_task_abi.h"
struct KvmCtx {
    const KvmBuf* bufs;
    int32_t nbufs;
    int32_t* err;   // device word: first nonzero code wins
    __device__ void fail(int code, int = 0, int = 0) const { atomicCAS(err, 0, code); }
};
