#pragma once
// Minimal KvmCtx for the standalone op harnesses (no dependency on the executor). Field names are the three the op library
// touches (see include/kvm_ops_gpt.cuh).
#include "kvm_task_abi.h"
struct KvmCtx {
    const KvmBuf* bufs;
    int32_t nbufs;
    int32_t* err;
};
