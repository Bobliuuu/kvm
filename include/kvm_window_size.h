// kvm_window_size.h: the two counts kvm_exec's admission pools rest on, host only so CPU tests can check them.
//
// A bundle goes to the device as records: each holds a header, its tasks' descriptors, their successor edges and their arguments
// (kvm_record_words). The capacities that bound a record, and the rule that sizes them for one record, are KvmConfig's (kvm_exec.cu).
// The successor ring (succ_cap words) holds the edges of the records still running; kvm_succ_ring_words is its floor.
#pragma once

// The words of one record of `tasks` tasks, `edges` successor edges and `args_words` argument words. KvmBundle::words() and
// KvmConfig::record_words both count with it, so what the host sizes for and what it admits cannot differ.
inline long long kvm_record_words(long long tasks, long long edges, long long args_words, long long hdr_words, long long task_words) {
    return (hdr_words + tasks * task_words + edges + args_words + 7) & ~7ll;
}

// The successor ring a configuration needs. A record's edges are among its words, and the host never sends a record larger than the
// admission ring (submit() refuses one), so a successor ring at least as large as the admission ring fits every record, whatever produced
// it: a sizing flag, or a transform that adds edges after the window was sized. A smaller ring can meet a record it can never admit
// (DEVICE ERROR POOL).
inline unsigned kvm_succ_ring_words(unsigned succ_cap, int ring_words) {
    while (succ_cap < (unsigned)ring_words) succ_cap <<= 1;   // succ_cap is a power of two >= 2; ring_words is at most 2^28 (the CLI's bound)
    return succ_cap;
}
