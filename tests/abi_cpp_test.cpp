// The task ABI's binary contract, read back in C++: the host compiler writes a 4-task diamond (0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3; task 1 at
// priority 10; tests/abi_diamond.json) and this program reads its files through include/kvm_task_abi.h.
//   python3 runtime/compile_workload.py tests/abi_diamond.json /tmp/abi && g++ -std=c++17 -O2 tests/abi_cpp_test.cpp -o /tmp/abi_test && /tmp/abi_test /tmp/abi/tasks.bin /tmp/abi/successors.bin
#undef NDEBUG   // the asserts are the test: a -DNDEBUG build must not compile them out
#include "../include/kvm_task_abi.h"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <vector>
int main(int argc, char** argv) {
    assert(argc == 3);
    std::ifstream tasks(argv[1], std::ios::binary);
    std::ifstream successors(argv[2], std::ios::binary);
    std::vector<KvmTaskDesc> desc(4);
    tasks.read(reinterpret_cast<char*>(desc.data()), 4 * sizeof(KvmTaskDesc));
    assert(tasks.gcount() == 4 * 64);
    assert(tasks.peek() == std::char_traits<char>::eof());   // exactly four descriptors
    for (const KvmTaskDesc& d : desc) {
        assert(d.op_id == desc[0].op_id && d.args_words == 0);   // four noops, no arguments
        assert(d.deadline_us == -1 && d.wait_var == KVM_NO_VAR && d.pub_var == KVM_NO_VAR);
    }
    assert(desc[0].indegree == 0 && desc[0].succ_off == 0 && desc[0].succ_cnt == 2);
    assert(desc[1].indegree == 1 && desc[1].succ_off == 2 && desc[1].succ_cnt == 1 && desc[1].priority == 10);
    assert(desc[2].indegree == 1 && desc[2].succ_off == 3 && desc[2].succ_cnt == 1 && desc[2].priority == 0);
    assert(desc[3].indegree == 2 && desc[3].succ_off == 4 && desc[3].succ_cnt == 0);
    std::vector<int32_t> edges(4);
    successors.read(reinterpret_cast<char*>(edges.data()), 16);
    assert(successors.gcount() == 16);
    assert((edges == std::vector<int32_t>{1, 2, 3, 3}));
    std::puts("abi: the compiled diamond reads back through KvmTaskDesc");
    return 0;
}
