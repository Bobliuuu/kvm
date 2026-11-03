#include "../include/kvm_task_abi.h"
#include <cassert>
#include <fstream>
#include <vector>
#include <string>
int main(int argc, char** argv) {
    assert(argc == 3);
    std::ifstream tasks(argv[1], std::ios::binary);
    std::ifstream successors(argv[2], std::ios::binary);
    std::vector<KvmTaskDesc> desc(4);
    tasks.read(reinterpret_cast<char*>(desc.data()), 4 * sizeof(KvmTaskDesc));
    assert(tasks.gcount() == 128);
    assert(desc[0].initial_indegree == 0);
    assert(desc[0].successor_count == 2);
    assert(desc[1].priority == 10);
    assert(desc[3].initial_indegree == 2);
    std::vector<int32_t> edges(4);
    successors.read(reinterpret_cast<char*>(edges.data()), 16);
    assert(successors.gcount() == 16);
    assert((edges == std::vector<int32_t>{1,2,3,3}));
    return 0;
}
