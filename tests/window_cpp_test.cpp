#undef NDEBUG   // the asserts are the test: a -DNDEBUG build must not compile them out
#include <vector>
#include <stdexcept>
#include <cassert>
struct KvmReadyGraph {std::vector<int> offsets,successors,indegree,initial;};
#include "../integration/window_graph.inc"
int main(){
 KvmReadyGraph g{{0,2,3,4,4},{1,2,3,3},{0,1,1,2},{0}};
 auto a=kvm_slice_graph(g,0,2);
 assert(a.count==2 && a.initial==std::vector<int>({0}));
 assert(a.successors==std::vector<int>({1}));
 auto b=kvm_slice_graph(g,2,2);
 assert(b.indegree==std::vector<int>({0,1}));
 assert(b.initial==std::vector<int>({0}));
 bool threw=false;try{kvm_slice_graph(g,3,2);}catch(const std::invalid_argument&){threw=true;}
 assert(threw);
}
