// Exact incremental-prefix search with bounded shallow OpenMP splitting.

#define main incremental_serial_main
#include "incremental.cpp"
#undef main

#include <omp.h>

struct IncrementalTask {
    std::vector<bint> basis;
    IncrementalState state;
};

static void GenerateIncrementalTasks(IncrementalSearch& search,
                                     const size_t depth,
                                     const size_t split_depth,
                                     std::vector<bint>& basis,
                                     std::vector<IncrementalTask>& tasks) {
    if (depth == split_depth || basis.size() == search.k) {
        tasks.push_back({basis, search.stack[depth]});
        return;
    }
    const IncrementalState& state(search.stack[depth]);
    const size_t low((size_t)basis.back()+1u);
    const size_t high((size_t)state.range+1u);
    for (size_t a=low; a<=high; ++a) {
        basis.push_back(bint(a));
        IncrementalExtend(state, a, search.stack[depth+1u]);
        GenerateIncrementalTasks(search, depth+1u, split_depth, basis, tasks);
        basis.pop_back();
    }
}

int main(int argc, char** argv) {
    if (argc <= 2) {
        std::cerr << "usage: " << argv[0]
                  << " #k #h [threads] [split-depth].\n";
        return 1;
    }
    const size_t k(std::stoul(argv[1]));
    const size_t h(std::stoul(argv[2]));
    const int threads(argc>3 ? std::stoi(argv[3]) : 8);
    const size_t split_depth(argc>4 ? std::stoul(argv[4]) : 2u);

    IncrementalSearch seed{k,h,true,true,true,true,4,true,true};
    seed.stack.resize(k);
    seed.stack[0] = IncrementalInitial(h);
    std::vector<bint> seed_basis;
    seed.best = FSelect(seed_basis,k,(uint8_t)h,0,true,0);
    seed.best_basis = seed_basis;

    std::vector<bint> basis{1};
    std::vector<IncrementalTask> tasks;
    GenerateIncrementalTasks(seed, 0, std::min(split_depth,k-1u), basis, tasks);
    bint global_best(seed.best);
    std::vector<bint> global_basis(seed.best_basis);
    unsigned long long global_states(0);
    const auto start(std::chrono::steady_clock::now());
#pragma omp parallel for schedule(dynamic,1) num_threads(threads)
    for (size_t ti=0; ti<tasks.size(); ++ti) {
        IncrementalSearch local{k,h,true,true,true,true,4,false,true};
        local.best=seed.best;
        local.best_basis=seed.best_basis;
        local.stack.resize(k);
        const size_t depth(tasks[ti].basis.size()-1u);
        local.stack[depth]=tasks[ti].state;
        std::vector<bint> local_basis(tasks[ti].basis);
        local.visit(depth,local_basis);
#pragma omp critical
        {
            global_states += local.states;
            if (local.best>global_best) {
                global_best=local.best;
                global_basis=local.best_basis;
            }
        }
    }
    const auto stop(std::chrono::steady_clock::now());
    std::cout << "#[IncrementalParallel] range: " << global_best
              << " tasks: " << tasks.size()
              << " states: " << global_states
              << " seconds: "
              << std::chrono::duration<double>(stop-start).count()
              << " basis: ";
    for (const auto& value : global_basis) std::cout << value << ' ';
    std::cout << '\n';
}
