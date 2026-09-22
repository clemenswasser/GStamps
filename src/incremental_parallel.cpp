// Exact incremental-prefix search with bounded shallow OpenMP splitting.

#define main incremental_serial_main
#include "incremental.cpp"
#undef main

int main(int argc, char** argv) {
    if (argc <= 2) {
        std::cerr << "usage: " << argv[0]
                  << " #k #h [threads] [split-depth].\n";
        return 1;
    }
    const size_t k(std::stoul(argv[1]));
    const size_t h(std::stoul(argv[2]));
    const int threads(argc>3 ? std::stoi(argv[3]) : 8);
    // Fastest measured default (matches auto policy): split4 above 8
    // threads, split3 at/below (9/4: split3 wins at 8t, split4 at 12+t).
    const size_t split_depth(argc>4 ? std::stoul(argv[4]) :
                                     (threads > 8 ? 4u : 3u));

    IncrementalOptions options=IncrementalAutomaticOptions(k, h);
    options.automatic = false;
    options.parallel = true;
    options.parallel_threads = threads;
    options.parallel_split_depth = split_depth;
    const auto start(std::chrono::steady_clock::now());
    const IncrementalParallelResult result(
        IncrementalRunParallel(k, h, options));
    const auto stop(std::chrono::steady_clock::now());
    std::cout << "#[IncrementalParallel] range: " << result.best
              << " tasks: " << result.tasks
              << " threads: " << std::max(1, threads)
              << " states: " << result.states
              << " target-skips: " << result.target_skips
              << " seconds: "
              << std::chrono::duration<double>(stop-start).count()
              << " basis: ";
    for (const auto& value : result.best_basis) std::cout << value << ' ';
    std::cout << '\n';
}
