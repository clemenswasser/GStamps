// Exact incremental-prefix search with adaptive oneTBB work stealing.

#define main incremental_serial_main
#include "incremental.cpp"
#undef main

#include <oneapi/tbb/concurrent_priority_queue.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for_each.h>
#include <oneapi/tbb/task.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>

#include <atomic>
#include <string>

struct IncrementalTbbLocalResult : IncrementalParallelResult {
    bool initialized = false;
};

inline size_t IncrementalTbbCandidateWidth(const IncrementalTask& task) {
    const size_t low((size_t)task.basis.back()+1u);
    const size_t high((size_t)task.state.range+1u);
    return high >= low ? high-low+1u : 0u;
}

inline size_t IncrementalTbbSplitScore(const size_t width,
                                       const size_t remaining) {
    const size_t limit(1u<<20);
    size_t score(1);
    for (size_t i=0; i<remaining && score<limit; ++i) {
        if (score > limit/width) return limit;
        score *= width;
    }
    return score;
}

inline void IncrementalTbbVisitLeaf(
    const size_t k, const size_t h, const IncrementalOptions& options,
    const bint seed_best, const std::vector<bint>& seed_basis,
    std::atomic<bint>* global_best, const bint upper_bound,
    const IncrementalTask& task,
    oneapi::tbb::enumerable_thread_specific<IncrementalTbbLocalResult>& workers) {
    // Cooperative group cancellation: if another worker already proved the
    // global upper bound, skip remaining leaves entirely. This is the only
    // safe use of cancel_group_execution() for branch-and-bound optimization:
    // per-subtree pruning must use the shared atomic best, not group cancel.
    if (oneapi::tbb::is_current_task_group_canceling()) return;
    IncrementalSearch local{k,h,options.bound,options.descending,
                           options.target_filter,options.final_fast,
                           options.target_count,options.automatic,
                           options.pair_filter};
    local.best=seed_best;
    local.best_basis=seed_basis;
    local.shared_best = global_best;
    local.stack.resize(k);
    const size_t depth(task.basis.size()-1u);
    local.stack[depth]=task.state;
    // S12: stack buffer for realistic k; vector fallback beyond.
    if (k <= 40) {
        bint stol[40];
        const size_t blen(task.basis.size());
        std::copy(task.basis.begin(), task.basis.end(), stol);
        IncrementalBasisBuf basis;
        basis.p = stol;
        basis.n = blen;
        local.visit(depth,basis);
    } else {
        std::vector<bint> basis(task.basis);
        local.visit(depth,basis);
    }

    IncrementalTbbLocalResult& result(workers.local());
    if (!result.initialized) {
        result.initialized=true;
        result.best=seed_best;
        result.best_basis=seed_basis;
    }
    ++result.tasks;
    result.states += local.states;
    result.target_skips += local.target_skips;
    if (local.best>result.best) {
        result.best=local.best;
        result.best_basis=local.best_basis;
    }
    // If this worker proved the root completion bound, cancel all
    // not-yet-started tasks without throwing (cancellation without exception).
    if (local.best >= upper_bound) {
        auto* ctx = oneapi::tbb::task::current_context();
        if (ctx) ctx->cancel_group_execution();
    }
}

struct IncrementalTbbContext {
    size_t k;
    size_t h;
    const IncrementalOptions& options;
    bint seed_best;
    const std::vector<bint>& seed_basis;
    std::atomic<bint>* global_best;
    bint upper_bound;
    size_t max_depth;
    size_t split_score;
    std::atomic<size_t> split_budget;
    oneapi::tbb::enumerable_thread_specific<IncrementalTbbLocalResult>& workers;

    void visit(IncrementalTask task,
               oneapi::tbb::feeder<IncrementalTask>& feeder) {
        if (oneapi::tbb::is_current_task_group_canceling()) return;
        const size_t depth(task.basis.size()-1u);
        const size_t remaining(k-task.basis.size());
        // Bound-aware: never split a subtree the serial search would prune.
        // The shared atomic best makes this check progressively stronger
        // when workers publish improvements (matters for k beyond the
        // precomputed table; for 9/4 the seed is already optimal so the
        // bound is static, but the check still avoids materializing pruned
        // tasks that the old blind splitter created).
        const bint eff(global_best ?
                       global_best->load(std::memory_order_relaxed) :
                       seed_best);
        if (options.bound &&
            IncrementalCompletionBound(task.state.range, remaining, h) <= eff) {
            // Count the pruned node as a leaf so state totals stay comparable.
            IncrementalTbbVisitLeaf(k,h,options,seed_best,seed_basis,
                                    global_best,upper_bound,task,workers);
            return;
        }
        const size_t width(IncrementalTbbCandidateWidth(task));
        const bool eligible(depth < max_depth && remaining > 1u && width > 1u &&
                             IncrementalTbbSplitScore(width,remaining)>=
                             split_score);
        size_t budget(split_budget.load(std::memory_order_relaxed));
        while (eligible && budget != 0u &&
               !split_budget.compare_exchange_weak(
                   budget, budget-1u, std::memory_order_relaxed,
                   std::memory_order_relaxed)) {}
        if (eligible && budget != 0u) {
            const size_t low((size_t)task.basis.back()+1u);
            const size_t high((size_t)task.state.range+1u);
            // Descending feed: large-a children own larger subtrees (LPT).
            for (size_t a=high;;) {
                IncrementalTask child;
                child.basis=task.basis;
                child.basis.push_back(bint(a));
                IncrementalExtend(task.state,a,child.state);
                // Mirror visit() pruning before the child enters the queue:
                // completion bound + pair feasibility. Serial visit() does
                // NOT count pruned children (only the parent), so dropped
                // children contribute 0 states here. This preserves the
                // pair-filter tree reduction that blind splitting lost
                // (the old +22k-state inflation at k=8).
                const size_t child_remaining(k-child.basis.size());
                bool keep(true);
                if (options.bound &&
                    IncrementalCompletionBound(
                        child.state.range, child_remaining, h) <= eff)
                    keep = false;
                if (keep && options.pair_filter && options.target_filter &&
                    child.basis.size()+1u==k-1u && eff>task.state.range) {
                    const size_t pair_target((size_t)eff+1u);
                    const size_t pair_low(a+1u);
                    const size_t pair_high((size_t)child.state.range+1u);
                    bool possible=false;
                    for (size_t b=pair_low; b<=pair_high && !possible; ++b)
                        possible=IncrementalCanReachTargetsWithLast(
                            child.state,b,pair_target,options.target_count);
                    if (!possible) keep = false;
                }
                if (keep) {
                    feeder.add(std::move(child));
                }
                // else: dropped, 0 states (matches serial: parent counted
                // once via its own visit, children never visited).
                if (a==low) break;
                --a;
            }
            return;
        }
        IncrementalTbbVisitLeaf(k,h,options,seed_best,seed_basis,
                                global_best,upper_bound,task,workers);
    }
};

// ---------------------------------------------------------------------------
// Shared ETS reduction (all three TBB modes).
// ---------------------------------------------------------------------------
inline IncrementalParallelResult IncrementalTbbReduce(
    oneapi::tbb::enumerable_thread_specific<IncrementalTbbLocalResult>& workers,
    const bint seed_best, const std::vector<bint>& seed_basis) {
    IncrementalParallelResult result;
    result.best=seed_best;
    result.best_basis=seed_basis;
    for (const IncrementalTbbLocalResult& local : workers) {
        if (!local.initialized) continue;
        result.tasks += local.tasks;
        result.states += local.states;
        result.target_skips += local.target_skips;
        if (local.best>result.best) {
            result.best=local.best;
            result.best_basis=local.best_basis;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Mode 2: recursive task_group (task_examples.cpp / sudoku pattern).
// Adaptive stop rule: split while the estimated subtree cost
// (width^remaining, same score as the feeder) reaches split_score, bounded
// by max_depth. Big subtrees split deep, tiny ones run serially — no manual
// depth tuning per k, no straggler leaves.
// ---------------------------------------------------------------------------
struct IncrementalTgContext {
    size_t k;
    size_t h;
    const IncrementalOptions& options;
    bint seed_best;
    const std::vector<bint>& seed_basis;
    std::atomic<bint>* global_best;
    bint upper_bound;
    size_t max_depth;
    size_t split_score;
    oneapi::tbb::task_group* tg;
    oneapi::tbb::enumerable_thread_specific<IncrementalTbbLocalResult>& workers;

    inline bint effective() const {
        return global_best ?
            global_best->load(std::memory_order_relaxed) : seed_best;
    }

    void visit_recursive(const IncrementalTask& task) {
        if (oneapi::tbb::is_current_task_group_canceling()) return;
        const size_t depth(task.basis.size()-1u);
        const size_t remaining(k-task.basis.size());
        const bint eff(effective());
        if (options.bound &&
            IncrementalCompletionBound(task.state.range, remaining, h) <= eff) {
            IncrementalTbbVisitLeaf(k,h,options,seed_best,seed_basis,
                                    global_best,upper_bound,task,workers);
            return;
        }
        const size_t width(IncrementalTbbCandidateWidth(task));
        // Adaptive: split big estimated subtrees deeper, run small ones.
        // split_score=0 keeps the old always-split-to-max-depth behavior.
        const bool big_enough =
            (split_score == 0u) ||
            (IncrementalTbbSplitScore(width, remaining) >= split_score);
        if (depth < max_depth && remaining > 1u && width > 1u &&
            big_enough) {
            const size_t low((size_t)task.basis.back()+1u);
            const size_t high((size_t)task.state.range+1u);
            // Descending spawn: large-a children (larger subtrees) first (LPT).
            for (size_t a=high;;) {
                IncrementalTask child;
                child.basis=task.basis;
                child.basis.push_back(bint(a));
                IncrementalExtend(task.state,a,child.state);
                const size_t child_remaining(k-child.basis.size());
                bool keep(true);
                if (options.bound &&
                    IncrementalCompletionBound(
                        child.state.range, child_remaining, h) <= eff)
                    keep = false;
                if (keep && options.pair_filter && options.target_filter &&
                    child.basis.size()+1u==k-1u && eff>(bint)task.state.range) {
                    const size_t pair_target((size_t)eff+1u);
                    const size_t pair_low(a+1u);
                    const size_t pair_high((size_t)child.state.range+1u);
                    bool possible=false;
                    for (size_t b=pair_low; b<=pair_high && !possible; ++b)
                        possible=IncrementalCanReachTargetsWithLast(
                            child.state,b,pair_target,options.target_count);
                    if (!possible) keep = false;
                }
                if (keep) {
                    // sudoku pattern: subproblem copied into the group.
                    // Parent is borrowed (const ref); each child owns its
                    // copy — one copy per spawn, no per-visit copies.
                    tg->run([this, child]() {
                        visit_recursive(child);
                    });
                }
                if (a==low) break;
                --a;
            }
            return;
        }
        IncrementalTbbVisitLeaf(k,h,options,seed_best,seed_basis,
                                global_best,upper_bound,task,workers);
    }
};

inline IncrementalParallelResult IncrementalRunTbb(
    const size_t k, const size_t h, const IncrementalOptions& options,
    const int threads, const size_t max_depth, const size_t split_budget) {
    IncrementalSearch seed{k,h,options.bound,options.descending,
                           options.target_filter,options.final_fast,
                           options.target_count,options.automatic,
                           options.pair_filter};
    seed.stack.resize(k);
    seed.stack[0] = IncrementalInitial(h);
    std::vector<bint> seed_basis;
    if (options.seed) {
        seed.best = FSelect(seed_basis, k, h, 0, options.seed_approx, 0);
        seed.best_basis = seed_basis;
    }

    const size_t bounded_depth(std::min(max_depth,
                                        k > 0u ? k-1u : size_t(0)));
    const size_t initial_depth(std::min(bounded_depth,
                                        k > 9u ? size_t(4) :
                                        (k > 4u ? size_t(3) :
                                         (k > 0u ? k-1u : size_t(0)))));
    std::vector<IncrementalTask> initial;
    std::vector<bint> basis{1};
    IncrementalGenerateTasks(seed,0,initial_depth,basis,initial);

    oneapi::tbb::enumerable_thread_specific<IncrementalTbbLocalResult> workers;
    const int concurrency(std::max(1, threads));
    oneapi::tbb::global_control control(
        oneapi::tbb::global_control::max_allowed_parallelism, concurrency);
    std::atomic<bint> global_best(seed.best);
    const bint upper_bound(
        IncrementalCompletionBound(IncrementalInitial(h).range, k-1u, h));
    IncrementalTbbContext context{k,h,options,seed.best,seed_basis,
                                  &global_best,upper_bound,
                                  bounded_depth,1u<<16,split_budget,workers};
    oneapi::tbb::parallel_for_each(
        initial.begin(), initial.end(),
        [&](const IncrementalTask& task,
            oneapi::tbb::feeder<IncrementalTask>& feeder) {
            context.visit(task,feeder);
        });

    return IncrementalTbbReduce(workers, seed.best, seed_basis);
}

// ---------------------------------------------------------------------------
// Mode 3: best-first priority queue (shortpath pattern).
// Global concurrent_priority_queue ordered by completion bound (largest
// first); task_group helpers try_pop -> expand-and-push or serial leaf,
// spawning a new helper while num_spawn < max_spawn. Descending child
// expansion + bound/pair gating identical to the other modes.
// ---------------------------------------------------------------------------
struct IncrementalPqItem {
    IncrementalTask task;
    bint priority = 0;
    bool operator<(const IncrementalPqItem& o) const {
        return priority < o.priority; // max-heap: largest bound on top
    }
};

struct IncrementalPqContext {
    size_t k;
    size_t h;
    const IncrementalOptions& options;
    bint seed_best;
    const std::vector<bint>& seed_basis;
    std::atomic<bint>* global_best;
    bint upper_bound;
    size_t max_depth;
    size_t max_spawn;
    std::atomic<size_t>* num_spawn;
    oneapi::tbb::task_group* tg;
    oneapi::tbb::concurrent_priority_queue<IncrementalPqItem>* pq;
    oneapi::tbb::enumerable_thread_specific<IncrementalTbbLocalResult>& workers;

    inline bint effective() const {
        return global_best ?
            global_best->load(std::memory_order_relaxed) : seed_best;
    }

    inline bint priority_of(const IncrementalTask& t) const {
        return IncrementalCompletionBound(t.state.range,
                                          k-t.basis.size(), h);
    }

    void helper() {
        IncrementalPqItem item;
        while (pq->try_pop(item)) {
            if (oneapi::tbb::is_current_task_group_canceling()) {
                // Keep num_spawn accounting exact even on early exit so a
                // future cancellation regime cannot strand work.
                num_spawn->fetch_sub(1u, std::memory_order_relaxed);
                return;
            }
            IncrementalTask task(std::move(item.task));
            const size_t depth(task.basis.size()-1u);
            const size_t remaining(k-task.basis.size());
            const bint eff(effective());
            if (options.bound &&
                IncrementalCompletionBound(task.state.range,
                                           remaining, h) <= eff) {
                IncrementalTbbVisitLeaf(k,h,options,seed_best,seed_basis,
                                        global_best,upper_bound,task,workers);
                continue;
            }
            const size_t width(IncrementalTbbCandidateWidth(task));
            if (depth < max_depth && remaining > 1u && width > 1u) {
                const size_t low((size_t)task.basis.back()+1u);
                const size_t high((size_t)task.state.range+1u);
                bool pushed=false;
                for (size_t a=high;;) {
                    IncrementalTask child;
                    child.basis=task.basis;
                    child.basis.push_back(bint(a));
                    IncrementalExtend(task.state,a,child.state);
                    const size_t child_remaining(k-child.basis.size());
                    bool keep(true);
                    if (options.bound &&
                        IncrementalCompletionBound(
                            child.state.range, child_remaining, h) <= eff)
                        keep = false;
                    if (keep && options.pair_filter && options.target_filter &&
                        child.basis.size()+1u==k-1u &&
                        eff>(bint)task.state.range) {
                        const size_t pair_target((size_t)eff+1u);
                        const size_t pair_low(a+1u);
                        const size_t pair_high((size_t)child.state.range+1u);
                        bool possible=false;
                        for (size_t b=pair_low; b<=pair_high && !possible; ++b)
                            possible=IncrementalCanReachTargetsWithLast(
                                child.state,b,pair_target,options.target_count);
                        if (!possible) keep = false;
                    }
                    if (keep) {
                        IncrementalPqItem ci;
                        ci.priority = IncrementalCompletionBound(
                            child.state.range, child_remaining, h);
                        ci.task = std::move(child);
                        pq->push(std::move(ci));
                        pushed=true;
                    }
                    if (a==low) break;
                    --a;
                }
                if (pushed) {
                    // shortpath pattern: keep parallelism fed, capped.
                    const size_t n(
                        num_spawn->fetch_add(1u, std::memory_order_relaxed)+1u);
                    if (n < max_spawn) {
                        tg->run([this]{ helper(); });
                    } else {
                        num_spawn->fetch_sub(1u, std::memory_order_relaxed);
                    }
                }
                continue;
            }
            IncrementalTbbVisitLeaf(k,h,options,seed_best,seed_basis,
                                    global_best,upper_bound,task,workers);
        }
        num_spawn->fetch_sub(1u, std::memory_order_relaxed);
    }
};

inline IncrementalParallelResult IncrementalRunTaskGroup(
    const size_t k, const size_t h, const IncrementalOptions& options,
    const int threads, const size_t max_depth, const size_t split_score) {
    IncrementalSearch seed{k,h,options.bound,options.descending,
                           options.target_filter,options.final_fast,
                           options.target_count,options.automatic,
                           options.pair_filter};
    seed.stack.resize(k);
    seed.stack[0] = IncrementalInitial(h);
    std::vector<bint> seed_basis;
    if (options.seed) {
        seed.best = FSelect(seed_basis, k, h, 0, options.seed_approx, 0);
        seed.best_basis = seed_basis;
    }

    const size_t bounded_depth(std::min(max_depth,
                                        k > 0u ? k-1u : size_t(0)));
    const size_t initial_depth(std::min(bounded_depth,
                                        k > 9u ? size_t(4) :
                                        (k > 4u ? size_t(3) :
                                         (k > 0u ? k-1u : size_t(0)))));
    std::vector<IncrementalTask> initial;
    std::vector<bint> basis{1};
    IncrementalGenerateTasks(seed,0,initial_depth,basis,initial);

    oneapi::tbb::enumerable_thread_specific<IncrementalTbbLocalResult> workers;
    const int concurrency(std::max(1, threads));
    oneapi::tbb::global_control control(
        oneapi::tbb::global_control::max_allowed_parallelism, concurrency);
    // task_arena placement control (fractal pattern): search runs inside an
    // explicitly sized arena instead of the default one.
    oneapi::tbb::task_arena arena(concurrency);
    std::atomic<bint> global_best(seed.best);
    const bint upper_bound(
        IncrementalCompletionBound(IncrementalInitial(h).range, k-1u, h));
    oneapi::tbb::task_group tg;
    IncrementalTgContext context{k,h,options,seed.best,seed_basis,
                                 &global_best,upper_bound,
                                 bounded_depth,split_score,&tg,workers};
    arena.execute([&]{
        for (auto& t : initial) {
            tg.run([&context, task=t]() {
                context.visit_recursive(task);
            });
        }
        tg.wait();
    });
    return IncrementalTbbReduce(workers, seed.best, seed_basis);
}

inline IncrementalParallelResult IncrementalRunPq(
    const size_t k, const size_t h, const IncrementalOptions& options,
    const int threads, const size_t max_depth, const size_t max_spawn_cfg) {
    IncrementalSearch seed{k,h,options.bound,options.descending,
                           options.target_filter,options.final_fast,
                           options.target_count,options.automatic,
                           options.pair_filter};
    seed.stack.resize(k);
    seed.stack[0] = IncrementalInitial(h);
    std::vector<bint> seed_basis;
    if (options.seed) {
        seed.best = FSelect(seed_basis, k, h, 0, options.seed_approx, 0);
        seed.best_basis = seed_basis;
    }

    const size_t bounded_depth(std::min(max_depth,
                                        k > 0u ? k-1u : size_t(0)));
    const size_t initial_depth(std::min(bounded_depth,
                                        k > 9u ? size_t(4) :
                                        (k > 4u ? size_t(3) :
                                         (k > 0u ? k-1u : size_t(0)))));
    std::vector<IncrementalTask> initial;
    std::vector<bint> basis{1};
    IncrementalGenerateTasks(seed,0,initial_depth,basis,initial);

    oneapi::tbb::enumerable_thread_specific<IncrementalTbbLocalResult> workers;
    const int concurrency(std::max(1, threads));
    oneapi::tbb::global_control control(
        oneapi::tbb::global_control::max_allowed_parallelism, concurrency);
    oneapi::tbb::task_arena arena(concurrency);
    std::atomic<bint> global_best(seed.best);
    const bint upper_bound(
        IncrementalCompletionBound(IncrementalInitial(h).range, k-1u, h));
    const size_t max_spawn(std::max<size_t>(
        2u, max_spawn_cfg ? max_spawn_cfg : size_t(concurrency)*4u));
    oneapi::tbb::concurrent_priority_queue<IncrementalPqItem> pq;
    for (auto& t : initial) {
        IncrementalPqItem item;
        item.priority = IncrementalCompletionBound(t.state.range,
                                                   k-t.basis.size(), h);
        item.task = std::move(t);
        pq.push(std::move(item));
    }
    std::atomic<size_t> num_spawn(1u); // first helper below is counted
    oneapi::tbb::task_group tg;
    IncrementalPqContext context{k,h,options,seed.best,seed_basis,
                                 &global_best,upper_bound,
                                 bounded_depth,max_spawn,&num_spawn,&tg,
                                 &pq,workers};
    arena.execute([&]{
        tg.run([&context]{ context.helper(); });
        tg.wait();
    });
    return IncrementalTbbReduce(workers, seed.best, seed_basis);
}

int main(int argc, char** argv) {
    if (argc <= 2) {
        std::cerr << "usage: " << argv[0]
                  << " #k #h [threads] [max-depth] [split-budget|max-spawn]"
                  << " [feeder|taskgroup|pq] [split-score].\n"
                  << "split-score (taskgroup only): split while "
                  << "width^remaining >= score; 0 = always split to max-depth.\n";
        return 1;
    }
    const size_t k(std::stoul(argv[1]));
    const size_t h(std::stoul(argv[2]));
    const int threads(argc>3 ? std::stoi(argv[3]) :
                      oneapi::tbb::info::default_concurrency());
    // Fastest measured TBB mode is taskgroup (9/4 ~5.0-5.1s vs feeder
    // ~5.1-5.4s, pq ~5.1-5.2s); keep it as default, others selectable.
    const std::string method(argc>6 ? argv[6] : "taskgroup");
    const size_t max_depth_default(
        method == "taskgroup" ? std::min(size_t(5), k > 0u ? k-1u : size_t(0))
        : std::min(k > 7u ? size_t(7) : size_t(5),
                   k > 0u ? k-1u : size_t(0)));
    const size_t max_depth(argc>4 ? std::stoul(argv[4]) : max_depth_default);
    const size_t split_budget_default(
        method == "pq" ? size_t(std::max(1, threads))*4u :
                         size_t(std::max(1, threads))*32u);
    const size_t split_budget(argc>5 ? std::stoul(argv[5]) :
                                      split_budget_default);
    // Taskgroup stop rule is the score; 0 preserves old always-split.
    // Default tuned at 9/4 (see worklist); max-depth stays a safety cap.
    const size_t split_score_default(1u<<16);
    const size_t split_score(argc>7 ? std::stoul(argv[7]) :
                                      split_score_default);

    IncrementalOptions options=IncrementalAutomaticOptions(k, h);
    options.automatic=false;
    options.parallel=false;
    const auto start(std::chrono::steady_clock::now());
    IncrementalParallelResult result;
    if (method == "taskgroup")
        result = IncrementalRunTaskGroup(k,h,options,threads,max_depth,
                                         split_score);
    else if (method == "pq")
        result = IncrementalRunPq(k,h,options,threads,max_depth,split_budget);
    else
        result = IncrementalRunTbb(k,h,options,threads,max_depth,split_budget);
    const auto stop(std::chrono::steady_clock::now());
    std::cout << "#[IncrementalTbb] range: " << result.best
              << " leaves: " << result.tasks
              << " threads: " << std::max(1, threads)
              << " max-depth: " << max_depth
              << " split-budget: " << split_budget
              << " method: " << method
              << " split-score: " << split_score
              << " states: " << result.states
              << " target-skips: " << result.target_skips
              << " seconds: "
              << std::chrono::duration<double>(stop-start).count()
              << " basis: ";
    for (const auto& value : result.best_basis) std::cout << value << ' ';
    std::cout << '\n';
    IncrementalPrintPublishes("tbb");
#ifdef GSTAMPS_CENSUS
    result.census.print("tbb");
#endif
}
