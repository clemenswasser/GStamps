// ============================================================================
// GStamps: exact incremental-prefix exhaustive search
// ============================================================================

#include <gstamps.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <string>

// S1: always-on improvement census. publish_best is cold (never fires in a
// seeded-optimal 9/4 proof), so this relaxed increment costs nothing
// measurable on the hot path. Report with GSTAMPS_CENSUS=1 in the env.
inline std::atomic<unsigned long long> g_incremental_publishes{0};

inline void IncrementalPrintPublishes(const char* tag) {
    if (std::getenv("GSTAMPS_CENSUS")) {
        std::fprintf(stderr, "#[census-publishes] %s %llu\n", tag,
                     g_incremental_publishes.load(std::memory_order_relaxed));
    }
}

#ifdef GSTAMPS_CENSUS
// S2/S3: distribution census (census builds only; production binary has zero
// trace of this: no fields, no branches). Thread-local sink pointer so the
// free-function predicates can record without signature changes.
struct IncrementalCensus {
    unsigned long long singular_calls = 0;
    unsigned long long direct_hits = 0;
    unsigned long long copies_iters = 0;
    unsigned long long copies_hits[9] = {};
    unsigned long long plural_calls = 0;
    unsigned long long plural_targets_tested = 0;
    unsigned long long direct_sameword = 0;
    unsigned long long copies_sameword = 0;
    unsigned long long finalrange_calls = 0;
    unsigned long long finalrange_losers = 0;
    unsigned long long pair_nodes = 0;
    unsigned long long pair_b_tested = 0;
    unsigned long long pair_pruned = 0;
    void merge(const IncrementalCensus& o) {
        singular_calls += o.singular_calls;
        direct_hits += o.direct_hits;
        copies_iters += o.copies_iters;
        for (int i = 0; i < 9; ++i) copies_hits[i] += o.copies_hits[i];
        plural_calls += o.plural_calls;
        plural_targets_tested += o.plural_targets_tested;
        direct_sameword += o.direct_sameword;
        copies_sameword += o.copies_sameword;
        finalrange_calls += o.finalrange_calls;
        finalrange_losers += o.finalrange_losers;
        pair_nodes += o.pair_nodes;
        pair_b_tested += o.pair_b_tested;
        pair_pruned += o.pair_pruned;
    }
    void print(const char* tag) const {
        std::fprintf(stderr, "#[census] %s singular=%llu direct_hits=%llu "
                     "copies_iters=%llu copies_hits=[%llu %llu %llu %llu] "
                     "plural=%llu targets_tested=%llu direct_sameword=%llu "
                     "copies_sameword=%llu finalrange=%llu losers=%llu "
                     "pair_nodes=%llu pair_b=%llu pair_pruned=%llu\n",
                     tag, singular_calls, direct_hits, copies_iters,
                     copies_hits[1], copies_hits[2], copies_hits[3],
                     copies_hits[4], plural_calls, plural_targets_tested,
                     direct_sameword, copies_sameword, finalrange_calls,
                     finalrange_losers, pair_nodes, pair_b_tested,
                     pair_pruned);
    }
};
thread_local IncrementalCensus* t_incremental_census = nullptr;
// Lightweight census taps for new code (compiled out when the macro is off).
#define GSTAMPS_COUNT(field) \
    do { if (t_incremental_census) ++t_incremental_census->field; } while (0)
#define GSTAMPS_COUNT_IDX(field, i) \
    do { if (t_incremental_census && (size_t)(i) < 9) \
        ++t_incremental_census->field[i]; } while (0)
#else
#define GSTAMPS_COUNT(field) do {} while (0)
#define GSTAMPS_COUNT_IDX(field, i) do {} while (0)
#endif

struct IncrementalOptions {
    bool automatic = false;
    bool parallel = false;
    bool bound = false;
    bool descending = false;
    bool seed = false;
    bool seed_approx = false;
    bool target_filter = false;
    bool final_fast = false;
    size_t target_count = 1;
    bool pair_filter = false;
    int parallel_threads = 1;
    size_t parallel_split_depth = 2;
};

inline IncrementalOptions IncrementalAutomaticOptions(const size_t k,
                                                      const size_t h) {
    IncrementalOptions options;
    options.automatic = true;
    options.bound = true;
    options.descending = true;
    options.seed = (k >= 4u);
    options.seed_approx = true;
    options.target_filter = (k >= 4u);
    options.final_fast = (k >= 4u);
    options.target_count = (h == 4u) ? 5u : 1u;
    options.pair_filter = (h == 4u && k >= 6u);
    options.parallel = (h == 4u && k >= 8u && omp_get_max_threads() > 1);
    options.parallel_threads = omp_get_max_threads();
    // Fastest static partition measured 2026-09-22 on i7-12700K (9/4):
    // split3/291 tasks ~4.9s at 20 threads, split4/4752 tasks ~4.86s;
    // at 8 threads split3 ~6.99s vs split4 ~7.02s, at 12+ threads split4
    // wins. Keep the fastest as default via thread-count rule.
    options.parallel_split_depth =
        (options.parallel_threads > 8) ? 4u : 3u;
    return options;
}

struct IncrementalState {
    size_t h = 0;
    size_t max_value = 0;
    size_t words = 0;
    std::vector<uint64_t> bits;
    bint range = 0;
};

// S12: DFS-internal basis as a raw stack buffer (no capacity branches).
// Same minimal API as the vector subset used by visit(); task transport
// and snapshots stay vectors. k<=40 uses the buffer, larger k keeps the
// vector path via overload resolution.
struct IncrementalBasisBuf {
    bint* p = nullptr;
    size_t n = 0;
    size_t size() const { return n; }
    bint back() const { return p[n-1]; }
    const bint* data() const { return p; }
    void push_back(const bint v) { p[n++] = v; }
    void pop_back() { --n; }
};

inline bint IncrementalFirstZero(const uint64_t* bits, const size_t upper) {
    const size_t full(upper >> 6);
    for (size_t w=0; w<full; ++w) {
        const uint64_t inv(~bits[w]);
        if (inv) return bint(w*64u + __builtin_ctzll(inv));
    }
    const unsigned rem((unsigned)(upper & 63u));
    if (rem) {
        const uint64_t inv(~(bits[full] | (~0ULL << rem)));
        if (inv) return bint(full*64u + __builtin_ctzll(inv));
    }
    return bint(upper);
}

inline bint IncrementalFirstZeroFrom(const uint64_t* bits, const size_t upper,
                                     const size_t start) {
    if (start >= upper) return bint(upper);
    size_t word(start >> 6);
    const size_t full(upper >> 6);
    uint64_t mask(~0ULL << (start & 63u));
    for (; word<full; ++word, mask=~0ULL) {
        const uint64_t inv((~bits[word]) & mask);
        if (inv) return bint(word*64u + __builtin_ctzll(inv));
    }
    const unsigned rem((unsigned)(upper & 63u));
    if (rem && word==full) {
        const uint64_t valid(~0ULL >> (64u-rem));
        const uint64_t inv((~bits[full]) & mask & valid);
        if (inv) return bint(full*64u + __builtin_ctzll(inv));
    }
    return bint(upper);
}

inline void IncrementalResize(IncrementalState& state, const size_t h,
                              const size_t max_value) {
    state.h = h;
    state.max_value = max_value;
    state.words = (max_value + 64u) / 64u;
    state.bits.resize((h+1u)*state.words);
    std::fill(state.bits.begin(), state.bits.end(), 0u);
}

inline IncrementalState IncrementalInitial(const size_t h) {
    IncrementalState state;
    IncrementalResize(state, h, h);
    for (size_t d=0; d<=h; ++d) {
        for (size_t x=0; x<=d; ++x)
            state.bits[d*state.words+(x>>6)] |= 1ULL << (x&63u);
    }
    state.range = bint(h);
    return state;
}

inline void IncrementalShiftOr(uint64_t* destination, const uint64_t* previous,
                               const size_t words, const size_t a) {
    const size_t word_shift(a >> 6);
    if (word_shift >= words) return;
    const unsigned bit_shift((unsigned)(a & 63u));
    if (bit_shift == 0) {
        for (size_t i=word_shift; i<words; ++i)
            destination[i] |= previous[i-word_shift];
        return;
    }
    const unsigned reverse_shift(64u-bit_shift);
    destination[word_shift] |= previous[0] << bit_shift;
    for (size_t i=word_shift+1u; i<words; ++i)
        destination[i] |= (previous[i-word_shift] << bit_shift) |
                          (previous[i-word_shift-1u] >> reverse_shift);
}

inline void IncrementalExtendH4(const IncrementalState& parent,
                                const size_t a, IncrementalState& child) {
    IncrementalResize(child, 4u, 4u*a);
    for (size_t d=0; d<=4u; ++d)
        std::copy(parent.bits.begin()+d*parent.words,
                  parent.bits.begin()+d*parent.words+parent.words,
                  child.bits.begin()+d*child.words);
    IncrementalShiftOr(child.bits.data()+child.words,
                       child.bits.data(), child.words, a);
    IncrementalShiftOr(child.bits.data()+2u*child.words,
                       child.bits.data()+child.words, child.words, a);
    IncrementalShiftOr(child.bits.data()+3u*child.words,
                       child.bits.data()+2u*child.words, child.words, a);
    IncrementalShiftOr(child.bits.data()+4u*child.words,
                       child.bits.data()+3u*child.words, child.words, a);
    child.range = IncrementalFirstZeroFrom(
        child.bits.data()+4u*child.words, child.max_value+1u,
        (size_t)parent.range+1u) - 1;
}

// Add one denomination without recomputing the prefix from scratch.
inline void IncrementalExtend(const IncrementalState& parent, const size_t a,
                             IncrementalState& child) {
    if (parent.h == 4u) {
        IncrementalExtendH4(parent, a, child);
        return;
    }
    IncrementalResize(child, parent.h, parent.h*a);
    for (size_t d=0; d<=parent.h; ++d) {
        const size_t copy_words(std::min(parent.words, child.words));
        std::copy(parent.bits.begin()+d*parent.words,
                  parent.bits.begin()+d*parent.words+copy_words,
                  child.bits.begin()+d*child.words);
    }

    const size_t word_shift(a >> 6);
    const unsigned bit_shift((unsigned)(a & 63u));
    const unsigned reverse_shift(64u-bit_shift);
    for (size_t d=1; d<=parent.h; ++d) {
        uint64_t* destination(child.bits.data()+d*child.words);
        const uint64_t* previous(child.bits.data()+(d-1u)*child.words);
        for (size_t i=word_shift; i<child.words; ++i) {
            uint64_t shifted(previous[i-word_shift] << bit_shift);
            if (bit_shift && i>word_shift)
                shifted |= previous[i-word_shift-1u] >> reverse_shift;
            destination[i] |= shifted;
        }
    }
    child.range = IncrementalFirstZeroFrom(
        child.bits.data()+parent.h*child.words,
        child.max_value+1u,
        (size_t)parent.range+1u) - 1;
}

inline bint IncrementalCompletionBound(const bint range, const size_t remaining,
                                       const size_t h) {
    bint last(range+1);
    for (size_t i=1; i<remaining; ++i)
        last = bint(h)*last + 1;
    return bint(h)*last;
}

inline bool IncrementalCanReachTargetWithLast(const IncrementalState& prefix,
                                              const size_t a,
                                              const size_t target) {
#ifdef GSTAMPS_CENSUS
    IncrementalCensus* census = t_incremental_census;
    if (census) ++census->singular_calls;
#endif
    if (target <= prefix.max_value &&
        ((prefix.bits[prefix.h*prefix.words+(target>>6)] >>
          (target&63u)) & 1u)) {
#ifdef GSTAMPS_CENSUS
        if (census) ++census->direct_hits;
#endif
        return true;
    }
    for (size_t copies=1; copies<=prefix.h; ++copies) {
        if (copies*a > target) break;
#ifdef GSTAMPS_CENSUS
        if (census) ++census->copies_iters;
#endif
        const size_t remainder(target-copies*a);
        const size_t depth(prefix.h-copies);
        if (remainder > depth*prefix.max_value) continue;
        if ((prefix.bits[depth*prefix.words+(remainder>>6)] >>
             (remainder&63u)) & 1u) {
#ifdef GSTAMPS_CENSUS
            if (census && copies < 9) ++census->copies_hits[copies];
#endif
            return true;
        }
    }
    return false;
}

inline bool IncrementalCanReachTargetsWithLast(const IncrementalState& prefix,
                                               const size_t a,
                                               const size_t first_target,
                                               const size_t target_count) {
#ifdef GSTAMPS_CENSUS
    IncrementalCensus* census = t_incremental_census;
    if (census) {
        ++census->plural_calls;
        // Same-word coincidence rates for S4 design (direct row-h over
        // T..T+count-1; per-copies remainder runs).
        if (((first_target >> 6) == ((first_target+target_count-1) >> 6)))
            ++census->direct_sameword;
        for (size_t c = 1; c <= prefix.h; ++c) {
            if (c*a > first_target+target_count-1) break;
            if (c*a <= first_target &&
                ((first_target-c*a) >> 6) ==
                ((first_target+target_count-1-c*a) >> 6))
                ++census->copies_sameword;
        }
    }
#endif
    for (size_t offset=0; offset<target_count; ++offset) {
#ifdef GSTAMPS_CENSUS
        if (census) ++census->plural_targets_tested;
#endif
        if (!IncrementalCanReachTargetWithLast(prefix, a,
                                               first_target+offset))
            return false;
    }
    return true;
}

// S5: per-visit prefix view. Row bases, words, per-depth reachability
// limits and the top row are hoisted ONCE per visit() node (amortized over
// all candidates x targets) instead of recomputed per predicate call.
struct IncrementalPrefixView {
    const uint64_t* rows[16];
    size_t h;
    size_t words;
    size_t max_value;
    size_t limits[16];
    const uint64_t* rowh;
    const IncrementalState* state;
};

inline IncrementalPrefixView IncrementalMakeView(const IncrementalState& prefix) {
    IncrementalPrefixView view;
    view.h = prefix.h;
    view.words = prefix.words;
    view.max_value = prefix.max_value;
    const uint64_t* base = prefix.bits.data();
    const size_t n = prefix.h < 16 ? prefix.h : 15;
    for (size_t d = 0; d <= n; ++d) {
        view.rows[d] = base + d*prefix.words;
        view.limits[d] = d*prefix.max_value;
    }
    view.rowh = base + prefix.h*prefix.words;
    view.state = &prefix;
    return view;
}

// S4r/S5/S6: per-target (short-circuit) order with per-node-hoisted
// rows/limits. Full cross-target fusion (first S4 attempt) lost to the
// plural short-circuit (legacy evaluates ~1.7 of 5 targets per call);
// this keeps the legacy evaluation ORDER (same fail-fast behavior) and
// only removes per-test overhead: hoisted row bases/limits, raw indexing,
// h==4-unrolled copies loop (S2 census: copies=1 resolves ~74% of hits,
// ascending confirmed optimal). Same boolean function as the legacy loop.
inline bool IncrementalRowBit(const uint64_t* row, const size_t idx) {
    return (bool)((row[idx >> 6] >> (idx & 63u)) & 1u);
}

inline bool IncrementalCanReachTargetsView(const IncrementalPrefixView& V,
                                           const size_t a,
                                           const size_t first_target,
                                           const size_t target_count) {
    GSTAMPS_COUNT(plural_calls);
    for (size_t o = 0; o < target_count; ++o) {
        GSTAMPS_COUNT(plural_targets_tested);
        const size_t t = first_target + o;
        if (t <= V.max_value && IncrementalRowBit(V.rowh, t)) {
            GSTAMPS_COUNT(direct_hits);
            continue;
        }
        if (V.h == 4) {
            size_t r;
            if (a > t) return false;
            GSTAMPS_COUNT(copies_iters);
            r = t - a;
            if (r <= V.limits[3] && IncrementalRowBit(V.rows[3], r)) {
                GSTAMPS_COUNT_IDX(copies_hits, 1);
                continue;
            }
            if (2u*a > t) return false;
            GSTAMPS_COUNT(copies_iters);
            r = t - 2u*a;
            if (r <= V.limits[2] && IncrementalRowBit(V.rows[2], r)) {
                GSTAMPS_COUNT_IDX(copies_hits, 2);
                continue;
            }
            if (3u*a > t) return false;
            GSTAMPS_COUNT(copies_iters);
            r = t - 3u*a;
            if (r <= V.limits[1] && IncrementalRowBit(V.rows[1], r)) {
                GSTAMPS_COUNT_IDX(copies_hits, 3);
                continue;
            }
            if (4u*a > t) return false;
            GSTAMPS_COUNT(copies_iters);
            r = t - 4u*a;
            if (r <= V.limits[0] && IncrementalRowBit(V.rows[0], r)) {
                GSTAMPS_COUNT_IDX(copies_hits, 4);
                continue;
            }
            return false;
        } else {
            bool ok = false;
            for (size_t c = 1; c <= V.h; ++c) {
                if (c*a > t) break;
                GSTAMPS_COUNT(copies_iters);
                const size_t r = t - c*a;
                const size_t d = V.h - c;
                if (r > V.limits[d]) continue;
                if (IncrementalRowBit(V.rows[d], r)) {
                    GSTAMPS_COUNT_IDX(copies_hits, c);
                    ok = true;
                    break;
                }
            }
            if (!ok) return false;
        }
    }
    return true;
}

#ifdef GSTAMPS_FUZZ_PRED
inline bool IncrementalPredFuzzCheck(const IncrementalState& prefix,
                                     const IncrementalPrefixView& view,
                                     const size_t a,
                                     const size_t first_target,
                                     const size_t target_count) {
    const bool leg = IncrementalCanReachTargetsWithLast(prefix, a,
                                                        first_target,
                                                        target_count);
    const bool fus = IncrementalCanReachTargetsView(view, a, first_target,
                                                     target_count);
    if (leg != fus) {
        std::fprintf(stderr, "PRED MISMATCH a=%zu first=%zu count=%zu "
                     "leg=%d fus=%d\n",
                     a, first_target, target_count, (int)leg, (int)fus);
        std::abort();
    }
    return fus;
}
#endif

// S4r/S5 dispatch: view-based kernel on a caller-built per-node view;
// legacy path for out-of-range configurations (never in h=4 practice).
inline bool IncrementalCanReachTargetsCall(const IncrementalState& prefix,
                                           const IncrementalPrefixView& view,
                                           const bool use_view,
                                           const size_t a,
                                           const size_t first_target,
                                           const size_t target_count) {
    if (!use_view)
        return IncrementalCanReachTargetsWithLast(prefix, a, first_target,
                                                  target_count);
#ifdef GSTAMPS_FUZZ_PRED
    return IncrementalPredFuzzCheck(prefix, view, a, first_target,
                                    target_count);
#else
    return IncrementalCanReachTargetsView(view, a, first_target,
                                          target_count);
#endif
}

inline bint IncrementalFinalRange(const IncrementalState& prefix,
                                  const size_t a,
                                  std::vector<uint64_t>& scratch) {
    (void)scratch; // S10: no scratch build anymore (see below).
    const size_t words((prefix.h*a+64u)/64u);
    const uint64_t* hrow = prefix.bits.data()+prefix.h*prefix.words;
    const size_t copy_words(std::min(prefix.words, words));
    const size_t start((size_t)prefix.range+1u);
    const size_t upper(prefix.h*a+1u);
    if (start >= upper) return bint(upper)-1;
    // S10: gather each word across copies in ascending order and test the
    // hole immediately. The first hole at/above start decides the range, so
    // words above the hole word are never built (legacy built all of them).
    // Per-word values match the legacy copies-outer build exactly: h-row
    // copy for w < copy_words, shifted OR per copies with identical guards.
    for (size_t w = start>>6; w < words; ++w) {
        uint64_t v = (w < copy_words) ? hrow[w] : 0u;
        for (size_t copies=1; copies<=prefix.h; ++copies) {
            const size_t shift(copies*a);
            const size_t word_shift(shift>>6);
            if (word_shift > w) continue;
            const unsigned bit_shift((unsigned)(shift&63u));
            const uint64_t* source(prefix.bits.data()+
                                   (prefix.h-copies)*prefix.words);
            const size_t si(w-word_shift);
            if (si >= prefix.words) continue;
            uint64_t part(source[si] << bit_shift);
            if (bit_shift && si > 0)
                part |= source[si-1u] >> (64u-bit_shift);
            v |= part;
        }
        uint64_t test(v);
        if (w == (start>>6)) {
            const unsigned lo((unsigned)(start&63u));
            if (lo) test |= (1ULL << lo) - 1u; // below start: known set
        }
        const uint64_t inv(~test);
        if (inv) {
            const size_t pos(w*64u+(unsigned)__builtin_ctzll(inv));
            if (pos < upper) return bint(pos)-1;
            return bint(upper)-1; // hole past the range: capped like rem-mask
        }
    }
    return bint(upper)-1;
}

struct IncrementalSearch {
    size_t k;
    size_t h;
    bool bound;
    bool descending;
    bool target_filter;
    bool final_fast;
    size_t target_count;
    bool automatic;
    bool pair_filter;
    bint best = 0;
    std::vector<bint> best_basis;
    unsigned long long states = 0;
    unsigned long long target_skips = 0;
    std::vector<IncrementalState> stack;
    std::vector<uint64_t> final_scratch;
    std::atomic<bint>* shared_best = nullptr;

    inline bint effective_best() const {
        if (shared_best) {
            const bint g(shared_best->load(std::memory_order_relaxed));
            return g > best ? g : best;
        }
        return best;
    }

    inline void publish_best(const bint value) {
        if (value <= best) return;
        best = value;
        g_incremental_publishes.fetch_add(1, std::memory_order_relaxed);
        if (shared_best) {
            bint cur(shared_best->load(std::memory_order_relaxed));
            while (cur < value &&
                   !shared_best->compare_exchange_weak(
                       cur, value, std::memory_order_relaxed,
                       std::memory_order_relaxed)) {}
        }
    }

    template <typename Basis>
    void visit(const size_t depth, Basis& basis) {
        ++states;
        const IncrementalState& state(stack[depth]);
        if (shared_best) {
            const bint g(shared_best->load(std::memory_order_relaxed));
            if (g > best) best = g;
        }
        if (basis.size() == k) {
            if (state.range > best) {
                publish_best(state.range);
                best_basis.assign(basis.data(), basis.data()+basis.size());
            }
            return;
        }

        const size_t remaining(k-basis.size());
        if (bound && IncrementalCompletionBound(state.range, remaining, h)<=best)
            return;

        const size_t low((size_t)basis.back()+1u);
        const size_t high((size_t)state.range+1u);
        const bool final_filter(target_filter && basis.size()+1u==k &&
                                best>state.range);
        const size_t target(final_filter ? (size_t)best+1u : 0u);
        // S11: reserve once per node from the widest candidate (descending
        // visits high first, but high is the max either way); skip entirely
        // at final depth where final_fast never extends.
        if (!(final_fast && basis.size()+1u==k)) {
            const size_t required_max((h+1u)*((h*high+64u)/64u));
            if (stack[depth+1u].bits.capacity() < required_max)
                stack[depth+1u].bits.reserve(required_max);
        }
        // S5: fused kernel needs a per-node view; build once here (amortized
        // over all candidates), and only when the final filter will use it.
        const bool fused_ok =
            (h < 16u && target_count >= 1u && target_count <= 64u);
        IncrementalPrefixView node_view;
        if (final_filter && fused_ok) node_view = IncrementalMakeView(state);
        const bool use_node_view = final_filter && fused_ok;
        auto visit_one = [&](const size_t a) {
            if (final_filter &&
                !IncrementalCanReachTargetsCall(state, node_view,
                                                use_node_view, a, target,
                                                target_count)) {
                ++target_skips;
                return;
            }
            basis.push_back(bint(a));
            if (final_fast && basis.size()==k) {
                const bint candidate_range(IncrementalFinalRange(
                    state, a, final_scratch));
#ifdef GSTAMPS_CENSUS
                if (t_incremental_census) {
                    ++t_incremental_census->finalrange_calls;
                    if (candidate_range <= best)
                        ++t_incremental_census->finalrange_losers;
                }
#endif
                if (candidate_range > best) {
                    publish_best(candidate_range);
                    best_basis.assign(basis.data(),
                                      basis.data()+basis.size());
                }
                basis.pop_back();
                return;
            }
            IncrementalExtend(state, a, stack[depth+1u]);
            if (pair_filter && target_filter && basis.size()+1u==k-1u &&
                best>state.range) {
                const IncrementalState& pair_prefix(stack[depth+1u]);
                const size_t pair_target((size_t)best+1u);
                const size_t pair_low(a+1u);
                const size_t pair_high((size_t)pair_prefix.range+1u);
                bool possible=false;
                // S5: one view per pair node, amortized over the b-loop.
                IncrementalPrefixView pair_view;
                const bool use_pair_view = fused_ok;
                if (use_pair_view)
                    pair_view = IncrementalMakeView(pair_prefix);
                size_t b = pair_low;
                for (; b<=pair_high && !possible; ++b)
                    possible=IncrementalCanReachTargetsCall(
                        pair_prefix, pair_view, use_pair_view, b,
                        pair_target, target_count);
#ifdef GSTAMPS_CENSUS
                if (t_incremental_census) {
                    ++t_incremental_census->pair_nodes;
                    // Exit value b counts tested iterations exactly:
                    // full scan -> high+1-low; early success at bv -> bv+1-low.
                    t_incremental_census->pair_b_tested += b - pair_low;
                    if (!possible) ++t_incremental_census->pair_pruned;
                }
#endif
                if (!possible) {
                    ++target_skips;
                    basis.pop_back();
                    return;
                }
            }
            visit(depth+1u, basis);
            basis.pop_back();
        };

        if (descending) {
            for (size_t a=high;; --a) {
                visit_one(a);
                if (a==low) break;
            }
        } else {
            for (size_t a=low; a<=high; ++a)
                visit_one(a);
        }
    }
};

struct IncrementalTask {
    std::vector<bint> basis;
    IncrementalState state;
};

struct IncrementalParallelResult {
    bint best = 0;
    std::vector<bint> best_basis;
    unsigned long long states = 0;
    unsigned long long target_skips = 0;
    size_t tasks = 0;
#ifdef GSTAMPS_CENSUS
    IncrementalCensus census;
#endif
};

inline void IncrementalGenerateTasks(IncrementalSearch& search,
                                     const size_t depth,
                                     const size_t split_depth,
                                     std::vector<bint>& basis,
                                     std::vector<IncrementalTask>& tasks) {
    if (depth == split_depth || basis.size() == search.k) {
        tasks.push_back({basis, search.stack[depth]});
        return;
    }
    const IncrementalState& state(search.stack[depth]);
    // Bound-aware: mirror visit() pruning so we never materialize tasks
    // under a subtree the serial search would prune immediately.
    if (search.bound &&
        IncrementalCompletionBound(state.range, search.k-basis.size(),
                                   search.h) <= search.best)
        return;
    const size_t low((size_t)basis.back()+1u);
    const size_t high((size_t)state.range+1u);
    // Descending (largest-first) generation: large early denominations own
    // larger subtrees, so scheduling them first (LPT) minimizes tail idle.
    for (size_t a=high;;) {
        basis.push_back(bint(a));
        IncrementalExtend(state, a, search.stack[depth+1u]);
        // Mirror visit() pair pruning so static generation never
        // materializes subtrees the serial search would discard.
        bool keep(true);
        if (keep && search.pair_filter && search.target_filter &&
            basis.size()+1u==search.k-1u && search.best>state.range) {
            const IncrementalState& pair_prefix(search.stack[depth+1u]);
            const size_t pair_target((size_t)search.best+1u);
            const size_t pair_low(a+1u);
            const size_t pair_high((size_t)pair_prefix.range+1u);
            bool possible=false;
            const bool gen_fused_ok =
                (search.h < 16u && search.target_count >= 1u &&
                 search.target_count <= 64u);
            IncrementalPrefixView gen_view;
            if (gen_fused_ok) gen_view = IncrementalMakeView(pair_prefix);
            for (size_t b=pair_low; b<=pair_high && !possible; ++b)
                possible=IncrementalCanReachTargetsCall(
                    pair_prefix, gen_view, gen_fused_ok, b,
                    pair_target, search.target_count);
            if (!possible) keep = false;
        }
        if (keep)
            IncrementalGenerateTasks(search, depth+1u, split_depth, basis, tasks);
        basis.pop_back();
        if (a==low) break;
        --a;
    }
}

inline IncrementalParallelResult IncrementalRunParallel(
    const size_t k, const size_t h, const IncrementalOptions& options) {
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

    std::vector<bint> basis{1};
    const size_t max_split_depth(k > 0u ? k-1u : 0u);
    const size_t split_depth(std::min(options.parallel_split_depth,
                                      max_split_depth));
    std::vector<IncrementalTask> tasks;
    IncrementalGenerateTasks(seed, 0, split_depth, basis, tasks);

    IncrementalParallelResult result;
    result.best = seed.best;
    result.best_basis = seed.best_basis;
    result.tasks = tasks.size();
    const int threads(std::max(1, options.parallel_threads));
    std::atomic<bint> global_best(seed.best);
#pragma omp parallel for schedule(dynamic,1) num_threads(threads)
    for (size_t ti=0; ti<tasks.size(); ++ti) {
        IncrementalSearch local{k,h,options.bound,options.descending,
                               options.target_filter,options.final_fast,
                               options.target_count,options.automatic,
                               options.pair_filter};
        local.best=seed.best;
        local.best_basis=seed.best_basis;
        local.shared_best = &global_best;
        local.stack.resize(k);
        const size_t depth(tasks[ti].basis.size()-1u);
        local.stack[depth]=tasks[ti].state;
#ifdef GSTAMPS_CENSUS
        IncrementalCensus thread_census;
        t_incremental_census = &thread_census;
#endif
        // S12: stack buffer for realistic k; vector fallback beyond.
        if (k <= 40) {
            bint stol[40];
            const size_t blen(tasks[ti].basis.size());
            std::copy(tasks[ti].basis.begin(), tasks[ti].basis.end(), stol);
            IncrementalBasisBuf local_basis;
            local_basis.p = stol;
            local_basis.n = blen;
            local.visit(depth,local_basis);
        } else {
            std::vector<bint> local_basis(tasks[ti].basis);
            local.visit(depth,local_basis);
        }
#pragma omp critical
        {
            result.states += local.states;
            result.target_skips += local.target_skips;
            if (local.best>result.best) {
                result.best=local.best;
                result.best_basis=local.best_basis;
            }
#ifdef GSTAMPS_CENSUS
            result.census.merge(thread_census);
#endif
        }
    }
    // In case another worker published a better value but lost the
    // critical race on best_basis, reconcile from the atomic.
    {
        const bint g(global_best.load(std::memory_order_relaxed));
        if (g > result.best) result.best = g;
    }
    return result;
}

int main(int argc, char** argv) {
    if (argc <= 2) {
        std::cerr << "usage: " << argv[0]
                  << " #k #h [auto|serial|parallel|bound] [descending] [seed]"
                  << " [target-filter] [final-fast] [target-count].\n"
                  << "without optional arguments, automatic exact settings are used.\n"
                  << "automatic mode parallelizes validated h=4,k>=8 searches;"
                  << " use serial to disable it.\n";
        return 1;
    }
    const size_t k(std::stoul(argv[1]));
    const size_t h(std::stoul(argv[2]));
    IncrementalOptions options;
    const std::string mode(argc>3 ? argv[3] : "auto");
    if (mode == "auto") {
        options = IncrementalAutomaticOptions(k, h);
    } else if (mode == "serial" || mode == "parallel") {
        options = IncrementalAutomaticOptions(k, h);
        options.parallel = mode == "parallel";
        options.automatic = true;
    } else {
        options.automatic = false;
        options.parallel = false;
        options.bound = std::stoi(argv[3]) != 0;
        options.descending = argc>4 ? std::stoi(argv[4])!=0 : false;
        options.seed = argc>5 ? std::stoi(argv[5])!=0 : false;
        options.target_filter = argc>6 ? std::stoi(argv[6])!=0 : false;
        options.final_fast = argc>7 ? std::stoi(argv[7])!=0 : false;
        options.target_count = argc>8 ? std::stoul(argv[8]) : 1u;
    }
    const bool pair_filter(argc>9 ? std::stoi(argv[9])!=0 : false);

    const auto start(std::chrono::steady_clock::now());
    if (options.parallel) {
        const IncrementalParallelResult result(
            IncrementalRunParallel(k, h, options));
        const auto stop(std::chrono::steady_clock::now());
        std::cout << "#[Incremental] range: " << result.best
                  << " policy: "
                  << (options.automatic ? "auto-parallel" : "parallel")
                  << " tasks: " << result.tasks
                  << " threads: " << std::max(1, options.parallel_threads)
                  << " states: " << result.states
                  << " target-skips: " << result.target_skips
                  << " seconds: "
                  << std::chrono::duration<double>(stop-start).count()
                  << " basis: ";
        for (const auto& value : result.best_basis) std::cout << value << ' ';
        std::cout << '\n';
        IncrementalPrintPublishes("parallel");
#ifdef GSTAMPS_CENSUS
        result.census.print("parallel");
#endif
    } else {
        IncrementalSearch search{k,h,options.bound,options.descending,
                                 options.target_filter,options.final_fast,
                                 options.target_count,options.automatic,
                                 options.pair_filter || pair_filter};
        search.stack.resize(k);
        search.stack[0] = IncrementalInitial(h);
        if (options.seed) {
            std::vector<bint> seed_basis;
            search.best = FSelect(seed_basis, k, h, 0, options.seed_approx, 0);
            search.best_basis = seed_basis;
        }
#ifdef GSTAMPS_CENSUS
        IncrementalCensus serial_census;
        t_incremental_census = &serial_census;
#endif
        // S12: stack buffer for realistic k; vector fallback beyond.
        if (k <= 40) {
            bint stol[40];
            stol[0] = 1;
            IncrementalBasisBuf basis;
            basis.p = stol;
            basis.n = 1;
            search.visit(0, basis);
        } else {
            std::vector<bint> basis{1};
            search.visit(0, basis);
        }
        const auto stop(std::chrono::steady_clock::now());

        std::cout << "#[Incremental] range: " << search.best
                  << " policy: "
                  << (options.automatic ? "auto-serial" : "manual")
                  << " states: " << search.states
                  << " target-skips: " << search.target_skips
                  << " seconds: "
                  << std::chrono::duration<double>(stop-start).count()
                  << " basis: ";
        for (const auto& value : search.best_basis) std::cout << value << ' ';
        std::cout << '\n';
        IncrementalPrintPublishes("serial");
#ifdef GSTAMPS_CENSUS
        if (t_incremental_census) t_incremental_census->print("serial");
#endif
    }
    return 0;
}
