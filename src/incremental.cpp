// ============================================================================
// GStamps: exact incremental-prefix exhaustive search
// ============================================================================

#include <gstamps.h>
#include <chrono>
#include <iostream>
#include <string>

struct IncrementalOptions {
    bool automatic = false;
    bool bound = false;
    bool descending = false;
    bool seed = false;
    bool seed_approx = false;
    bool target_filter = false;
    bool final_fast = false;
    size_t target_count = 1;
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
    options.target_count = (h == 4u) ? 4u : 1u;
    return options;
}

struct IncrementalState {
    size_t h = 0;
    size_t max_value = 0;
    size_t words = 0;
    std::vector<uint64_t> bits;
    bint range = 0;
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
    if (target <= prefix.max_value &&
        ((prefix.bits[prefix.h*prefix.words+(target>>6)] >>
          (target&63u)) & 1u))
        return true;
    for (size_t copies=1; copies<=prefix.h; ++copies) {
        if (copies*a > target) break;
        const size_t remainder(target-copies*a);
        const size_t depth(prefix.h-copies);
        if (remainder > depth*prefix.max_value) continue;
        if ((prefix.bits[depth*prefix.words+(remainder>>6)] >>
             (remainder&63u)) & 1u)
            return true;
    }
    return false;
}

inline bool IncrementalCanReachTargetsWithLast(const IncrementalState& prefix,
                                               const size_t a,
                                               const size_t first_target,
                                               const size_t target_count) {
    for (size_t offset=0; offset<target_count; ++offset) {
        if (!IncrementalCanReachTargetWithLast(prefix, a,
                                               first_target+offset))
            return false;
    }
    return true;
}

inline bint IncrementalFinalRange(const IncrementalState& prefix,
                                  const size_t a,
                                  std::vector<uint64_t>& scratch) {
    const size_t words((prefix.h*a+64u)/64u);
    scratch.assign(words, 0u);
    const size_t copy_words(std::min(prefix.words, words));
    std::copy(prefix.bits.begin()+prefix.h*prefix.words,
              prefix.bits.begin()+prefix.h*prefix.words+copy_words,
              scratch.begin());
    for (size_t copies=1; copies<=prefix.h; ++copies) {
        const size_t shift(copies*a);
        const size_t word_shift(shift>>6);
        const unsigned bit_shift((unsigned)(shift&63u));
        const unsigned reverse_shift(64u-bit_shift);
        const uint64_t* source(prefix.bits.data()+(prefix.h-copies)*prefix.words);
        for (size_t i=word_shift; i<words; ++i) {
            if (i-word_shift>=prefix.words) break;
            uint64_t shifted(source[i-word_shift]<<bit_shift);
            if (bit_shift && i>word_shift && i-word_shift-1u<prefix.words)
                shifted |= source[i-word_shift-1u]>>reverse_shift;
            scratch[i] |= shifted;
        }
    }
    return IncrementalFirstZeroFrom(scratch.data(), prefix.h*a+1u,
                                    (size_t)prefix.range+1u)-1;
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
    bint best = 0;
    std::vector<bint> best_basis;
    unsigned long long states = 0;
    unsigned long long target_skips = 0;
    std::vector<IncrementalState> stack;
    std::vector<uint64_t> final_scratch;

    void visit(const size_t depth, std::vector<bint>& basis) {
        ++states;
        const IncrementalState& state(stack[depth]);
        if (basis.size() == k) {
            if (state.range > best) {
                best = state.range;
                best_basis = basis;
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
        auto visit_one = [&](const size_t a) {
            if (final_filter &&
                !IncrementalCanReachTargetsWithLast(state, a, target,
                                                    target_count)) {
                ++target_skips;
                return;
            }
            basis.push_back(bint(a));
            if (final_fast && basis.size()==k) {
                const bint candidate_range(IncrementalFinalRange(
                    state, a, final_scratch));
                if (candidate_range > best) {
                    best = candidate_range;
                    best_basis = basis;
                }
                basis.pop_back();
                return;
            }
            const size_t required((h+1u)*((h*a+64u)/64u));
            if (stack[depth+1u].bits.capacity() < required)
                stack[depth+1u].bits.reserve(required);
            IncrementalExtend(state, a, stack[depth+1u]);
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

int main(int argc, char** argv) {
    if (argc <= 2) {
        std::cerr << "usage: " << argv[0]
                  << " #k #h [auto|bound] [descending] [seed]"
                  << " [target-filter] [final-fast] [target-count].\n"
                  << "without optional arguments, automatic exact settings are used.\n";
        return 1;
    }
    const size_t k(std::stoul(argv[1]));
    const size_t h(std::stoul(argv[2]));
    IncrementalOptions options;
    if (argc <= 3 || std::string(argv[3]) == "auto") {
        options = IncrementalAutomaticOptions(k, h);
    } else {
        options.automatic = false;
        options.bound = std::stoi(argv[3]) != 0;
        options.descending = argc>4 ? std::stoi(argv[4])!=0 : false;
        options.seed = argc>5 ? std::stoi(argv[5])!=0 : false;
        options.target_filter = argc>6 ? std::stoi(argv[6])!=0 : false;
        options.final_fast = argc>7 ? std::stoi(argv[7])!=0 : false;
        options.target_count = argc>8 ? std::stoul(argv[8]) : 1u;
    }

    IncrementalSearch search{k,h,options.bound,options.descending,
                             options.target_filter,options.final_fast,
                             options.target_count,options.automatic};
    search.stack.resize(k);
    search.stack[0] = IncrementalInitial(h);
    std::vector<bint> basis{1};
    if (options.seed) {
        std::vector<bint> seed_basis;
        search.best = FSelect(seed_basis, k, h, 0, options.seed_approx, 0);
        search.best_basis = seed_basis;
    }
    const auto start(std::chrono::steady_clock::now());
    search.visit(0, basis);
    const auto stop(std::chrono::steady_clock::now());

    std::cout << "#[Incremental] range: " << search.best
              << " policy: " << (options.automatic ? "auto" : "manual")
              << " states: " << search.states
              << " target-skips: " << search.target_skips
              << " seconds: "
              << std::chrono::duration<double>(stop-start).count()
              << " basis: ";
    for (const auto& value : search.best_basis) std::cout << value << ' ';
    std::cout << '\n';
    return 0;
}
