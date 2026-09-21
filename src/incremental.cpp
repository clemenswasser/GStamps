// ============================================================================
// GStamps: exact incremental-prefix exhaustive search
// ============================================================================

#include <gstamps.h>
#include <chrono>
#include <iostream>

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

// Add one denomination without recomputing the prefix from scratch.
inline void IncrementalExtend(const IncrementalState& parent, const size_t a,
                             IncrementalState& child) {
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

struct IncrementalSearch {
    size_t k;
    size_t h;
    bool bound;
    bool descending;
    bint best = 0;
    std::vector<bint> best_basis;
    unsigned long long states = 0;
    std::vector<IncrementalState> stack;

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
        auto visit_one = [&](const size_t a) {
            basis.push_back(bint(a));
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
                  << " #k #h [bound] [descending] [seed].\n";
        return 1;
    }
    const size_t k(std::stoul(argv[1]));
    const size_t h(std::stoul(argv[2]));
    const bool bound(argc>3 ? std::stoi(argv[3])!=0 : false);
    const bool descending(argc>4 ? std::stoi(argv[4])!=0 : false);
    const bool seed(argc>5 ? std::stoi(argv[5])!=0 : false);

    IncrementalSearch search{k,h,bound,descending};
    search.stack.resize(k);
    search.stack[0] = IncrementalInitial(h);
    std::vector<bint> basis{1};
    if (seed) {
        std::vector<bint> seed_basis;
        search.best = FSelect(seed_basis, k, h, 0, false, 0);
        search.best_basis = seed_basis;
    }
    const auto start(std::chrono::steady_clock::now());
    search.visit(0, basis);
    const auto stop(std::chrono::steady_clock::now());

    std::cout << "#[Incremental] range: " << search.best
              << " states: " << search.states
              << " seconds: "
              << std::chrono::duration<double>(stop-start).count()
              << " basis: ";
    for (const auto& value : search.best_basis) std::cout << value << ' ';
    std::cout << '\n';
    return 0;
}
