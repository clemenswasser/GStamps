// ==========================================================================
// GStamps: C++ routines for the Local & the Global Postage Stamp Problem
// Authors:
//   L. Colisson, J-G. Dumas, A. Galan, B. Grenet, A. Maignan, D. S. Roche
// ==========================================================================

/****************************************************************
 * GStamps Library, LPSP inline implementations
 ****************************************************************/

#ifdef __AVX2__
#include <immintrin.h>
#endif

// ============================================
// Masking Tools
// upmask: Round up to the next (highest power of 2, minus 1) of (input+1)

#ifdef __GSTAMPS_EXTENDED_PRECISION
Givaro::Integer upmask(const Givaro::Integer& w) {
    Givaro::Integer v(w);
    uint32_t exp(1);
    for(Givaro::Integer shi(1); shi>0; exp <<=1) {
        shi = v >> exp;
        v |= shi;
    }
    return std::move(v);
}

#else
// See: https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
uint64_t upmask(const uint64_t& w) {
    bint v(w);
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return std::move(v);
}
#endif



// ============================================
// Range: local postage stamp problem

// Loop from 1 to s, with a binary vector
template<typename Iterator, typename stype_t>
inline bint _SRange(const Iterator& start, const Iterator& end,
                    const stype_t s, const int verbose) {
	// Binary range
    const bint& back(*std::prev(end));				// k>=1
    if (back == __St_One) return s;
    bint vs(back);
    const bint upper(s*vs+1);
    boost::dynamic_bitset<> reached(upper,false);
    reached[0]=true;                                // 0 reached
    for(auto it=start; it!=end; ++it) reached[*it]=true;   // points reached

#if __GSTAMPS_SELMER_LEMMA
    const bint& penult(*std::prev(std::prev(end))); // back>1 => k>=2
    bint lb(penult-back);
#endif

    bint notin(1);
    for(stype_t d(1); d<s; ++d, vs += back) {
#if __GSTAMPS_SELMER_LEMMA
        notin = vs+1;
#endif
        for(size_t i=vs; i>=d; --i) {
            if(reached[i]) {
                for(auto right=start; right!=end; ++right) {
                    reached[i+(*right)]=true;
                }
            }
#if __GSTAMPS_SELMER_LEMMA
            else {
                notin=i;
            }
#endif
        }

#if __GSTAMPS_SELMER_LEMMA
            // Test Selmer's Lemma for early termination
        lb += penult;
        if ( (notin > back) && (notin > lb) ) {
                    // Range will now surely attain c+(s-d)ak
            if (verbose>0) std::clog << "#[ET(" << (size_t)d << '|' << lb
                                     << ")]: " << notin << " --> "
                                     << (notin-1+(s-d)*back) << std::endl;
            return --notin += (s-d)*back;
        }
#endif
    }

    size_t max(reached.size()-1);
    for(size_t jr(notin); jr<reached.size(); ++jr){
        if (! reached[jr]) {
            max = jr-1;
            break;
        }
    }

    return max;
}


// Loop from 1 to n -- data-oriented AoS version (2 bytes/cell when
// stype_t is uint8_t). Single allocation, raw pointer, size_t loop
// counters (fixes stype_t wrap for k>255 in the loop condition).
// For k>255 with 8-bit stype_t the argmin would truncate: use wide path.
template<typename List, typename stype_t>
inline bint _KRangeWide(const List& points, const size_t k, const stype_t s,
                        const int verbose);
template<typename List, typename stype_t>
inline bint _KRange(const List& points, const size_t k, const stype_t s,
                    const int verbose) {
    assert( (k>=1) && (k<=points.size()) );
    if ((sizeof(stype_t)==1u) && (k>255u))
        return _KRangeWide(points, k, s, verbose);
    const bint back(points.back());				// k>=1
    if (back == __St_One) return s;

    const size_t window(upmask((uint64_t)back));	// highest 1-full mask gt
    const stype_t spu(s+1);				// s+1 is unreachable
    using Cell = std::pair<stype_t,stype_t>;
    std::vector<Cell> reached(window+1u, std::make_pair(spu,(stype_t)0u));
    Cell* __restrict__ rp(reached.data());

    for(size_t i=0; i<k; ++i) {
        const size_t v((size_t)points[i]);
        rp[v].first = (stype_t)1u;
        rp[v].second = (stype_t)i;
    }

#if __GSTAMPS_SELMER_LEMMA > 1
    const auto& penult(points[k-2]); // back>1 => k>=2
    std::vector<size_t> selmer(s+1);
    std::iota(selmer.begin(), selmer.end(), 1);
    stype_t maxs(0), mins(0);
    for(auto& is: selmer) {
        is *= penult;
        if (is <= back) ++mins;
        is -= back;
    }
#endif

    size_t index(1);
    for(; rp[index & window].first<=s; ++index) {
        const size_t wcur(index & window);
        const stype_t slfirst(rp[wcur].first);
        const stype_t vlocal(slfirst+1u);
        const size_t rstart((size_t)rp[wcur].second);
        for(size_t right=rstart; right<k; ++right) {
            const size_t wt((index+(size_t)points[right]) & window);
            if (rp[wt].first > vlocal) {
                rp[wt].first = vlocal;
                rp[wt].second = (stype_t)right;
            }
        }

#if __GSTAMPS_SELMER_LEMMA > 1
        maxs = (slfirst>maxs?slfirst:maxs);
        if ( (index & 1048575u) == 1048575u) { // Reduce overhead
                // Selmer's lemma
            if ((maxs>mins) && (maxs<s) && (index > selmer[maxs])) {
                    // Find maxs_range
                for(size_t i=1; i<=(size_t)back; ++i) {
                        // Complete s_range
                    if (rp[(index+i) & window].first>maxs) {
                        if (verbose>0) std::clog << "#[ET(" << (size_t)maxs
                                                 << '|' << selmer[maxs] << ")]: "
                                                 << i << " -> "
                                                 << (index+i-1+(s-maxs)*back)
                                                 << std::endl;
                        index += i;
                        return --index += (s-maxs)*back;
                    }
                }
            }
        }
#endif
        rp[wcur].first=spu;		// clean up sliding window
    }

    return --index;

}

// Wide-argmin fallback for k>255 with 8-bit stype_t (rare): 32-bit index.
template<typename List, typename stype_t>
inline bint _KRangeWide(const List& points, const size_t k, const stype_t s,
                        const int verbose) {
    assert( (k>=1) && (k<=points.size()) );
    const bint back(points.back());
    if (back == __St_One) return s;
    const size_t window(upmask((uint64_t)back));
    const stype_t spu(s+1);
    std::vector<stype_t> best(window+1u, spu);
    std::vector<uint32_t> arg(window+1u, 0u);
    stype_t* __restrict__ bestp(best.data());
    uint32_t* __restrict__ argp(arg.data());
    for(size_t i=0; i<k; ++i) {
        const size_t v((size_t)points[i]);
        bestp[v] = (stype_t)1u;
        argp[v] = (uint32_t)i;
    }
    size_t index(1);
    for(; bestp[index & window]<=s; ++index) {
        const size_t wcur(index & window);
        const stype_t vlocal(bestp[wcur]+1u);
        const size_t rstart(argp[wcur]);
        for(size_t right=rstart; right<k; ++right) {
            const size_t wt((index+(size_t)points[right]) & window);
            if (bestp[wt] > vlocal) { bestp[wt]=vlocal; argp[wt]=(uint32_t)right; }
        }
        bestp[wcur]=spu;
    }
    (void)verbose;
    return --index;
}

template<typename List, typename stype_t>
inline bint _KRange(const List& points, const stype_t s,
                    const int verbose) {
    return _KRange(points,points.size(),s,verbose);
}

// ============================================
// Bitset shift-OR Range: bits_{d+1} = bits_d | OR_a(bits_d << a).
// Word-streaming, AVX2 4x64b when available. Wins for small s (s<=5):
// O(s*k*N/64) sequential vs _KRange O(N*k) scattered. Loses for large s
// where upper=s*back >> range (processes full upper each depth).
// Falls back to _KRange when AVX2 unavailable or upper too large.
template<typename List, typename stype_t>
inline bint _BRange(const List& points, const size_t k, const stype_t s,
                    const int verbose); // fwd for tiny fallback
// E2: dedicated tiny path (nwords<=8, k<=256): fused copy+first shift,
// hoisted decode, dense scalar fully unrolled. Generic path keeps AVX2.
template<typename List>
inline bint _BRangeTiny(const List& points, const size_t k,
                        const size_t us, const size_t upper,
                        const size_t nwords) {
    uint64_t cur[8], nxt[8];
    for(size_t i=0; i<nwords; ++i) cur[i] = 0u;
    cur[0] = 1ULL;
    // K2: fused bit-set + denomination decode (was two passes over k).
    size_t wsA[256]; unsigned bsA[256]; unsigned rbA[256];
    for(size_t j=0; j<k; ++j) {
        const size_t v((size_t)points[j]);
        if (v < upper) cur[v>>6] |= (1ULL<<(v&63u));
        const size_t a(v);
        wsA[j] = (a>>6); bsA[j] = (unsigned)(a&63u); rbA[j] = 64u-bsA[j];
    }
    // K1: pointer-swapped buffers (copy-back loop was 7.6% profile).
    uint64_t *cp(cur), *np(nxt);
    for(size_t d=1; d<us; ++d) {
        // Fused copy + first denomination (E1 for tiny included here).
        {
            const size_t ws(wsA[0]); const unsigned bs(bsA[0]);
            if (bs == 0) {
#pragma GCC unroll 8
                for(size_t i=0; i<nwords; ++i)
                    np[i] = cp[i] | (i>=ws ? cp[i-ws] : 0u);
            } else {
                const unsigned rb(rbA[0]);
#pragma GCC unroll 8
                for(size_t i=0; i<nwords; ++i) {
                    uint64_t sh(0u);
                    if (i>=ws) {
                        sh = cp[i-ws]<<bs;
                        if (i>ws) sh |= (cp[i-ws-1u]>>rb);
                    }
                    np[i] = cp[i] | sh;
                }
            }
        }
        for(size_t j=1; j<k; ++j) {
            const size_t ws(wsA[j]); const unsigned bs(bsA[j]);
            if (ws >= nwords) continue;
            if (bs == 0) {
#pragma GCC unroll 8
                for(size_t i=ws; i<nwords; ++i) np[i] |= cp[i-ws];
            } else {
                const unsigned rb(rbA[j]);
                if (ws < nwords) np[ws] |= (cp[0]<<bs);
#pragma GCC unroll 8
                for(size_t i=ws+1u; i<nwords; ++i)
                    np[i] |= (cp[i-ws]<<bs) | (cp[i-ws-1u]>>rb);
            }
        }
        { uint64_t* t(cp); cp = np; np = t; }
    }
    const size_t full(upper>>6);
    for(size_t w=0; w<full; ++w) {
        const uint64_t inv(~cp[w]);
        if (inv) return (bint)(w*64u + __builtin_ctzll(inv)) - 1;
    }
    const unsigned rem((unsigned)(upper&63u));
    if (rem) {
        const uint64_t inv(~(cp[full] | (~0ULL << rem)));
        if (inv) return (bint)(full*64u + __builtin_ctzll(inv)) - 1;
    }
    return (bint)upper - 1;
}
template<typename List, typename stype_t>
inline bint _BRange(const List& points, const size_t k, const stype_t s,
                    const int verbose) {
    assert( (k>=1) && (k<=points.size()) );
    const bint back(points.back());
    if (back == __St_One) return s;
    // Guard: need upper=s*back+1 bits. Fallback for huge/negative.
    if (back <= 0) return _KRange(points, k, s, verbose);
    // A4: cheap pre-guard -- 128-bit mul was ~300 samples/call hot.
    // back<=1M and s<=255 implies upper<=255Mb < cap; skip __int128.
    if (!((back <= bint(1<<20)) && ((size_t)s <= 255u))) {
    // Avoid overflow: use unsigned __int128 for upper check
    {
        unsigned __int128 ub = (unsigned __int128)(uint64_t)s
                             * (unsigned __int128)(uint64_t)back + 1u;
        // Cap at 256M bits (32MB per bitset x2 = 64MB); beyond that the
        // streaming passes cost more than _KRange's early-exit scan.
        if (ub > (unsigned __int128)(256u*1024u*1024u))
            return _KRange(points, k, s, verbose);
    }
    }
    const size_t uback((size_t)back);
    const size_t us((size_t)s);
    const size_t upper(us*uback+1u);
    const size_t nwords((upper+63u)/64u);
    // E2: tiny fast path (brute prefixes live here).
    if ((nwords <= 8u) && (k <= 256u))
        return _BRangeTiny(points, k, us, upper, nwords);
    const size_t nvec4((nwords+3u)/4u);
    const size_t nwords4(nvec4*4u);
    // A1: SBO -- brute prefixes are tiny (upper~200 bits = 4 words);
    // two heap vectors per Range (~100ns malloc/free each) dominate.
    // Keep bitsets on stack up to 64 words (512B x2 = 1KB).
    constexpr size_t SBO_WORDS = 64u;
    std::vector<uint64_t> curH, nxtH;
    uint64_t curS[SBO_WORDS], nxtS[SBO_WORDS];
    const bool heap(nwords4 > SBO_WORDS);
    uint64_t *cbuf(heap ? nullptr : curS);
    uint64_t *nbuf(heap ? nullptr : nxtS);
    if (heap) {
        curH.assign(nwords4, 0u); nxtH.assign(nwords4, 0u);
        cbuf = curH.data(); nbuf = nxtH.data();
    } else {
        for(size_t i=0; i<nwords4; ++i) cbuf[i] = 0u;
    }
    cbuf[0] = 1ULL;
    for(size_t j=0; j<k; ++j) {
        const size_t v((size_t)points[j]);
        if (v < upper) cbuf[v>>6] |= (1ULL<<(v&63u));
    }
    uint64_t* __restrict__ cp(cbuf);
    uint64_t* __restrict__ np(nbuf);
    for(size_t d=1; d<us; ++d) {
        std::copy(cp, cp+nwords4, np);
        // Early depths are sparse: skip zero words (scalar, predictable).
        // Later depths are dense: AVX2 streaming (no branches).
        // A5: tiny bitsets (nwords<=8, one AVX2 vector) go dense always:
        // the sparse branch mispredicts dominate (L305 was 24% profile).
        const bool sparse((d <= 2u) && (nwords > 8u));
        for(size_t j=0; j<k; ++j) {
            const size_t a((size_t)points[j]);
            const size_t ws(a>>6);
            if (ws >= nwords) continue;
            const unsigned bs((unsigned)(a&63u));
            if (bs == 0) {
                size_t i(ws);
                if (sparse) {
                    for(; i<nwords; ++i) {
                        const uint64_t c(cp[i-ws]);
                        if (c) np[i] |= c;
                    }
                    continue;
                }
#ifdef __AVX2__
                for(; i<nwords && (i&3u); ++i) np[i] |= cp[i-ws];
                for(; i+4<=nwords; i+=4) {
                    __m256i v = _mm256_loadu_si256((const __m256i*)(cp+i-ws));
                    __m256i w = _mm256_loadu_si256((const __m256i*)(np+i));
                    w = _mm256_or_si256(w, v);
                    _mm256_storeu_si256((__m256i*)(np+i), w);
                }
#endif
                for(; i<nwords; ++i) np[i] |= cp[i-ws];
            } else {
                if (ws < nwords) np[ws] |= (cp[0]<<bs);
                const unsigned rbs(64u-bs);
                size_t i(ws+1u);
                if (sparse) {
                    for(; i<nwords; ++i) {
                        const uint64_t c0(cp[i-ws]), c1(cp[i-ws-1u]);
                        if (c0 | c1) np[i] |= (c0<<bs) | (c1>>rbs);
                    }
                    continue;
                }
#ifdef __AVX2__
                for(; i<nwords && (i&3u); ++i)
                    np[i] |= (cp[i-ws]<<bs) | (cp[i-ws-1u]>>rbs);
                for(; i+4<=nwords; i+=4) {
                    __m256i v0 = _mm256_loadu_si256((const __m256i*)(cp+i-ws));
                    __m256i v1 = _mm256_loadu_si256((const __m256i*)(cp+i-ws-1u));
                    __m256i lo = _mm256_slli_epi64(v0, bs);
                    __m256i hi = _mm256_srli_epi64(v1, rbs);
                    __m256i sh = _mm256_or_si256(lo, hi);
                    __m256i w = _mm256_loadu_si256((const __m256i*)(np+i));
                    w = _mm256_or_si256(w, sh);
                    _mm256_storeu_si256((__m256i*)(np+i), w);
                }
#endif
                for(; i<nwords; ++i)
                    np[i] |= (cp[i-ws]<<bs) | (cp[i-ws-1u]>>rbs);
            }
        }
        std::swap(cp, np); // raw storage is fixed (stack or heap vectors)
    }
    // A3: word-level first-zero scan -- bit-by-bit loop was 22% of
    // brute profile (shift+and per bit). Scan full words, ctz the rest.
    {
        const uint64_t* bits(cp);
        const size_t full(upper>>6);
        for(size_t w=0; w<full; ++w) {
            const uint64_t inv(~bits[w]);
            if (inv) return (bint)(w*64u + __builtin_ctzll(inv)) - 1;
        }
        const unsigned rem((unsigned)(upper&63u));
        if (rem) {
            const uint64_t last(bits[full] | (~0ULL << rem));
            const uint64_t inv(~last);
            if (inv) return (bint)(full*64u + __builtin_ctzll(inv)) - 1;
            return (bint)upper - 1;
        }
        (void)verbose;
        return (bint)upper - 1;
    }
}

template<typename List, typename stype_t>
inline bint _BRange(const List& points, const stype_t s,
                    const int verbose) {
    return _BRange(points, points.size(), s, verbose);
}

// Prefix-aware dispatch (for FixedPoints prefix queries): _BRange for s<=5.
template<typename List, typename stype_t>
inline bint KDispatch(const List& points, const size_t k, const stype_t s,
                      const int verbose) {
    return (((size_t)s<=5u) ? _BRange(points, k, s, verbose)
                            : _KRange(points, k, s, verbose));
}


template<typename List, typename stype_t>
inline bint Range(const List& points, const stype_t s, const int verbose) {
    if (verbose>1)
        ScopePrint(std::clog << "#[Range] Basis: ", points) << std::endl;

    StTimer chrono; chrono.start();
    // Dispatch (measured on i7-12700K, Balanced k=40):
    // s=2: B 5.6x over K, 2.9x over S; s=3: B 4.5x; s=4: B 2.4x;
    // s=5: B 1.25x; s>=6: K wins (0.76x, 0.25x). _SRange kept for reference.
    const bint max( ((size_t)s<=5u) ?
                    _BRange(points, s, verbose) :
                    _KRange(points, s, verbose)
                    );
    chrono.stop();

    if (verbose>1) {
        std::clog << "#[Range(" << size_t(s) << ")]: 1.." << max
                  << " ..." << std::endl;
    }

    if (verbose>0) std::clog << "#[Range(" << size_t(s) << ")]: " << max
                             << ' ' << chrono <<std::endl;
    return max;
}




// ==========================================================================
// Reads a basis from std::cin
// Computes the range and decompositions that basis with argv[1] stamps
// Modification of the _KRange algorithm without the sliding window
template<typename List, typename stype_t>
std::vector<PsVs<stype_t>> _Decompose(const List& points, const stype_t s,
                                      const int verbose) {
    const auto& back(points.back());				// k>=1
    const stype_t spu(s+1);				// s+1 is unreachable

    if (back == __St_One) {
        std::vector<PsVs<stype_t>> reached(spu);
        for(size_t i(1); i<spu; ++i) {
            reached[i]={0,std::vector<stype_t>(i,0)};
        }
        return reached;
    }

    const PsVs<stype_t> spustart{spu, {0u}};
        // Maximal valid index is s*back
        //   thus maximal tested in loop is s*back+1
        //   and maximal starget is at s*back+back=spu*back>=s*back+1
        //   with 0 indexing this gives a table of size: spu*back+1
    std::vector<PsVs<stype_t>> reached(spu*back+1,spustart);

    for(stype_t i=0; i<points.size(); ++i)
        reached[points[i]]= PsVs<stype_t>{1u,{i}};

    size_t index(1);
    for(; reached[index].first<=s; ++index) {
        const auto& slocal(reached[index]);
        const stype_t slfirst(slocal.first);
        const stype_t vlocal(slfirst+1u);
        for(auto right=slocal.second.back(); right<points.size(); ++right) {
            auto& starget(reached[index+points[right]]);
            if (starget.first>vlocal) {
                starget.first = vlocal;
                starget.second.resize(0);
                starget.second.assign(slocal.second.begin(),
                                      slocal.second.end());
                starget.second.push_back(right);
            }
        }

    }

    reached.resize(index); // range is --index

    return reached;
}
// ==========================================================================
