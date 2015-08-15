#ifndef METAOBS_h
#define METAOBS_h


#include <random>


namespace mo {
    struct metaobs {
        int i1;
        int i2;
    };

    metaobs metaobs_unif(int T, int L) {
        int ll = L;
        int uu = T - 1 - L;

        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(ll, uu+1);
        int c = distribution(generator);
        return metaobs{c - L, c + L};
    }
}


#endif
