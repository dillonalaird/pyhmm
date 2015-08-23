#ifndef METAOBS_h
#define METAOBS_h

#include <iostream>
#include <random>


namespace mo {
    struct metaobs {
        int i1;
        int i2;
    };

    metaobs metaobs_unif(int T, int L) {
        int ll = L;
        int uu = T - 1 - L;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> distribution(ll, uu);
        int c = distribution(gen);
        return metaobs{c - L, c + L};
    }
}


#endif
