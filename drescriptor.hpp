#ifndef DRESCRIPTOR_HPP
#define DRESCRIPTOR_HPP
#include <vpp/vpp.hh>


namespace vppx {

using namespace vpp;

template <typename C>
struct descriptor{
    descriptor() : age(0) {}
    descriptor(vector<C, 2> pos) : position(pos), velocity(0,0),
        age(1) {}

    vector<C, 2> position;
    vector<C, 2> velocity;
    int age;
    int nb_best_nbh;

    void die() { age = 0; }
    bool alive() { return age > 0; }
};

}


#endif // DRESCRIPTOR_HPP
