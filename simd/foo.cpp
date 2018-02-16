// Original: Nicolas Brailovsky

#define SIZE (5)
long sum(int v[SIZE]) throw() {
    long s = 0;
    for (unsigned i=0; i<SIZE; i++) s += v[i];
    return s;
}
