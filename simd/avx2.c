// Original: Matt Scarpino

#include <x86intrin.h>
#include <stdio.h>
int main() {

	float int_array[8] = {100, 200, 300, 400, 500, 600, 700, 800};

	/* Initialize the mask vector */
	__m256i mask = _mm256_setr_epi32(-20, -72, -48, -9, -100, 3, 5, 8);

	/* Selectively load data into the vector */
	__m256 result = _mm256_maskload_ps(int_array, mask);

	/* Display the elements of the result vector */
	float* res = (float*)&result;
	printf("%f %f %f %f %f %f %f %f\n",
		res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]);

	return 0;
}

