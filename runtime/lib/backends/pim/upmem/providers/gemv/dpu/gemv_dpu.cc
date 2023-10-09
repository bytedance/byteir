/*
 * Matrix vector multiplication with multiple tasklet
 *
 */
#include <stdint.h>
#include <stdio.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>

#include "./common.h"

#define roundup(n, m) ((n / m) * m + m)

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

// GEMV
static void gemv(T *bufferC, T *bufferA, T *bufferB, int pos) {
	for (unsigned int i = 0; i < BLOCK_SIZE / sizeof(T); i++) {
		bufferC[pos] += bufferA[i] * bufferB[i];
	}
	return;
}

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

// main
int main() {
	unsigned int tasklet_id = me();

	if (tasklet_id == 0){ // Initialize once the cycle counter
		mem_reset(); // Reset the heap
	}
	// Barrier
	barrier_wait(&my_barrier);

	int32_t n_size = DPU_INPUT_ARGUMENTS.n_size;
	int32_t n_size_pad = DPU_INPUT_ARGUMENTS.n_size_pad;
	uint32_t nr_rows = DPU_INPUT_ARGUMENTS.nr_rows;
	uint32_t max_rows = DPU_INPUT_ARGUMENTS.max_rows;

	unsigned int element_per_cacheC = 8/sizeof(T);

	unsigned int nrows = nr_rows;
	unsigned int rows_per_tasklet; 
	unsigned int start_row;
	unsigned int chunks = nrows / (NR_TASKLETS * element_per_cacheC);
	unsigned int dbl_chunks = chunks * element_per_cacheC; //chunks + chunks; 
	rows_per_tasklet = dbl_chunks;
	unsigned int rest_rows = nrows % (NR_TASKLETS * element_per_cacheC); //(NR_TASKLETS + NR_TASKLETS);

	if ((tasklet_id * element_per_cacheC) < rest_rows)
		rows_per_tasklet += element_per_cacheC;
	if (rest_rows > 0) {
		if ((tasklet_id * element_per_cacheC) >= rest_rows) {
			// unsigned int hlf_rest_rows = rest_rows >> 1;
			if ((rest_rows % element_per_cacheC) != 0)
				start_row = roundup(rest_rows, element_per_cacheC) + tasklet_id * dbl_chunks; 
				// start_row = (hlf_rest_rows + 1) * (dbl_chunks + 2) + (tasklet_id - 1 - hlf_rest_rows) * dbl_chunks;
			else
				start_row = rest_rows + tasklet_id * dbl_chunks; 
				// start_row = (hlf_rest_rows) * (dbl_chunks + 2) + (tasklet_id - hlf_rest_rows) * dbl_chunks;
		} else 
			start_row = tasklet_id * (dbl_chunks + element_per_cacheC);
			// start_row = tasklet_id * (dbl_chunks + 2);
	} else {
		start_row = tasklet_id * (dbl_chunks);
	}

	// Address of the current row in MRAM
	uint32_t mram_base_addr_A = (uint32_t) (DPU_MRAM_HEAP_POINTER + start_row * n_size * sizeof(T));
	uint32_t mram_base_addr_B = (uint32_t) (DPU_MRAM_HEAP_POINTER + max_rows * n_size_pad * sizeof(T));
	uint32_t mram_base_addr_C = (uint32_t) (DPU_MRAM_HEAP_POINTER + max_rows * n_size_pad * sizeof(T) + n_size_pad * sizeof(T) + start_row * sizeof(T));
	uint32_t mram_temp_addr_A = mram_base_addr_A;
	uint32_t mram_temp_addr_B = mram_base_addr_B;

	// Inititalize a local cache to store the MRAM block
	T *cache_A = (T *) mem_alloc(BLOCK_SIZE + 8);
	T *cache_A_aux = (T *) mem_alloc(8);
	T *cache_B = (T *) mem_alloc(BLOCK_SIZE);
	T *cache_C = (T *) mem_alloc(8);

	int offset = 0;


	// Iterate over nr_rows
	// for (unsigned int i = start_row; i < start_row + rows_per_tasklet; i += 2) {
	for (unsigned int i = start_row; i < start_row + rows_per_tasklet; i += element_per_cacheC) {

		mram_temp_addr_A = (uint32_t) (DPU_MRAM_HEAP_POINTER + i * n_size * sizeof(T));
		mram_temp_addr_B = mram_base_addr_B;

		// cache_C[0] = 0;
		// cache_C[1] = 0;

		// clear the cache
		for(unsigned int c = 0; c < element_per_cacheC; c++){
			cache_C[c] = 0; 
		}

		// for(unsigned int pos = 0; pos < 2 && i + pos < nr_rows; pos++){
		// for(unsigned int pos = 0; (pos < element_per_cacheC) && ((i + pos) < (start_row + rows_per_tasklet)); pos++){
		// for(unsigned int pos = 0; pos < element_per_cacheC && i + pos < nr_rows; pos++){ 
		for(unsigned int pos = 0; pos < element_per_cacheC; pos++){ 
			if(i + pos >= nr_rows){
				// printf("id: %d, nrows: %d, error\n", tasklet_id, nrows);
				break;
			} 

			int n = 0, j;
			for (n = 0; n < (int32_t) (n_size - (BLOCK_SIZE/sizeof(T))); n += (BLOCK_SIZE / sizeof(T)))
			{

				mram_read((__mram_ptr void const*) (mram_temp_addr_A), cache_A, BLOCK_SIZE);
				mram_read((__mram_ptr void const*) (mram_temp_addr_B), cache_B, BLOCK_SIZE);

				if(offset)
				{

					for(unsigned int off = 0; off < (BLOCK_SIZE / sizeof(T)) - 1; off++)
					{
						cache_A[off] = cache_A[off + 1];
					}

					mram_read((__mram_ptr void const*) (mram_temp_addr_A + BLOCK_SIZE), cache_A_aux, 8);

					cache_A[BLOCK_SIZE / sizeof(T) - 1] = cache_A_aux[0];
				}

				// Compute GEMV
				gemv(cache_C, cache_A, cache_B, pos);

				// Update memory addresses
				mram_temp_addr_A += BLOCK_SIZE;
				mram_temp_addr_B += BLOCK_SIZE;
			}

			mram_read((__mram_ptr void const*) (mram_temp_addr_A), cache_A, BLOCK_SIZE);


			if(offset)
			{
				for(unsigned int off = 0; off < (BLOCK_SIZE / sizeof(T)) -1; off++)
				{

					cache_A[off] = cache_A[off + 1];
				}

				mram_read((__mram_ptr void const*) (mram_temp_addr_A + BLOCK_SIZE ), cache_A_aux, 8);

  			       cache_A[BLOCK_SIZE / sizeof(T) - 1] = cache_A_aux[0];
			}


			mram_read((__mram_ptr void const*) (mram_temp_addr_B), cache_B, BLOCK_SIZE);

			for (j = 0; j < (int) (n_size - n); j++) {
				// Compute GEMV
				if(j >= (int)(BLOCK_SIZE / sizeof(T))){ 
					printf("error\n");
					break;
				}
				cache_C[pos] += cache_A[j] * cache_B[j];
			}


			mram_temp_addr_A += (BLOCK_SIZE - ((BLOCK_SIZE / sizeof(T)) - (n_size - n)) * sizeof(T));
			mram_temp_addr_B = mram_base_addr_B;

			if(mram_temp_addr_A % 8 != 0)
			{
				offset = 1;
			}
			else
			{
				offset = 0;
			}
		}
		// Write cache to current MRAM block
		mram_write(cache_C, (__mram_ptr void *) (mram_base_addr_C), 8);

		// Update memory address
		// mram_base_addr_C += 2 * sizeof(T);
		mram_base_addr_C += 8; 

	}

	return 0;
}
