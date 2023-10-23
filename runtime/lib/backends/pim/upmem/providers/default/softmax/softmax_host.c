// #include <assert.h>
// #include <dpu.h>
// #include <dpu_log.h>
// #include <stdio.h>
// #include <math.h>
// #include <sys/time.h>

// #ifdef CORDIC_F2F
//     #include "../../host/cordic_f2f_host.c"
//     char method[]="cordic_f2f";
// #elif defined CORDIC_LUT
//     #include "../../host/cordic_lut_host.c"
//     char method[]="cordic_lut";
// #elif defined LUT_LDEXPF
//     #include "../../host/lut_ldexpf_host.c"
//     char method[]="lut_ldexpf_nointerpolate";
// #elif defined LUT_LDEXPF_INTERPOLATE
//     #include "../../host/lut_ldexpf_host.c"
//     char method[]="lut_ldexpf_interpolate";
// #elif defined LUT_MULTI
//     #include "../../host/lut_multi_host.c"
//     char method[]="lut_multi_nointerpolate";
// #elif defined LUT_MULTI_INTERPOLATE
//     #include "../../host/lut_multi_host.c"
//     char method[]="lut_multi_interpolate";
// #elif defined LUT_DIRECT
//     #include "../../host/lut_direct_host.c"
//     char method[]="lut_direct_interpolate";
// #elif defined POLYNOMIAL
//     char method[]="polynomial";
// #endif


// #ifndef DPU_BINARY
// #define DPU_BINARY "bin/softmax_float"
// #endif

// #define PAD 2560*24

// int numError = 0;
// int nThreads;
// float *input;
// float *output;


// int softmax (int argc, char **argv)
// {
//     int i;
//     int loopnum;
//     int numOptions;
//     int nr_dpus;
//     int rv;

//     if (argc != 3) {
//         printf("Usage:\n\t%s <inputFile> <outputFile>\n", argv[0]);
//         exit(1);
//     }
//     char *inputFile = argv[1];
//     char *outputFile = argv[2];

//     //Read input data from file
//     file = fopen(inputFile, "r");

//     if(file == NULL) {
//         printf("ERROR: Unable to open file `%s'.\n", inputFile);
//         exit(1);
//     }
//     rv = fscanf(file, "%i", &numOptions);
//     if(rv != 1) {
//         printf("ERROR: Unable to read from file `%s'.\n", inputFile);
//         fclose(file);
//         exit(1);
//     }

//     // alloc spaces for the option data
//     input = (float*)malloc((numOptions+PAD)*sizeof(float));
//     output = (float*)malloc((numOptions+PAD)*sizeof(float));
//     for ( loopnum = 0; loopnum < numOptions; ++ loopnum )
//     {
//         rv = fscanf(file, "%f", &input[loopnum]);
//         if(rv != 1) {
//             printf("ERROR: Unable to read from file `%s'.\n", inputFile);
//             fclose(file);
//             exit(1);
//         }
//     }
//     rv = fclose(file);
//     if(rv != 0) {
//         printf("ERROR: Unable to close file `%s'.\n", inputFile);
//         exit(1);
//     }

//     // Start measuring time
//     struct timeval begin, end, begin_inner, end_inner;
  

//     // Allocate DPUs
//     struct dpu_set_t set, dpu;
//     DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, NULL, &set));
//     DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

//     // Distribute Workload
//     unsigned int dpu_amount;
//     DPU_ASSERT(dpu_get_nr_dpus(set, &dpu_amount));

//     int used_rows = (numOptions - 1) / (dpu_amount) + 1;
//     used_rows += used_rows % 2; // Round up so transfers are aligned to 8 bytes
//     DPU_ASSERT(dpu_broadcast_to(set, "used_rows", 0, &used_rows, sizeof(int), DPU_XFER_DEFAULT));

//     DPU_FOREACH(set, dpu, i) {
//         DPU_ASSERT(dpu_prepare_xfer(dpu, &input[i * used_rows]));
//     }
//     DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "data_array", 0, sizeof(float) * used_rows, DPU_XFER_DEFAULT));



//     // Setup the tables we need for cordic / lut
// #ifndef POLYNOMIAL
//     broadcast_tables(set);
// #endif

//     // Launch first Task
//     int step = 0;
//     DPU_ASSERT(dpu_broadcast_to(set, "step", 0, &step, sizeof(int), DPU_XFER_DEFAULT));
//     DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

//     //
//     float total_sum = 0.0;
//     float next_part = 0.0;
//     DPU_FOREACH(set, dpu, i) {
//         // DPU_ASSERT(dpu_log_read(dpu, stdout));
//         DPU_ASSERT(dpu_copy_from(dpu, "shared_sum", 0, &next_part, sizeof(float)));
//         total_sum += next_part;
//     }

//     float inverted_sum = 1.0 / total_sum;

//     DPU_ASSERT(dpu_broadcast_to(set, "inverted_sum", 0, &inverted_sum, sizeof(float), DPU_XFER_DEFAULT));

//     step = 1;
//     DPU_ASSERT(dpu_broadcast_to(set, "step", 0, &step, sizeof(int), DPU_XFER_DEFAULT));

//     DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

//     // Collect output
//     DPU_FOREACH(set, dpu, i) {
//         // DPU_ASSERT(dpu_log_read(dpu, stdout));
//         DPU_ASSERT(dpu_copy_from(dpu, "data_array", 0, &output[i * used_rows], sizeof(float) * used_rows));
//     }
//     DPU_ASSERT(dpu_free(set));


//     double inner_time = (end_inner.tv_sec - begin_inner.tv_sec) + (end_inner.tv_usec - begin_inner.tv_usec)*1e-6;
//     double total_time = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec)*1e-6;



//     return 0;
// }