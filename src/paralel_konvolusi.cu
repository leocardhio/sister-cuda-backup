%%file paralel.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NMAX 100
#define DATAMAX 1000
#define DATAMIN -1000

/* 
 * Struct Matrix
 *
 * Matrix representation consists of matrix data 
 * and effective dimensions 
 * */
typedef struct Matrix {
	int mat[NMAX][NMAX];	// Matrix cells
	int row_eff;			// Matrix effective row
	int col_eff;			// Matrix effective column
} Matrix;


/* 
 * Procedure init_matrix
 * 
 * Initializing newly allocated matrix
 * Setting all data to 0 and effective dimensions according
 * to nrow and ncol 
 * */
void init_matrix(Matrix *m, int nrow, int ncol) {
	m->row_eff = nrow;
	m->col_eff = ncol;

	for (int i = 0; i < m->row_eff; i++) {
		for (int j = 0; j < m->col_eff; j++) {
			m->mat[i][j] = 0;
		}
	}
}


/* 
 * Function input_matrix
 *
 * Returns a matrix with values from stdin input
 * */
Matrix input_matrix(int nrow, int ncol, FILE *fp) {
	Matrix input;
	init_matrix(&input, nrow, ncol);

	for (int i = 0; i < nrow; i++) {
		for (int j = 0; j < ncol; j++) {
			fscanf(fp, "%d", &input.mat[i][j]);
		}
	}

	return input;
}


/* 
 * Procedure print_matrix
 * 
 * Print matrix data
 * */
void print_matrix(Matrix *m) {
	for (int i = 0; i < m->row_eff; i++) {
		for (int j = 0; j < m->col_eff; j++) {
			printf("%d ", m->mat[i][j]);
		}
		printf("\n");
	}
}


/* 
 * Function get_matrix_datarange
 *
 * Returns the range between maximum and minimum
 * element of a matrix
 * */
int get_matrix_datarange(Matrix *m) {
	int max = DATAMIN;
	int min = DATAMAX;
	for (int i = 0; i < m->row_eff; i++) {
		for (int j = 0; j < m->col_eff; j++) {
			int el = m->mat[i][j];
			if (el > max) max = el;
			if (el < min) min = el;
		}
	}

	return max - min;
}

void matrixToArr (Matrix *in, int* out){
	
	for(int row = 0; row < in->row_eff; row++){
		for(int col = 0; col < in->col_eff; col++){
			out[row * in->col_eff + col] = in->mat[row][col];
		}
	}
}

void arrToMatrix (int* in, Matrix *out){
	for (int i = 0; i < out->row_eff; i++){
		for (int j = 0; j < out->col_eff; j++){
			out->mat[i][j] = in[i * out->col_eff + j];
		}
	}
}

__global__ void cudaSupression_op(int* kernel_arr, int* target_arr, int* out_arr, int target_col_eff, int kernel_row_eff, int kernel_col_eff, int res_row_eff, int res_col_eff){
	
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int intermediate_sum = 0;

	if ((row < res_row_eff) && (col < res_col_eff)){
		for (int kr = 0; kr < kernel_row_eff; kr++){
			for (int kc = 0; kc < kernel_col_eff; kc++){
				intermediate_sum += kernel_arr[kr * kernel_row_eff + kc] * target_arr[(row + kr) * target_col_eff + (col + kc)];
			}
		}
		out_arr[row * res_col_eff + col] = intermediate_sum;
	}
}

/*
 * Function supression_op
 *
 * Returns the sum of intermediate value of special multiplication
 * operation where kernel[0][0] corresponds to target[row][col]
 * */
int supression_op(Matrix *kernel, Matrix *target, int row, int col) {
	int intermediate_sum = 0;

	for (int i = 0; i < kernel->row_eff; i++) {
		for (int j = 0; j < kernel->col_eff; j++) {
			intermediate_sum += kernel->mat[i][j] * target->mat[row + i][col + j];
		}
	}

	return intermediate_sum;
}


/* 
 * Function convolution
 *
 * Return the output matrix of convolution operation
 * between kernel and target
 * */
Matrix convolution(Matrix *kernel, Matrix *target) {
	Matrix out;
	int *gpu_arr_kernel;
	int *gpu_arr_target;
	int *gpu_arr_out;
	int *gpu_target_col_eff;
	int *gpu_kernel_row_eff;
	int *gpu_kernel_col_eff;
	int *gpu_out_row_eff;
	int *gpu_out_col_eff;

	int out_row_eff = target->row_eff - kernel->row_eff + 1;
	int out_col_eff = target->col_eff - kernel->col_eff + 1;
	cudaError_t result_kernel_arr;
	cudaError_t result_target_arr;
	cudaError_t result_out_arr;
	cudaError_t target_col_eff;
	cudaError_t kernel_row_eff;
	cudaError_t kernel_col_eff;
	cudaError_t cuda_out_row_eff;
	cudaError_t cuda_out_col_eff;
	cudaError_t get_result;
	
	//ini buat naruh hasilnya
	init_matrix(&out, out_row_eff, out_col_eff);

	int arr_kernel[kernel->col_eff * kernel->row_eff];
	int arr_target[target->col_eff * target->row_eff];
	int arr_out[out_col_eff * out_row_eff];

	matrixToArr(kernel, arr_kernel);
	matrixToArr(target, arr_target);
	matrixToArr(&out, arr_out);

	result_kernel_arr = cudaMalloc((void **)&gpu_arr_kernel, kernel->col_eff * kernel->row_eff * sizeof(int));
	result_target_arr = cudaMalloc((void **)&gpu_arr_target, target->col_eff * target->row_eff * sizeof(int));
	result_out_arr = cudaMalloc((void **)&gpu_arr_out, out_col_eff * out_row_eff * sizeof(int));
	target_col_eff = cudaMalloc((void **)&gpu_target_col_eff, sizeof(int));
	kernel_col_eff = cudaMalloc((void **)&gpu_kernel_col_eff, sizeof(int));
	kernel_row_eff = cudaMalloc((void **)&gpu_kernel_row_eff, sizeof(int));
	cuda_out_row_eff = cudaMalloc((void **)&gpu_out_row_eff, sizeof(int));
	cuda_out_col_eff = cudaMalloc((void **)&gpu_out_col_eff, sizeof(int));

	printf("malloc arr_kernel status: %s\n", (
		result_kernel_arr == cudaSuccess && 
		kernel_col_eff == cudaSuccess && 
		kernel_row_eff == cudaSuccess) 
		? "success" : "fail");
	printf("malloc arr_target status: %s\n", (
		result_target_arr == cudaSuccess &&
		target_col_eff == cudaSuccess
		) ? "success" : "fail");
	printf("malloc gpu_arr_out status: %s\n", (
		result_out_arr == cudaSuccess &&
		cuda_out_col_eff == cudaSuccess &&
		cuda_out_row_eff == cudaSuccess
		) ? "success" : "fail");

	

	result_kernel_arr = cudaMemcpy(gpu_arr_kernel, arr_kernel, kernel->col_eff * kernel->row_eff * sizeof(int), cudaMemcpyHostToDevice);
	result_target_arr = cudaMemcpy(gpu_arr_target, arr_target, target->col_eff * target->row_eff * sizeof(int), cudaMemcpyHostToDevice);
	result_out_arr = cudaMemcpy(gpu_arr_out, arr_out, out_col_eff * out_row_eff * sizeof(int), cudaMemcpyHostToDevice);
	target_col_eff = cudaMemcpy(gpu_target_col_eff, &target->col_eff, sizeof(int), cudaMemcpyHostToDevice);
	kernel_col_eff = cudaMemcpy(gpu_kernel_col_eff, &kernel->col_eff, sizeof(int), cudaMemcpyHostToDevice);
	kernel_row_eff = cudaMemcpy(gpu_kernel_row_eff, &kernel->row_eff, sizeof(int), cudaMemcpyHostToDevice);
	cuda_out_row_eff = cudaMemcpy(gpu_out_row_eff, &out_row_eff, sizeof(int), cudaMemcpyHostToDevice);
	cuda_out_col_eff = cudaMemcpy(gpu_out_col_eff, &out_col_eff, sizeof(int), cudaMemcpyHostToDevice);
	printf("send arr_kernel status: %s\n", (
		result_kernel_arr == cudaSuccess && 
		kernel_col_eff == cudaSuccess && 
		kernel_row_eff == cudaSuccess) 
		? "success" : "fail");
	printf("send arr_target status: %s\n", (
		result_target_arr == cudaSuccess &&
		target_col_eff == cudaSuccess
		) ? "success" : "fail");
	printf("send gpu_arr_out status: %s\n", (
		result_out_arr == cudaSuccess &&
		cuda_out_col_eff == cudaSuccess &&
		cuda_out_row_eff == cudaSuccess
		) ? "success" : "fail");
	
	double thread_per_block = 32;
	//pembagian kerja rata dikerjakan 32*32 thread
	int grid_col = ceil(double(out_col_eff) / thread_per_block);
	int grid_row = ceil(double(out_row_eff) / thread_per_block);
	dim3 DimGrid(grid_col, grid_row, 1);
	dim3 DimBlock(thread_per_block, thread_per_block, 1);

	printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
		DimGrid.x, DimGrid.y, DimGrid.z, DimBlock.x, DimBlock.y, DimBlock.z);


	for (int i = 0; i < out.row_eff; i++) {
		for (int j = 0; j < out.col_eff; j++) {
			// out.mat[i][j] = supression_op(kernel, target, i, j);
			cudaSupression_op <<<DimGrid,DimBlock>>>(
				(int*) gpu_arr_kernel, (int*) gpu_arr_target, (int*) gpu_arr_out, target->col_eff, 
				kernel->row_eff, kernel->col_eff, 
				out_row_eff, out_col_eff);
			// testPrint<<<DimGrid, DimBlock>>>();
		}
	}

	get_result = cudaMemcpy(arr_out, gpu_arr_out, out_col_eff * out_row_eff * sizeof(int), cudaMemcpyDeviceToHost);
	printf("get result: %s\n", (get_result == cudaSuccess) ? "success" : "failed");
	
	arrToMatrix(arr_out, &out);

	return out;
}


/*
 * Procedure merge_array
 *
 * Merges two subarrays of n with n[left..mid] and n[mid+1..right]
 * to n itself, with n now ordered ascendingly
 * */
void merge_array(int *n, int left, int mid, int right) {
	int n_left = mid - left + 1;
	int n_right = right - mid;
	int iter_left = 0, iter_right = 0, iter_merged = left;
	int arr_left[n_left], arr_right[n_right];

	for (int i = 0; i < n_left; i++) {
		arr_left[i] = n[i + left];
	}

	for (int i = 0; i < n_right; i++) {
		arr_right[i] = n[i + mid + 1];
	}

	while (iter_left < n_left && iter_right < n_right) {
		if (arr_left[iter_left] <= arr_right[iter_right]) {
			n[iter_merged] = arr_left[iter_left++];
		} else {
			n[iter_merged] = arr_right[iter_right++];
		}
		iter_merged++;
	}

	while (iter_left < n_left)  {
		n[iter_merged++] = arr_left[iter_left++];
	}
	while (iter_right < n_right) {
		n[iter_merged++] = arr_right[iter_right++];
	} 
}


/* 
 * Procedure merge_sort
 *
 * Sorts array n with merge sort algorithm
 * */
void merge_sort(int *n, int left, int right) {
	if (left < right) {
		int mid = left + (right - left) / 2;

		merge_sort(n, left, mid);
		merge_sort(n, mid + 1, right);

		merge_array(n, left, mid, right);
	}	
}
 

/* 
 * Procedure print_array
 *
 * Prints all elements of array n of size to stdout
 * */
void print_array(int *n, int size) {
	for (int i = 0; i < size; i++ ) printf("%d ", n[i]);
	printf("\n");
}


/* 
 * Function get_median
 *
 * Returns median of array n of length
 * */
int get_median(int *n, int length) {
	int mid = length / 2;
	if (length & 1) return n[mid];

	return (n[mid - 1] + n[mid]) / 2;
}


/* 
 * Function get_floored_mean
 *
 * Returns floored mean from an array of integers
 * */
long get_floored_mean(int *n, int length) {
	long sum = 0;
	for (int i = 0; i < length; i++) {
		sum += n[i];
	}

	return sum / length;
}



// main() driver
int main() {
	int kernel_row, kernel_col, target_row, target_col, num_targets;
	clock_t start, end;
	FILE *out;
	out = fopen(argv[2],"w");
	
	// reads kernel's row and column and initalize kernel matrix from input
	char *filename = argv[1];
    FILE *fp = fopen(filename, "r");
	fscanf(fp, "%d %d", &kernel_row, &kernel_col);
	Matrix kernel = input_matrix(kernel_row, kernel_col, fp);
	
	// reads number of target matrices and their dimensions.
	// initialize array of matrices and array of data ranges (int)
	fscanf(fp, "%d %d %d", &num_targets, &target_row, &target_col);
	Matrix* arr_mat = (Matrix*)malloc(num_targets * sizeof(Matrix));
	int arr_range[num_targets];

	// read each target matrix, compute their convolution matrices, and compute their data ranges
	for (int i = 0; i < num_targets; i++) {
		arr_mat[i] = input_matrix(target_row, target_col, fp);
	}

	start = clock();
	for (int i = 0; i < num_targets; i++) {
		arr_mat[i] = convolution(&kernel, &arr_mat[i]);
		arr_range[i] = get_matrix_datarange(&arr_mat[i]); 
	}

	// sort the data range array
	merge_sort(arr_range, 0, num_targets - 1);
	
	int median = get_median(arr_range, num_targets);	
	int floored_mean = get_floored_mean(arr_range, num_targets); 

	end = clock();
	printf("Execution time (cuda): %f\n", (double) (end-start)/CLOCKS_PER_SEC);

	// print the min, max, median, and floored mean of data range array
	printf("%d\n%d\n%d\n%d\n", 
			arr_range[0], 
			arr_range[num_targets - 1], 
			median, 
			floored_mean);

	fprintf(out,"%d\n%d\n%d\n%d\n", 
			arr_range[0], 
			arr_range[num_targets - 1], 
			median, 
			floored_mean);
	

	fclose(out);

	return 0;
}
