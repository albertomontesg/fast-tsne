
bool load_data(double* data, int n, int* d, char* data_file);
void save_data(double* data, int n, int d, char* data_file);
void csr_to_dense(unsigned int* data_row, unsigned int* data_col, double* data_value, double* data);
void save_csr_data(unsigned int* data_row, unsigned int* data_col, double* data_value, int n, int d, char* data_file);