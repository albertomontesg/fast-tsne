
#ifdef SINGLE_PRECISION
#define save_data(data, n, d, data_file) save_dataf(data, n, d, data_file);
#else
#define save_data(data, n, d, data_file) save_datad(data, n, d, data_file);
#endif

bool load_data(double* data, int n, int* d, char* data_file);
void save_datad(double* data, int n, int d, char* data_file);
void save_dataf(float* data, int n, int d, char* data_file);
