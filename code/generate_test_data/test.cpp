#include <iostream>
#include "../implementations/utils/io.h"
 
int main(int argc, char **argv)
{
	int N = 10000;
	int D;
	char* fname = argv[1];
	double *data = (double*) malloc(N * 2000 * sizeof(double));	
	std::cout << fname << std::endl;
	bool success = load_data(data, N, &D, fname);
    if (!success) 
    {
    	std::cout << "fail" << std::endl;
    	exit(1);
    }
	std::cout << "yay " << D << std::endl;
	return 0;
}