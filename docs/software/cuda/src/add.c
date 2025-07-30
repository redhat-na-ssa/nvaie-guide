#include <iostream>
#include <math.h>
#include <ctime>

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
 for (int i = 0; i < n; i++)
     y[i] = x[i] + y[i];
}
 
int main(void)
{
 int N = 1<<20; // 1M elements
 
 float *x = new float[N];
 float *y = new float[N];
 
 // initialize x and y arrays on the host
 for (int i = 0; i < N; i++) {
   x[i] = 1.0f;
   y[i] = 2.0f;
 }

 clock_t start_time = clock();

 // Run kernel on 1M elements on the CPU
 add(N, x, y);

 clock_t end_time = clock(); // Get the ending clock ticks

 // Calculate elapsed time in seconds
 double time_taken = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

 std::cout << "Time taken: " << time_taken << " seconds" << std::endl;

 
 // Check for errors (all values should be 3.0f)
 float maxError = 0.0f;
 for (int i = 0; i < N; i++)
   maxError = fmax(maxError, fabs(y[i]-3.0f));
 std::cout << "Max error: " << maxError << std::endl;
 
 // Free memory
 delete [] x;
 delete [] y;
 
 return 0;
}

