
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <math.h>
#include <complex>
#include <cufft.h>
#include <string>
//#pragma comment ( lib, "cufft.lib" )
#define BSZ 16 //block size
#define M_PI		3.14159265358979323846
using namespace std;

__global__ void cal_divergence(cufftComplex *fxt, cufftComplex *fyt, cufftComplex *dft, 
								float *k, int N)
{
	int i = threadIdx.x + blockIdx.x*BSZ;
    int j = threadIdx.y + blockIdx.y*BSZ; 
	int index = j*N+i; 

	if(i<N && j<N)
	{ 
	  //float k2 = k[i]*k[i]+k[j]*k[j];
	  dft[index].x = -(k[i]*fxt[index].y + k[j]*fyt[index].y);
	  dft[index].y = (k[i]*fxt[index].x + k[j]*fyt[index].x);
	}
}

__global__ void cal_grad(cufftComplex *pt, cufftComplex *pt_gradx, cufftComplex *pt_grady, float *k, int N)
{
	int i = threadIdx.x + blockIdx.x*BSZ;
    int j = threadIdx.y + blockIdx.y*BSZ; 
	int index = j*N+i; 
	if (i<N && j<N)
	{               
	  pt_gradx[index].x = -1*pt[index].y*k[i];
	  pt_gradx[index].y = pt[index].x*k[i];

	  pt_grady[index].x = -1*pt[index].y*k[j];
	  pt_grady[index].y = pt[index].x*k[j];
	}
}

__global__ void solve_poisson(cufftComplex *ft, cufftComplex *ft_k, float *k, int N) 
{  
	int i = threadIdx.x + blockIdx.x*BSZ;
    int j = threadIdx.y + blockIdx.y*BSZ; 
	int index = j*N+i; 
	if (i<N && j<N)
	{               
	  float k2 = k[i]*k[i]+k[j]*k[j];
	  if (i==0 && j==0) {k2 = 1.0f;} 
	  ft_k[index].x = -ft[index].x/k2;
	  ft_k[index].y = -ft[index].y/k2;
	}
}

__global__ void solve_velocity(cufftComplex *f, cufftComplex *pt_grad, cufftComplex *u, float *k, int N) 
{  
	int i = threadIdx.x + blockIdx.x*BSZ;
    int j = threadIdx.y + blockIdx.y*BSZ; 
	int index = j*N+i; 
	if (i<N && j<N)
	{               
	  float k2 = k[i]*k[i]+k[j]*k[j];
	  if (i==0 && j==0) {k2 = 1.0f;} 
	  u[index].x = -(pt_grad[index].x - f[index].x)/k2;
	  u[index].y = -(pt_grad[index].y - f[index].y)/k2;
	}
}

__global__ void real2complex(float *f, cufftComplex *fc, int N) 
  {          
  int i = threadIdx.x + blockIdx.x*BSZ; 
  int j = threadIdx.y + blockIdx.y*BSZ;  
  int index = j*N+i;  
  if (i<N && j<N)   
  {      fc[index].x = f[index];     
         fc[index].y = 0.0f; 
   }  
  }  

__global__ void complex2real(cufftComplex *fc, float *f, int N)
  {          
  int i = threadIdx.x + blockIdx.x*BSZ;
  int j = threadIdx.y + blockIdx.y*BSZ; 
  int index = j*N+i;         
  if (i<N && j<N)  
  {   f[index] = fc[index].x/((float)N*(float)N); 
      //divide by number of elements to recover value    
  }
  }


/******************************* MAIN ***************************************/
int main(int argc, char** argv) 
{
	//int N = 512;  //block number
	int N = 8;
	if (argc < 2)
	{
		cout << "please specify the half wave number N !" << endl;
		exit(0);
	}
	else if(argc > 2)
	{
		cout << "too many arguments!" << endl;
		exit(0);
	}
	else
	{
		N = 2*atoi(argv[1]);
	}
	float xmax=2*M_PI, xmin=0.0f,ymin=0.0f,h=(xmax-xmin)/((float)N),s=0.1,s2=s*s;   //define interval, sigma
	
	float *x=new float[N*N],*y=new float[N*N],
         *u=new float[N*N],*v = new float[N*N], *p = new float[N*N],
		 *fx = new float[N*N], *fy = new float[N*N]; //define x,y,u,v,p,fx,fy

	float *u_acc = new float[N*N];
	float r2;    
	clock_t start, stop;
	double duration;
	start = clock();
	for (int j=0; j<N; j++)
	{                 
      for (int i=0; i<N; i++)                 
	 { 
	   x[N*j+i] = xmin + i*h;  
	   y[N*j+i] = ymin + j*h;                        
       r2 = (x[N*j+i]-0.5*(xmax-xmin))*(x[N*j+i]-0.5*(xmax-xmin)) + (y[N*j+i]-0.5*(xmax-xmin))*(y[N*j+i]-0.5*(xmax-xmin));  //define r^2
	   fx[N*j+i] = exp(-r2/(0.01)); //define f at right hand side 
	   
	   //fx[N*j+i] = sin(4*(ymin + j*h));
	   //u_acc[N*j+i] = 1/(4*4)*sin(4*(ymin + j*h));
	   fy[N*j+i] = 0.0;   
	 }        
	}

	ofstream xfile("x.csv");
	ofstream yfile("y.csv");
	ofstream u_accfile("u_acc.csv");
	for (int j=0; j<N; j++)
	{
		for (int i=0; i<N; i++)
		{
			xfile << x[j*N+i];
			yfile << y[j*N+i];
			u_accfile << u_acc[j*N+i];
			if ( i<(N-1) ) {xfile << ','; yfile<<','; u_accfile<<',';}
		}
		xfile << endl;
		yfile << endl;
		u_accfile << endl;
	}  

	float alpha = xmax - xmin;
	float   *k = new float[N];       
	for (int i=0; i<=N/2; i++)          
	{
	   k[i] = i*2*M_PI/alpha;
    } 
	for (int i=N/2+1; i<N; i++)          
	{
        k[i] = (i - N) * 2*M_PI/alpha;
	}
		 
    // Allocate arrays on the device   
	float *k_d, *fx_d, *fy_d, *u_d, *v_d, *p_d; 
    cudaMalloc ((void**)&k_d, sizeof(float)*N); 
	cudaMalloc ((void**)&fx_d, sizeof(float)*N*N); 
	cudaMalloc ((void**)&fy_d, sizeof(float)*N*N);
	cudaMalloc ((void**)&u_d, sizeof(float)*N*N);
	cudaMalloc ((void**)&v_d, sizeof(float)*N*N);
	cudaMalloc ((void**)&p_d, sizeof(float)*N*N);
	cudaMemcpy(k_d, k, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(fx_d, fx, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(fy_d, fy, sizeof(float)*N*N, cudaMemcpyHostToDevice);

	cufftComplex *u_dc, *v_dc, *p_dc, *fx_dc, *fy_dc;
	cudaMalloc ((void**)&u_dc, sizeof(cufftComplex)*N*N); 
	cudaMalloc ((void**)&v_dc, sizeof(cufftComplex)*N*N); 
	cudaMalloc ((void**)&p_dc, sizeof(cufftComplex)*N*N); 
	cudaMalloc ((void**)&fx_dc, sizeof(cufftComplex)*N*N); 
	cudaMalloc ((void**)&fy_dc, sizeof(cufftComplex)*N*N); 

	cufftComplex *ut_d, *vt_d, *pt_d, *pt_gradx_d, *pt_grady_d, *fxt_d, *fyt_d, *dft_d;
	cudaMalloc ((void**)&ut_d, sizeof(cufftComplex)*N*N);
	cudaMalloc ((void**)&vt_d, sizeof(cufftComplex)*N*N);
	cudaMalloc ((void**)&pt_d, sizeof(cufftComplex)*N*N);
	cudaMalloc ((void**)&pt_gradx_d, sizeof(cufftComplex)*N*N);
	cudaMalloc ((void**)&pt_grady_d, sizeof(cufftComplex)*N*N);
	cudaMalloc ((void**)&fxt_d, sizeof(cufftComplex)*N*N);
	cudaMalloc ((void**)&fyt_d, sizeof(cufftComplex)*N*N);
	cudaMalloc ((void**)&dft_d, sizeof(cufftComplex)*N*N);

	dim3 dimGrid  (int((N-0.5)/BSZ) + 1, int((N-0.5)/BSZ) + 1);
	//Built-in variable gridDim specifies the size (or dimension) of the grid
	dim3 dimBlock (BSZ, BSZ); 
	real2complex<<<dimGrid, dimBlock>>>(fx_d, fx_dc, N);//real2complex(float *f, cufftComplex *fc, int N) 
	real2complex<<<dimGrid, dimBlock>>>(fy_d, fy_dc, N);
	cufftHandle plan;//Handle type to store CUFFT plans
	cufftPlan2d(&plan, N, N, CUFFT_C2C);//complex to complex

	//#1. solve the pressure field, then obtain the gradient of pressure for the next step:
	cufftExecC2C(plan, fx_dc, fxt_d, CUFFT_FORWARD);
	cufftExecC2C(plan, fy_dc, fyt_d, CUFFT_FORWARD);
	cal_divergence<<<dimGrid, dimBlock>>>(fxt_d, fyt_d, dft_d, k_d, N);
	solve_poisson<<<dimGrid, dimBlock>>> (dft_d, pt_d, k_d, N);
	cal_grad<<<dimGrid, dimBlock>>> (pt_d, pt_gradx_d, pt_grady_d, k_d, N);
	
	//#2. solve velocity field u, v;
	solve_velocity<<<dimGrid, dimBlock>>> (fxt_d, pt_gradx_d, ut_d, k_d, N);
	solve_velocity<<<dimGrid, dimBlock>>> (fyt_d, pt_grady_d, vt_d, k_d, N);

	//#3. reverse to real space from spectral space
	cufftExecC2C(plan, ut_d, u_dc, CUFFT_INVERSE);
	complex2real<<<dimGrid, dimBlock>>>(u_dc, u_d, N);
	cudaMemcpy(u, u_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost); 
	float constant = u[0]; 
	for (int i=0; i<N*N; i++)
	{       
	   u[i] -= constant; //substract u[0] to force the arbitrary constant to be 0
	}

	cufftExecC2C(plan, vt_d, v_dc, CUFFT_INVERSE);
	complex2real<<<dimGrid, dimBlock>>>(v_dc, v_d, N);
	cudaMemcpy(v, v_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost); 
	constant = v[0]; 
	for (int i=0; i<N*N; i++)
	{       
	   v[i] -= constant; //substract u[0] to force the arbitrary constant to be 0
	}

	cufftExecC2C(plan, pt_d, p_dc, CUFFT_INVERSE);
	complex2real<<<dimGrid, dimBlock>>>(p_dc, p_d, N);
	cudaMemcpy(p, p_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost); 
	constant = p[0]; 
	for (int i=0; i<N*N; i++)
	{       
	   p[i] -= constant; //substract u[0] to force the arbitrary constant to be 0
	}

	
	// cufftExecC2C(plan, f_dc, ft_d, CUFFT_FORWARD);
	// solve_poisson<<<dimGrid, dimBlock>>>(ft_d, ft_d_k, k_d, N);// solve_poisson(cufftComplex *ft, cufftComplex *ft_k, float *k, int N)
	// cufftExecC2C(plan, ft_d_k, u_dc, CUFFT_INVERSE);
	// complex2real<<<dimGrid, dimBlock>>>(u_dc, u_d, N);//complex2real(cufftComplex *fc, float *f, int N)
	// cudaMemcpy(u, u_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost); 
	// float constant = u[0]; 
	// for (int i=0; i<N*N; i++)
	// {       
	//    u[i] -= constant; //substract u[0] to force the arbitrary constant to be 0
	// }
	
	// cudaFree(k_d);
	// cudaFree(f_d);
	// cudaFree(u_d);
	// cudaFree(ft_d);
	// cudaFree(f_dc);
	// cudaFree(ft_d_k);
	// cudaFree(u_dc);
	stop = clock();
	duration = (double)(stop-start)/CLOCKS_PER_SEC;
	cout << duration << endl;

	std::ofstream ufile("u.csv");
	std::ofstream vfile("v.csv");
	std::ofstream pfile("p.csv");

	for (int j=0; j<N; j++)
	{
		for (int i=0; i<N; i++)
		{
			ufile << u[j*N+i];
			vfile << v[j*N+i];
			pfile << p[j*N+i];
			if ( i<(N-1) ) {ufile << ',';vfile << ',';pfile << ',';}
		}
		ufile << endl;
		vfile << endl;
		pfile << endl;
	}
}
