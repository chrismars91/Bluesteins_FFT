#include <iostream>
#include <fstream>
#include <cmath>
#include <complex> 
#include <vector>
using namespace std;
const float pi = M_PI; 
const float tau = 2*M_PI; 
complex<float> j(0, 1);

vector<float> smplspace(float start, float end, int n)
{
    vector<float> vec;
    float step = (end-start)/n;
    for (int i = 0; i < n; i++) {vec.push_back(start); start+=step;}
    return vec;
}

void FFT_Bluestein_Vec( complex<float> *x, complex<float> *y, int N, complex<float> *rslt )
{
    complex<float> M[2][2] = {{1.0,1.0},{1.0,-1.0}};
    int ROWS = 2;
    int COLS = N/2;
    int COLS_O2 = COLS/2;
    vector<vector<complex<float>>> Ax( ROWS , vector<complex<float>> (COLS)); 
    vector<vector<complex<float>>> Ay( ROWS , vector<complex<float>> (COLS));
    for (int i = 0; i < ROWS; i++)
    {
        for (int j = 0; j < COLS; j++)
        {
            Ax[i][j] =  0;  
            Ay[i][j] =  0;  
            for (int k = 0; k < ROWS; k++)
            {
                Ax[i][j] +=  M[i][k] * x[k*COLS+j];    
                Ay[i][j] +=  M[i][k] * y[k*COLS+j];              
            }
        }
    }
    complex<float> t;
    complex<float> t_real;
    complex<float> e;
    complex<float> o;  
    for (int ii = 0; ii < log2(N)-1; ii++)
    {
        complex<float> Jx[2*ROWS][COLS_O2];
        complex<float> Jy[2*ROWS][COLS_O2];        
        for (int i = 0; i < ROWS; i++)
        {
            t_real = pi*i/ROWS;
            t = exp(j*t_real);
            for (int k = 0; k < COLS_O2; k++)
            {
                e = Ax[i][k];
                o = Ax[i][k+COLS_O2];
                Jx[i][k] = e+t*o;
                Jx[i+ROWS][k] = e-t*o;
                e = Ay[i][k];
                o = Ay[i][k+COLS_O2];
                Jy[i][k] = e+t*o;
                Jy[i+ROWS][k] = e-t*o;     
            }
        }
        Ax = vector<vector<complex<float>>>(ROWS*2, vector<complex<float>>(COLS_O2));
        Ay = vector<vector<complex<float>>>(ROWS*2, vector<complex<float>>(COLS_O2));
        for (int i = 0; i < ROWS; i++)
        {
            for (int k = 0; k < COLS_O2; k++)
            {
                Ax[i][k]=Jx[i][k];
                Ax[i+ROWS][k]=Jx[i+ROWS][k];  
                Ay[i][k]=Jy[i][k];
                Ay[i+ROWS][k]=Jy[i+ROWS][k]; 
            }
        }
        ROWS = ROWS*2;
        COLS = COLS/2; 
        COLS_O2 = COLS/2;

    }
    for (int n = 0; n < N; n++){rslt[n] = Ax[n][0] * Ay[n][0];}
}


void IFFT_Bluestein_Vec( complex<float> *x, int N, complex<float> *ift_rslt )
{
    complex<float> M[2][2] = {{1.0,1.0},{1.0,-1.0}};
    float iLen = 1.0/N;
    int ROWS = 2;
    int COLS = N/2;
    int COLS_O2 = COLS/2;
    vector<vector<complex<float>>> Ax( ROWS , vector<complex<float>> (COLS)); 
    for (int i = 0; i < ROWS; i++)
    {
        for (int j = 0; j < COLS; j++)
        {
            for (int k = 0; k < ROWS; k++)
            {
                Ax[i][j] +=  M[i][k] * x[k*COLS+j];                 
            }
        }
    }
    complex<float> t;
    complex<float> t_real;
    complex<float> e;
    complex<float> o;  
    for (int ii = 0; ii < log2(N)-1; ii++)   
    {
        complex<float> Jx[2*ROWS][COLS_O2];    
        for (int i = 0; i < ROWS; i++)
        {
            t_real = pi*i/ROWS;
            t = exp(-j*t_real);
            for (int k = 0; k < COLS_O2; k++)
            {
                e = Ax[i][k];
                o = Ax[i][k+COLS_O2];
                Jx[i][k] = e+t*o;
                Jx[i+ROWS][k] = e-t*o;  
            }
        }
        Ax = vector<vector<complex<float>>>(ROWS*2, vector<complex<float>>(COLS_O2));
        for (int i = 0; i < ROWS; i++)
        {
            for (int k = 0; k < COLS_O2; k++)
            {
                Ax[i][k]=Jx[i][k];
                Ax[i+ROWS][k]=Jx[i+ROWS][k];  
            }
        }        
        ROWS = ROWS*2;
        COLS = COLS/2; 
        COLS_O2 = COLS/2;
    }
    for (int n = 0; n < N; n++){ift_rslt[n] = Ax[n][0]*iLen;}
}


class Bluestein
{
    public:
    int n;
    vector<complex<float>> dfft;
    Bluestein( vector<float>& signal )
    {
        n = signal.size();
        int l = pow(2,ceil(log2(2 * n + 1)));
        complex<float> comp_real;
        complex<float> comp;
        float idx = 0.0;
        float one = 1.0;
        complex<float> U_l[l];
        complex<float> V_l[l+1];
        complex<float> V_star[n];
        for(int i = 0; i < n; i ++)
        {
            comp_real = pi*(idx*idx)/n; 
            comp = exp(j*comp_real);
            V_star[i] = one/comp;
            U_l[i] = signal[i]/comp;
            V_l[i] = comp;
            V_l[l-i] = comp;
            idx+=1.0;
        }  
        complex<float> rslt[l];
        complex<float> ift_rslt[l];
        FFT_Bluestein_Vec(U_l, V_l,l,rslt);
        IFFT_Bluestein_Vec(rslt,l,ift_rslt);
        for(int i = 0; i < n; i ++){dfft.push_back(ift_rslt[i]*V_star[i]);}
    }
    vector<complex<float>> getFourCoeff(){ return dfft; } 
};

int main()
{
    int sr = 10;
    float t0 = 0;
    float tn = 1;
    vector<float> time = smplspace(t0,tn,sr);
    vector<float> y;
    for (int i = 0; i < time.size(); i++)
    {y.push_back(2*sin(tau*3*time[i]+pi/4));}    
    Bluestein b1(y);
    vector<complex<float>> rst = b1.getFourCoeff();
    for (auto i: rst)
    cout << i << ' ';     
}