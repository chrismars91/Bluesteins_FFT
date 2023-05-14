#include <iostream>
#include <fstream>
#include <cmath>
#include <complex> 
#include <vector>
using namespace std;
const float pi = M_PI; 
const float tau = 2*M_PI; 
complex<float> j(0, 1);

void fft(vector<complex<float>>& a, bool invert, int limit = 0) {
    int n;
    if (limit == 0){n = a.size();}
    else{n = limit;}

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            swap(a[i], a[j]);
        }
    }
    complex<float> u;
    complex<float> v;
    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2 * pi / len * (invert ? -1 : 1);
        complex<float> wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            int len_o2 = len/2;
            complex<float> w(1);
            for (int j = 0; j < len_o2; j++) {
                u = a[i + j];
                v = a[i + j + len_o2] * w;
                a[i + j] = u + v;
                a[i + j + len_o2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (int i = 0; i < n; i++) {
            a[i] /= n;
        }
    }
}


void fftblue(vector<complex<float>>& a,vector<complex<float>>& b, bool invert) {

    int n = a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            swap(a[i], a[j]);
            swap(b[i], b[j]);
        }
    }
    complex<float> u;
    complex<float> v;
    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2 * pi / len * (invert ? -1 : 1);
        complex<float> wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            int len_o2 = len/2;
            complex<float> w(1);
            for (int j = 0; j < len_o2; j++) {
                u = a[i + j];
                v = a[i + j + len_o2] * w;
                a[i + j] = u + v;
                a[i + j + len_o2] = u - v;
                u = b[i + j];
                v = b[i + j + len_o2] * w;
                b[i + j] = u + v;
                b[i + j + len_o2] = u - v;                        
                w *= wlen;
            }
        }
    }
    for (int i = 0; i < n; ++i){a[i]*=b[i];}
    if (invert) {
        for (int i = 0; i < n; i++) {
            a[i] /= n;
        }
    }
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
        float nInv = 1.0/n;
        complex<float> comp;
        float idx = 0.0;
        float onef = 1.0;
        vector<complex<float>> U_l(l);
        vector<complex<float>> V_l(l+1);
        vector<complex<float>> V_star(n);
        for(int i = 0; i < n; i ++)
        {
            comp = exp(j*pi*(idx*idx)*nInv);
            V_star[i] = onef/comp;
            U_l[i] = signal[i]/comp;
            V_l[i] = comp;
            V_l[l-i] = comp;
            idx+=1.0;
        }  
        fftblue(U_l,V_l,false);
        fft(U_l,true);
        for(int i = 0; i < n; i ++){dfft.push_back(U_l[i]*V_star[i]);}
    }
    vector<complex<float>> getFourCoeff(){ return dfft; } 
};


int main()
{
    vector<float> s = {1,2,3,4,5};
    Bluestein b2(s);
    vector<complex<float>> rst2 = b2.getFourCoeff();
    for (int i = 0; i < rst2.size(); i++) {
        cout << rst2[i] << " ";
    }
}
