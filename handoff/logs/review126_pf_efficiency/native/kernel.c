/* Same conditional Gaussian algebra, float64, no RNG or association here. */
#include <math.h>
#include <float.h>
#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

static void mul(const double *a,const double *b,double *out,int d,int transpose_b){
    for(int i=0;i<d;i++) for(int j=0;j<d;j++){
        double v=0.;
        for(int k=0;k<d;k++) v+=a[i*d+k]*b[transpose_b?j*d+k:k*d+j];
        out[i*d+j]=v;
    }
}

static void shape_predict(double *m,double *p,double dt,double q0,double q1){
    double f[4]={1.,dt,0.,1.},t[4],out[4];
    double x=m[0]+dt*m[1];
    mul(f,p,t,2,0);mul(t,f,out,2,1);
    for(int j=0;j<4;j++)p[j]=out[j];
    p[0]+=q0;p[3]+=q1;m[0]=x;
}

static void shape_update(double *m,double *p,double z,double r){
    double s=p[0]+r,k[2]={p[0]/s,p[2]/s};
    double residual=z-m[0],c[4]={1.-k[0],0.,-k[1],1.},t[4],out[4];
    m[0]+=k[0]*residual;m[1]+=k[1]*residual;
    mul(c,p,t,2,0);mul(t,c,out,2,1);
    for(int i=0;i<2;i++)for(int j=0;j<2;j++)out[i*2+j]+=(k[i]*k[j])*r;
    p[0]=out[0];p[3]=out[3];p[1]=p[2]=.5*(out[1]+out[2]);
}

API int predict(int n,double *means,double *cov,const double *f,
                double qpos,double qvel,double *shape_m,double *shape_p,double dt,int lost){
    for(int r=0;r<n;r++){
        double *m=means+4*r,*p=cov+16*r;const double *a=f+16*r;
        double next[4],t[16],out[16];
        for(int i=0;i<4;i++){
            next[i]=0.;for(int j=0;j<4;j++)next[i]+=a[4*i+j]*m[j];
        }
        mul(a,p,t,4,0);mul(t,a,out,4,1);
        for(int i=0;i<4;i++)m[i]=next[i];
        for(int i=0;i<16;i++)p[i]=out[i];
        p[0]+=qpos;p[5]+=qpos;p[10]+=qvel;p[15]+=qvel;
    }
    if(lost){shape_m[3]=0.;shape_p[5]=shape_p[6]=shape_p[7]=0.;}
    shape_predict(shape_m,shape_p,dt,.01*.01,.00001*.00001);
    shape_predict(shape_m+2,shape_p+4,dt,qpos,qvel);
    if(shape_m[2]<1e-3)shape_m[2]=1e-3;
    return 0;
}

API int update(int n,const double *means,const double *cov,const double *z,const double *r,
               double *next_m,double *next_p,double *whitened,double *chol_diag){
    for(int k=0;k<n;k++){
        const double *m=means+4*k,*p=cov+16*k;
        double a=p[0]+r[0],b=p[1]+r[1],c=p[4]+r[4],d=p[5]+r[5];
        double det=a*d-b*c;
        if(!(a>0.) || !isfinite(det) || det<=32.*DBL_EPSILON*(fabs(a*d)+fabs(b*c)))return k+1;
        double l0=sqrt(a),l10=c/l0,pivot=d-l10*l10;
        if(!(pivot>0.))return k+1;
        double l1=sqrt(pivot),residual[2]={z[0]-m[0],z[1]-m[1]},g[8];
        whitened[2*k]=residual[0]/l0;
        whitened[2*k+1]=(residual[1]-l10*whitened[2*k])/l1;
        chol_diag[2*k]=l0;chol_diag[2*k+1]=l1;
        for(int i=0;i<4;i++){
            g[2*i]=(d*p[4*i]-b*p[4*i+1])/det;
            g[2*i+1]=(a*p[4*i+1]-c*p[4*i])/det;
            next_m[4*k+i]=m[i]+(g[2*i]*residual[0]+g[2*i+1]*residual[1]);
        }
        double correction[16],t[16],out[16];
        for(int i=0;i<4;i++)for(int j=0;j<4;j++)
            correction[4*i+j]=(i==j?1.:0.)-(j<2?g[2*i+j]:0.);
        mul(correction,p,t,4,0);mul(t,correction,out,4,1);
        for(int i=0;i<4;i++)for(int j=0;j<4;j++){
            double v=0.;
            for(int x=0;x<2;x++)for(int y=0;y<2;y++)v+=g[2*i+x]*r[4*x+y]*g[2*j+y];
            out[4*i+j]+=v;
        }
        for(int i=0;i<4;i++)for(int j=0;j<4;j++)
            next_p[16*k+4*i+j]=.5*(out[4*i+j]+out[4*j+i]);
    }
    return 0;
}

API int shapes(double *m,double *p,const double *z,const double *r){
    shape_update(m,p,z[2],r[10]);shape_update(m+2,p+4,z[3],r[15]);
    if(m[2]<1e-3)m[2]=1e-3;
    return 0;
}

API int within(int n,const double *means,const double *cov,const double *points,
               const double *weights,double *output){
    for(int i=0;i<16;i++)output[i]=0.;
    for(int k=0;k<n;k++){
        const double *m=means+4*k,*p=cov+16*k;
        double speed=points[9*k+2];if(speed<1e-6)speed=1e-6;
        double j[16]={1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,m[2]/speed,m[3]/speed,
                       0.,0.,-m[3]/(speed*speed),m[2]/(speed*speed)};
        double t[16],v[16];mul(j,p,t,4,0);mul(t,j,v,4,1);
        for(int i=0;i<16;i++)output[i]+=weights[k]*v[i];
    }
    return 0;
}
