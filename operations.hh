#ifndef OPERATIONS_HH
#define OPERATIONS_HH

#include <vpp/vpp.hh>
using namespace vpp;
using namespace std;



inline vint4 getLineFromPoint(int rho,int theta,int T_theta,int nrows,int ncols)
{
    int x1,x2,y1,y2;
    x1=x2=y1=y2=0;
    //intersection avec l'axe des x
    float t = ((2*M_PI*(theta))/ (T_theta-1)) - M_PI;
    float r = rho;

    y1=0;
    float cosinus = cos(t);
    if(fabs(cosinus)>0.01)
    {
        x1=int(r/cosinus);
        if(cosinus<0)
        {
            for(int b = nrows -1 ; b >=0 ; b--)
            {
                x1 = (int)round((rho - b*sin(t))/cos(t));
                if(x1>0)
                {
                    y1 = b;
                    break;
                }
            }
        }
    }
    //intersection avec l'axe des y
    x2=0;
    float sinus = sin(t);
    if(fabs(sinus) < 0.01)
    {
        x2 = x1;
        y2 = nrows-1;
    }
    else
    {
        y2=int(r/sinus);
        if(y2<0)
        {
            for(int a = ncols-1 ; a>=0 ; a--)
            {
                y2 = (int)round((rho - a*cos(t))/sin(t));
                if(y2>0)
                {
                    x2 = a;
                    break;
                }
            }
        }
    }
    //cout << " rho " << r << " theta " << t << " val " << x[1] <<  endl;
    if(fabs(cosinus)<0.01)
    {
        x1 = ncols-1;
        y1 = y2;
        //continue;
    }
    return vint4(x1,y1,x2,y2);
}

#endif // OPERATIONS_HH
