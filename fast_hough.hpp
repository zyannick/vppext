#ifndef HOUGH_IMAGE_HPP
#define HOUGH_IMAGE_HPP

#include "fast_hough.hh"
#include "video_extruder_hough.hh"
#include "feature_matching_hough.hh"
#include "multipointtracker.hh"
#include "operations.hh"

namespace vppx{

int SobelX[3][3] =
{
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

int SobelY[3][3] =
{
    1, 2, 1,
    0, 0, 0,
    -1,-2,-1
};


template <typename type_>
inline
vint2 inverse_cantor(type_ z)
{
    auto w = floor((sqrt(8 * z + 1) - 1)/2);
    auto t = (w*w + w) / 2;
    int y = int(z - t);
    int x = int(w -y);
    return vint2(x,y);
}

template <typename type_>
inline
type_ cantor(type_ k1,type_ k2)
{
    type_ k =  0.5 * (k1 + k2) * ( k1 + k2 + 1 ) + k2;
    return k;
}

template <typename type_>
inline
float function_diff(int k1,int k2)
{
    return fabs((k1-k2)/(k1+k2));
}


class compare_1 {
public:
    bool operator()(const int x,const int y) {
        return x>y;
    }
};


//namespace vppx {

inline void initializeGXSobel3x3()
{
    GxSobel3x3 << -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1;
    GxSobel3x3 = GxSobel3x3/4;

    GySobel3x3 << 1, 2, 1,
            0, 0, 0,
            -1,-2,-1;
    GySobel3x3 = GySobel3x3/4;
}

inline void initializeGYSobel3x3()
{
    GySobel3x3 << 1, 2, 1,
            0, 0, 0,
            -1,-2,-1;
    GySobel3x3 = GySobel3x3/4;
}

inline void initializeSobel5x5()
{
    GxSobel5x5 << 2 ,  1  , 0 ,  -1 , -2,
            3  , 2 ,  0  , -2 , -3,
            4  , 3 ,  0  , -3 , -4,
            3  , 2  , 0  , -2 , -3,
            2  , 1 ,  0 ,  -1 , -2;
    GxSobel5x5 = GxSobel5x5/4;

    GySobel5x5 << -2, -3, -4, -3 , -2,
            -1,-2,-3,-2,-1,
            0, 0, 0,0,0,
            1,2,3,2,1,
            2, 3, 4, 3 , 2;
    GySobel5x5 = GySobel5x5/4;
}



struct foreach_videoframe1
{

    foreach_videoframe1(const char* f)
    {
        open_videocapture(f, cap_);
        frame_ = vpp::image2d<vpp::vuchar3>(videocapture_domain(cap_));
        cvframe_ = to_opencv(frame_);
    }

    template <typename F>
    void operator| (F f)
    {
        while (cap_.read(cvframe_)) f(frame_);
    }

private:
    cv::Mat cvframe_;
    vpp::image2d<vpp::vuchar3> frame_;
    cv::VideoCapture cap_;
};

struct foreach_frame
{

    foreach_frame(const char* f)
    {
        open_videocapture(f, cap_);
        frame_ = vpp::image2d<vpp::vuchar3>(videocapture_domain(cap_));
        cvframe_ = to_opencv(frame_);
    }

    template <typename F>
    void operator| (F f)
    {
        while (cap_.read(cvframe_)){ f(from_opencv<vuchar3>(cvframe_));
            cout << "ben voyons "<< endl;
            //cv::imwrite("see.bmp", to_opencv(f));
        }
    }

private:
    cv::Mat cvframe_;
    vpp::image2d<vpp::vuchar3> frame_;
    cv::VideoCapture cap_;
};


std::list<vint2> Hough_Lines_Parallel(image2d<vuchar1> img,
                                      std::vector<float>& t_accumulator,
                                      int Theta_max, float& max_of_the_accu, int kernel_size)
{
    timer t;
    t.start();
    typedef vfloat3 F;
    typedef vuchar3 V;
    int ncols = img.ncols();
    int nrows = img.nrows();
    int rhomax = int(sqrt(pow(ncols,2)+pow(nrows,2)));
    std::list<vint2> interestedPoints;
    float T_theta = Theta_max;
    image2d<vuchar1> out(img.domain());
    std::vector<vshort2> coord_x(rhomax*T_theta,vshort2(-1,-1));
    std::vector<vshort2> coord_y(rhomax*T_theta,vshort2(-1,-1));

    //cout << "border " << img.border();
    pixel_wise(out, relative_access(img), img.domain()) | [&] (auto& o, auto i, vint2 coord) {
        int x = coord[1];
        int y = coord[0];
        if(x>kernel_size && y>kernel_size && x<ncols-kernel_size && y<nrows-kernel_size)
        {
            float dx = 0;
            float dy = 0;
            dx = -(i(1,-1)).coeffRef(0)  +  (i(1,1)).coeffRef(0)
                    -2* (i(0,-1)).coeffRef(0)  +  2*(i(0,1)).coeffRef(0)
                    - (i(-1,-1)).coeffRef(0)  + (i(-1,1)).coeffRef(0) ;
            dy = (i(1,-1)).coeffRef(0)  + 2*(i(1,0)).coeffRef(0)  + (i(1,1)).coeffRef(0)
                    - (i(-1,-1)).coeffRef(0)  -2* (i(-1,0)).coeffRef(0)  - (i(-1,1)).coeffRef(0) ;
            dx /= 4;
            dy /=4;
            float deltaI = sqrt( dx*dx + dy*dy);
            if(deltaI>0)
            {
                float d = x*dx + y*dy;
                float rho = fabs(d/deltaI);
                int index_rho = (int)trunc(rho);
                float poids_rho = 1 - rho + index_rho;
                float theta;
                if(dx*dy<0 && d*dy>0)
                    theta = M_PI + atan(dy/dx);
                else
                    theta = atan(dy/dx);
                float pos_theta = ((theta + M_PI)*(T_theta-1))/(2*M_PI);
                //Prendre les coordonnees
                int index_theta = (int)(trunc(pos_theta));
                float poids_theta =  1 - pos_theta + index_theta;
                float vote_total = deltaI;
#pragma omp critical
                {
                    //vshort2 x_c = coord_x[index_rho*T_theta + index_theta];
                    //vshort2 y_c = coord_y[index_rho*T_theta + index_theta];
                    if((coord_x[index_rho*T_theta + index_theta])[0] == -1
                            && (coord_y[index_rho*T_theta + index_theta])[0]==-1)
                    {
                           (coord_x[index_rho*T_theta + index_theta])[0] = x;
                            (coord_y[index_rho*T_theta + index_theta])[0] = y;
                    }
                    t_accumulator[index_rho*T_theta + index_theta] += vote_total*poids_rho*poids_theta;
                    (coord_x[index_rho*T_theta + index_theta])[1] = x;
                    (coord_y[index_rho*T_theta + index_theta])[1] = y;
                    if (poids_rho < 1)
                    {
                        if((coord_x[(index_rho+1)*T_theta + index_theta])[0] == -1
                                && (coord_y[(index_rho+1)*T_theta + index_theta])[0]==-1)
                        {
                               (coord_x[(index_rho+1)*T_theta + index_theta])[0] = x;
                                (coord_y[(index_rho+1)*T_theta + index_theta])[0] = y;
                        }
                        t_accumulator[(index_rho+1)*T_theta + index_theta] += vote_total*(1-poids_rho)*poids_theta;
                        (coord_x[(index_rho+1)*T_theta + index_theta])[1] = x;
                        (coord_y[(index_rho+1)*T_theta + index_theta])[1] = y;
                    }
                    if (poids_theta < 1)
                    {
                        if((coord_x[index_rho*T_theta + index_theta+1])[0] == -1
                                && (coord_y[index_rho*T_theta + index_theta+1])[0]==-1)
                        {
                               (coord_x[index_rho*T_theta + index_theta+1])[0] = x;
                                (coord_y[index_rho*T_theta + index_theta+1])[0] = y;
                        }
                        t_accumulator[index_rho*T_theta + index_theta+1] += vote_total*poids_rho*(1-poids_theta);
                        (coord_x[index_rho*T_theta + index_theta+1])[1] = x;
                        (coord_y[index_rho*T_theta + index_theta+1])[1] = y;
                    }
                    if ((poids_rho < 1)&&(poids_theta<1))
                    {
                        if((coord_x[(index_rho+1)*T_theta + index_theta+1])[0] == -1
                                && (coord_y[(index_rho+1)*T_theta + index_theta+1])[0]==-1)
                        {
                               (coord_x[(index_rho+1)*T_theta + index_theta+1])[0] = x;
                                (coord_y[(index_rho+1)*T_theta + index_theta+1])[0] = y;
                        }
                        t_accumulator[(index_rho+1)*T_theta + index_theta+1] += vote_total*(1-poids_rho)*(1-poids_theta);
                        (coord_x[(index_rho+1)*T_theta + index_theta+1])[1] = x;
                        (coord_y[(index_rho+1)*T_theta + index_theta+1])[1] = y;
                    }
                }
            }
            o = vuchar1(uchar(round(deltaI)));
        }
        else
            o = vuchar1(0);
    };


    cv::imwrite("sortieP.jpg", to_opencv(out));
    Mat cimg = to_opencv(out);
    Mat result;
    cvtColor(cimg,result,CV_GRAY2BGR);

    std::list<vfloat2> list_temp;

    for(int rho = 0 ; rho < rhomax ; rho ++ )
    {
        for(int theta = 0 ; theta < T_theta ; theta++)
        {
            if(t_accumulator[rho*T_theta + theta]>1000)
            {
                list_temp.push_back(vfloat2(t_accumulator[rho*T_theta + theta], cantor<int>(rho,theta)));
                //cout << t_accumulator[rho*T_theta + theta] << endl;
            }
        }
    }

    cout << " la taille " <<  list_temp.size() << endl;

    list_temp.sort( [&](vfloat2& a, vfloat2& b){return a[0] > b[0];});


    int lines_drawn = 0;
    //std::vector<vint2> liste_ligne;


    int nb_lines = 0;
    for ( auto& x : list_temp )
    {
        if(lines_drawn==0)
            max_of_the_accu = x[0];
        if(lines_drawn>100)
            break;
        int k = x[1];
        vint2 coord = inverse_cantor<float>(k);
        int found = 0;
        for(auto& it : interestedPoints  )
        {
            vint2 val = it;
            if(fabs(val[0]-coord[0])<5 && fabs(val[1]-coord[1])<5)
            {
                found = 1;
                break;

            }
        }

        if(found==0)
        {
#pragma omp critical
            interestedPoints.push_back(coord);
            int rho = coord[0];
            int theta = coord[1];
            //vint4 ligne = getLineFromPoint(rho,theta,T_theta,nrows,ncols);
            //cv::line(result, cv::Point(ligne[0], ligne[1]), cv::Point(ligne[2], ligne[3]), (0,255,255),1);
            cv::line(result, cv::Point((coord_x[rho*T_theta+theta])[0],(coord_y[rho*T_theta+theta])[0]), cv::Point((coord_x[rho*T_theta+theta])[1],(coord_y[rho*T_theta+theta])[1]), (0,255,255),1);
            lines_drawn++;
        }
    }

    //cout << "nombre " << lines_drawn << endl;
    cv::imwrite("okay.bmp", result);
    t.end();

    cout << "hough timer " << t.us() << endl;

    return interestedPoints;
}

std::list<vint2> Hough_Lines_Parallel_V2(image2d<vuchar1> img,
                                         std::vector<float>& t_accumulator,
                                         int Theta_max, float& max_of_the_accu, int threshold
                                         , image2d<vuchar3>& cluster_colors, int nb_old)
{
    timer t;
    t.start();
    typedef vfloat3 F;
    typedef vuchar3 V;
    int ncols = img.ncols();
    int nrows = img.nrows();
    int rhomax = int(sqrt(pow(ncols,2)+pow(nrows,2)));
    std::list<vint2> interestedPoints100;
    std::list<vint2> interestedPoints50;
    float T_theta = Theta_max;
    image2d<vuchar1> out(img.domain());
    //cout << "border " << img.border();
    pixel_wise(out, relative_access(img), img.domain()) | [&] (auto& o, auto i, vint2 coord) {
        int x = coord[1];
        int y = coord[0];
        if(x>3 && y>3 && x<ncols-3 && y<nrows-3)
        {
            float dx = 0;
            float dy = 0;
            dx = -(i(1,-1)).coeffRef(0)  +  (i(1,1)).coeffRef(0)
                    -2* (i(0,-1)).coeffRef(0)  +  2*(i(0,1)).coeffRef(0)
                    - (i(-1,-1)).coeffRef(0)  + (i(-1,1)).coeffRef(0) ;
            dy = (i(1,-1)).coeffRef(0)  + 2*(i(1,0)).coeffRef(0)  + (i(1,1)).coeffRef(0)
                    - (i(-1,-1)).coeffRef(0)  -2* (i(-1,0)).coeffRef(0)  - (i(-1,1)).coeffRef(0) ;
            dx /= 4;
            dy /=4;
            float deltaI = sqrt( dx*dx + dy*dy);
            if(deltaI>100)
            {
                float d = x*dx + y*dy;
                float rho = fabs(d/deltaI);
                int index_rho = (int)trunc(rho);
                float poids_rho = 1 - rho + index_rho;
                float theta;
                float alpha;

                /*// Calcul de l'angle de la tangente au contour
                if (dx) alpha = atan(dy/dx); else alpha = M_PI/2;
                // Calcul de theta, l'angle entre Ox et la droite
                // passant par O et perpendiculaire à la droite D
                if (alpha < 0) theta = M_PI/2 + alpha;
                else {
                    // 1er cas : D est en dessous de O
                    // -PI/2 < Theta < 0 ; cx*j - cy*i < 0
                    if (dx*x - dy*y < 0) theta = alpha - M_PI/2;
                    // 2eme cas : D est au dessus de O
                    // PI/2 < Theta < PI ; cx*j - cy*i > 0
                    else theta = alpha + M_PI/2;
                }*/
                if(dx*dy<0 && d*dy>0)
                    theta = M_PI + atan(dy/dx);
                else
                    theta = atan(dy/dx);
                float pos_theta = ((theta + M_PI)*(T_theta-1))/(2*M_PI);
                //Prendre les coordonnees
                int index_theta = (int)(trunc(pos_theta));
                float poids_theta =  1 - pos_theta + index_theta;
                float vote_total = deltaI;
#pragma omp critical
                {
                    t_accumulator[index_rho*T_theta + index_theta] += vote_total*poids_rho*poids_theta;
                    if (poids_rho < 1)
                    {
                        t_accumulator[(index_rho+1)*T_theta + index_theta] += vote_total*(1-poids_rho)*poids_theta;
                    }
                    if (poids_theta < 1)
                    {
                        t_accumulator[index_rho*T_theta + index_theta+1] += vote_total*poids_rho*(1-poids_theta);
                    }
                    if ((poids_rho < 1)&&(poids_theta<1))
                    {
                        t_accumulator[(index_rho+1)*T_theta + index_theta+1] += vote_total*(1-poids_rho)*(1-poids_theta);
                    }
                }
            }
            o = vuchar1(uchar(round(deltaI)));
        }
        else
            o = vuchar1(0);
    };


    cv::imwrite("sortieP.jpg", to_opencv(out));
    Mat cimg = to_opencv(out);
    Mat result;
    cvtColor(cimg,result,CV_GRAY2BGR);

    std::list<vfloat3> list_temp;

    float threshold_hough = 500;

    for(int rho = 0 ; rho < rhomax ; rho ++ )
    {
        for(int theta = 0 ; theta < T_theta ; theta++)
        {
            if(t_accumulator[rho*T_theta + theta]>threshold_hough)
            {
                list_temp.push_back(vfloat3(t_accumulator[rho*T_theta + theta], rho,theta));
            }
        }
    }


    cout << " la taille " <<  list_temp.size() << endl;

    list_temp.sort( [&](vfloat3& a, vfloat3& b){return a[0] > b[0];});



    //std::vector<vint2> liste_ligne;


    int nb_lines = 0;
    for ( auto& x : list_temp )
    {
        if(nb_lines==0)
            max_of_the_accu = x[0];
        if(nb_lines>100)
            break;
        vint2 coord;
        coord = vint2(x[1],x[2]);
        int found = 0;
        for(auto& it : interestedPoints100  )
        {
            vint2 val = it;
            if(fabs(val[0]-coord[0])<10 && fabs(val[1]-coord[1])<10)
            {
                found = 1;
                break;
            }
        }

        if(found==0)
        {
            interestedPoints100.push_back(coord);
            int theta = coord[1];
            int rho = coord[0];
            cout << "theta " << theta << " rho " << rho << endl;
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

            //cout << " rho " << r << " theta " << t << " max " << max << endl;

            if(fabs(cosinus)<0.01)
            {
                x1 = ncols-1;
                y1 = y2;
                //continue;
            }

            cv::line(result, cv::Point(x1, y1), cv::Point(x2, y2), (0,255,255),1);
            //cout << " nb " << nb_lines << endl;
            nb_lines++;
        }
    }

    std::list<vfloat3> list_temp1;

    for(auto &l_c : interestedPoints100)
    {
        int rho = l_c[0];
        int theta = l_c[1];
        float moy = 0;
        int red_value = 0;
        int blue_value = 0;
        int green_value = 0;
        for(int cp_rho = rho-2; cp_rho <= rho +2 ; cp_rho++)
        {
            for(int cp_theta = theta - 2 ; cp_theta <= theta +2 ; cp_theta++ )
            {
                moy += t_accumulator[cp_rho*T_theta + cp_theta];
            }
        }
        moy /= 25;
        list_temp1.push_back(vfloat3(moy,rho,theta));
    }

    list_temp1.sort( [&](vfloat3& a, vfloat3& b){return a[0] > b[0];});


    int nb = 0;

    for(auto &l_c : list_temp1)
    {
        interestedPoints50.push_back(vint2(l_c[1],l_c[2]));
        if(nb>=50)
            break;
        nb++;
    }


    //cout << "nombre " << lines_drawn << endl;
    cv::imwrite("okay.bmp", result);
    t.end();

    cout << "hough timer " << t.us() << endl;

    return interestedPoints50;
}

void adap_thresold(std::list<vfloat3> &list_temp , float &threshold_hough , int &calls ,
                   int &nb_calls_limits_reached , int rhomax, int T_theta , std::vector<float> t_accumulator)
{
    if(calls>=5)
    {
        nb_calls_limits_reached=1;
        return;
    }
    if(list_temp.size() < 100 && list_temp.size() >50)
    {
        return;
    }
    else if(list_temp.size() > 100 )
    {
        calls++;
        threshold_hough *= calls;
        reduce_number_of_max_local(list_temp,threshold_hough,rhomax,T_theta,t_accumulator);
    }
    else if(list_temp.size())
    {
        calls++;
        threshold_hough /=calls;
        reduce_number_of_max_local(list_temp,threshold_hough,rhomax,T_theta,t_accumulator);
    }
}

void reduce_number_of_max_local(std::list<vfloat3> &list_temp , float threshold_hough , int rhomax, int T_theta , std::vector<float> t_accumulator)
{
    list_temp.clear();
    for(int rho = 0 ; rho < rhomax ; rho ++ )
    {
        for(int theta = 0 ; theta < T_theta ; theta++)
        {
            if(t_accumulator[rho*T_theta + theta]>threshold_hough)
            {
                list_temp.push_back(vfloat3(t_accumulator[rho*T_theta + theta], rho,theta));
            }
        }
    }
}


void interpolate_acculator(image2d<float> &acc, float seuil)
{
    image2d<float> out(acc.domain());
    pixel_wise(out, relative_access(acc), acc.domain()) | [&] (auto& o, auto i, vint2 coord) {
        int x = coord[1];
        int y = coord[0];
        if(x>3 && y>3 && x<acc.ncols()-3 && y<acc.ncols()-3 && i(0,0) > seuil)
        {
            o = i(0,0) + i(0,1) + i(0,-1) + i(1,0) ;
        }
    };
}

std::list<vint2> Hough_Lines_Parallel_new(image2d<vuchar1> img,
                                          std::vector<float>& t_accumulator,
                                          int Theta_max, float& max_of_the_accu, int kernel_size)
{
    timer t;
    t.start();
    typedef vfloat3 F;
    typedef vuchar3 V;
    int ncols = img.ncols();
    int nrows = img.nrows();
    int rhomax = int(sqrt(pow(ncols,2)+pow(nrows,2)));
    std::list<vint2> interestedPoints;
    float T_theta = Theta_max;
    image2d<vuchar1> out(img.domain());
    //cout << "border " << img.border();
    pixel_wise(out, relative_access(img), img.domain()) | [&] (auto& o, auto i, vint2 coord) {
        int x = coord[1];
        int y = coord[0];
        if(x>kernel_size && y>kernel_size && x<ncols-kernel_size && y<nrows-kernel_size)
        {
            float dx = 0;
            float dy = 0;
            dx = -(i(1,-1)).coeffRef(0)  +  (i(1,1)).coeffRef(0)
                    -2* (i(0,-1)).coeffRef(0)  +  2*(i(0,1)).coeffRef(0)
                    - (i(-1,-1)).coeffRef(0)  + (i(-1,1)).coeffRef(0) ;
            dy = (i(1,-1)).coeffRef(0)  + 2*(i(1,0)).coeffRef(0)  + (i(1,1)).coeffRef(0)
                    - (i(-1,-1)).coeffRef(0)  -2* (i(-1,0)).coeffRef(0)  - (i(-1,1)).coeffRef(0) ;
            dx /= 4;
            dy /=4;
            float deltaI = sqrt( dx*dx + dy*dy);
            if(deltaI>100)
            {
                float d = x*dx + y*dy;
                float rho = fabs(d/deltaI);
                int index_rho = (int)trunc(rho);
                float poids_rho = 1 - rho + index_rho;
                float theta;
                if(dx*dy<0 && d*dy>0)
                    theta = M_PI + atan(dy/dx);
                else
                    theta = atan(dy/dx);
                float pos_theta = ((theta + M_PI)*(T_theta-1))/(2*M_PI);
                //Prendre les coordonnees
                int index_theta = (int)(trunc(pos_theta));
                float poids_theta =  1 - pos_theta + index_theta;
                float vote_total = deltaI;
#pragma omp critical
                {
                    t_accumulator[index_rho*T_theta + index_theta] += vote_total*poids_rho*poids_theta;
                    if (poids_rho < 1)
                        t_accumulator[(index_rho+1)*T_theta + index_theta] += vote_total*(1-poids_rho)*poids_theta;
                    if (poids_theta < 1)
                        t_accumulator[index_rho*T_theta + index_theta+1] += vote_total*poids_rho*(1-poids_theta);
                    if ((poids_rho < 1)&&(poids_theta<1))
                        t_accumulator[(index_rho+1)*T_theta + index_theta+1] += vote_total*(1-poids_rho)*(1-poids_theta);
                }
            }
            o = vuchar1(uchar(round(deltaI)));
        }
        else
            o = vuchar1(0);
    };


    cv::imwrite("sortieP.jpg", to_opencv(out));
    Mat cimg = to_opencv(out);
    Mat result;
    cvtColor(cimg,result,CV_GRAY2BGR);

    std::list<vfloat3> local_maximax;


    //#pragma omp parallel for
    for(int rho = 0 ; rho < rhomax ; rho = rho + 10 )
    {
        for(int theta = 0 ; theta < T_theta ; theta = theta + 10)
        {
            float max = t_accumulator[rho*T_theta + theta];
            int r_max = rho;
            int t_max = theta;
            float moy = 0;
            for(int ly=0;ly<=9;ly++)
            {
                for(int lx=0;lx<=9;lx++)
                {
                    if( ( ly+rho<rhomax) && (lx+theta<T_theta) )
                    {
                        moy = t_accumulator[( (rho+ly)*T_theta) + (theta+lx)];
                        if( t_accumulator[( (rho+ly)*T_theta) + (theta+lx)] > max )
                        {
                            max = t_accumulator[( (rho+ly)*T_theta) + (theta+lx)];
                            r_max = rho+ly;
                            t_max = theta+lx;
                        }
                    }
                }
            }
            moy /= 25;

            t_accumulator[rho*T_theta + theta] = moy;
            local_maximax.push_back(vfloat3(r_max,t_max,moy));

        }

    }

    local_maximax.sort( [&](vfloat3& a, vfloat3& b){return a[2] > b[2];});

    int nb = 0;

    for(auto &l : local_maximax)
    {
        interestedPoints.push_back(vint2(l[0],l[1]));
        if(nb==0)
            max_of_the_accu = l[2];
        nb++;
        if(nb>=10)
            break;
    }


    return interestedPoints;
}

std::list<vint2> Hough_Lines_Parallel_Box(image2d<vuchar1> img,
                                          std::vector<float>& t_accumulator,
                                          int Theta_max, float& max_of_the_accu)
{
    typedef vfloat3 F;
    typedef vuchar3 V;
    int ncols = img.ncols();
    int nrows = img.nrows();
    int rhomax = int(sqrt(pow(ncols,2)+pow(nrows,2)));
    std::list<vint2> interestedPoints;
    float T_theta = Theta_max;
    image2d<vuchar1> out(img.domain());
    //cout << "border " << img.border();


    pixel_wise(out, relative_access(img), img.domain()) | [&] (auto& o, auto i, vint2 coord) {

        // a, b and c are sub images representing A, B
        // and C at the current cell.
        // All the algorithms of the library work on sub images.

        // The box argument is the cell representation of A.domain() and holds
        // the coordinates of the current cell. box.p1() and box.p2() are
        // respectively the first and the last pixel of the cell.
    };


    return interestedPoints;
}





void Hough_Lines_Parallel_Map(image2d<vuchar1> img)
{
    std::map<int,float> accumulator_high;
    std::list<vfloat2> list_accumulator;
    float T_theta = 100;
    int ncols = img.ncols();
    int nrows = img.nrows();
    image2d<vuchar1> out(img.domain());
    int cp = 0;
    int nb_cluster = 0;
    pixel_wise(out, relative_access(img), img.domain()) | [&] (auto& o, auto i, vint2 coord) {
        int x = coord[1];
        int y = coord[0];
        if(x>3 && y>3 && x<ncols-3 && y<nrows-3)
        {
            float dx = -(i(1,-1)).coeffRef(0)  +  (i(1,1)).coeffRef(0)
                    -2* (i(0,-1)).coeffRef(0)  +  2*(i(0,1)).coeffRef(0)
                    - (i(-1,-1)).coeffRef(0)  + (i(-1,1)).coeffRef(0) ;
            float dy = (i(1,-1)).coeffRef(0)  + 2*(i(1,0)).coeffRef(0)  + (i(1,1)).coeffRef(0)
                    - (i(-1,-1)).coeffRef(0)  -2* (i(-1,0)).coeffRef(0)  - (i(-1,1)).coeffRef(0) ;
            dx /= 4;
            dy /=4;
            float deltaI = sqrt( dx*dx + dy*dy);
            if(deltaI>125)
            {
                float d = x*dx + y*dy;
                float rho = fabs(d/deltaI);
                int index_rho = (int)trunc(rho);
                //float poids_rho = 1 - rho + index_rho;
                float theta;
                if(dx*dy<0 && d*dy>0)
                    theta = M_PI + atan(dy/dx);
                else
                    theta = atan(dy/dx);
                float pos_theta = ((theta + M_PI)*(T_theta-1))/(2*M_PI);
                int index_theta = (int)(trunc(pos_theta));
                //float poids_theta =  1 - pos_theta + index_theta;
#pragma omp critical
                {
                    int k = 0.5 * (index_rho + index_theta) * ( index_rho + index_theta + 1 ) + index_theta;
                    float value=0;
                    auto s = accumulator_high.find(k);
                    if(s!=accumulator_high.end())
                    {
                        value = s->second;
                    }
                    //accumulator_high[k] = deltaI*poids_rho*poids_theta + value;
                    accumulator_high[k] = deltaI + value;
                    nb_cluster++;
                }
            }
            o = vuchar1(uchar(round(deltaI)));
        }
        else
            o = vuchar1(0);
        ++cp;
    };
    cv::imwrite("sortieP.jpg", to_opencv(out));
    Mat result;
    cvtColor(to_opencv(out),result,CV_GRAY2BGR);
    int taille_map = accumulator_high.size();

    for ( auto const &val : accumulator_high)
    {
        vfloat2 vg(val.first,val.second);
        list_accumulator.push_back(vg);
    }
    list_accumulator.sort( [&](vfloat2& a, vfloat2& b){return a[1] > b[1];});

    int lines_taced = 0;
    std::vector<vint2> liste_ligne;
    //
    //#pragma omp parallel
    for ( auto& x : list_accumulator )
    {
        if(lines_taced>100)
            break;
        int k = x[0];
        vint2 coord = inverse_cantor<float>(k);
        //cout << " boule " << coord << endl;
        int found = 0;
        //cout << "affiche" << endl;
        for(int i = 0; i < liste_ligne.size() ; i++  )
        {
            vint2 val = liste_ligne[i];
            //cout << val << endl << endl;
            if(fabs(val[0]-coord[0])<5 && fabs(val[1]-coord[1])<5)
            {   //cout << " trouve " << coord << endl << endl ;
                found = 1;
                break;
            }
        }

        if(found==0 && x[1]>1000)
        {
#pragma omp critical
            liste_ligne.push_back(coord);
            //cout << "taille actuelle " << liste_ligne.size() << endl;
            int rho = coord[0];
            int theta = coord[1];
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
            cv::line(result, cv::Point(x1, y1), cv::Point(x2, y2), (0,255,255),1);
            lines_taced++;
        }
    }
    /*for (int i=0;i < liste_ligne.size();i++)
        cout << liste_ligne[i] << endl << endl;*/
    cout << " taille arbre " << taille_map << endl;
    cout << "nombre" << lines_taced << endl;
    cv::imwrite("okay.jpg", result);
}


void Hough_Accumulator(image2d<vuchar1> img, int mode, int T_theta)
{
    typedef vfloat3 F;
    typedef vuchar3 V;
    int ncols=img.ncols(),nrows=img.nrows();
    int rhomax = int(sqrt(pow(ncols,2)+pow(nrows,2)));
    int thetamax = T_theta;
    float max_of_accu = 0;
    std::vector<float> t_accumulator(rhomax*thetamax);
    std::fill(t_accumulator.begin(),t_accumulator.end() , 0);
    std::list<vint2> interestedPoints;
    if(mode==hough_parallel)
    {
        cout << "Parallel with a image" << endl;
        interestedPoints = Hough_Lines_Parallel(img,t_accumulator,thetamax,max_of_accu,3);
    }
}

Mat Hough_Accumulator_Video_Map_and_Clusters(image2d<vuchar1> img, int mode, int T_theta,
                                             std::vector<float>& t_accumulator, std::list<vint2> &interestedPoints, float rhomax)
{
    typedef vfloat3 F;
    typedef vuchar3 V;
    float max_of_accu = 0;

    if(mode==hough_parallel)
    {
        cout << "Parallel" << endl;
        interestedPoints = Hough_Lines_Parallel(img,t_accumulator,T_theta,max_of_accu,5);
        return accumulatorToFrame(t_accumulator,
                                  max_of_accu
                                  ,rhomax,T_theta);
    }
}

cv::Mat Hough_Accumulator_Video_Clusters(image2d<vuchar1> img, int mode , int T_theta,
                                         std::vector<float>& t_accumulator, std::list<vint2>& interestedPoints, float rhomax)
{
    typedef vfloat3 F;
    typedef vuchar3 V;
    float max_of_accu = 0;

    if(mode==hough_parallel)
    {
        cout << "Parallel" << endl;
        interestedPoints = Hough_Lines_Parallel(img,t_accumulator,T_theta,max_of_accu,5);
        return accumulatorToFrame(interestedPoints,rhomax,T_theta);
    }
}


int getThetaMax(Theta_max discr)
{
    if(Theta_max::XLARGE == discr)
        return 1500;
    else if(Theta_max::LARGE == discr)
        return 1000;
    else if(Theta_max::MEDIUM == discr)
        return 500;
    else if(Theta_max::SMALL == discr)
        return 255;
    else
        return 0;
}

cv::Mat accumulatorToFrame(std::vector<float> t_accumulator, float big_max, int rhomax, int T_theta)
{
    Mat T(int(rhomax),int(T_theta),CV_8UC1);
    for(int rho = 0 ; rho < rhomax ; rho ++ )
    {
        for(int theta = 0 ; theta < T_theta ; theta++)
        {
            T.at<uchar>(rho,theta) = ( t_accumulator[rho*T_theta + theta] * big_max ) / 255;
        }
    }
    return T;
}

cv::Mat accumulatorToFrame(std::list<vint2> interestedPoints, int rhomax, int T_theta)
{
    Mat T = Mat(int(rhomax),int(T_theta),CV_8UC1,cvScalar(0));
    int r = 0;
    for(auto& ip : interestedPoints)
    {
        int radius = round(5-0.1*r);
        circle(T,cv::Point(ip[1],ip[0]),radius,Scalar(255),CV_FILLED,8,0);
        r++;
        //break;
    }
    return T;
}

void Capture_Image(int mode, Theta_max discr,Type_video_hough type_video)
{
    typedef image2d<vuchar1> Image;
    char* link = "videos/endgp.avi";
    int T_theta = getThetaMax(discr);
    initializeGXSobel3x3();
    initializeGYSobel3x3();
    if(mode==mode_capture_photo)
    {
        Mat bv = cv::imread("m.png",0);
        Image img = (from_opencv<vuchar1>(bv));
        Hough_Accumulator(img,hough_parallel,T_theta);
    }
    else if(mode==mode_capture_video)
    {
        cv::VideoCapture cap("videos/psychedelic-white-lines-hd-animation.avi");
        Mat frame,bv;
        int ranked = 0;
        int ncols=0;
        int nrows=0;
        int rhomax = 0;
        bool ctrl = cap.read(frame);
        if(ctrl)
        {
            ncols = frame.cols;
            nrows = frame.rows;
            rhomax = int(sqrt(pow(ncols,2)+pow(nrows,2)));
        }
        else
        {
            return;
        }
        Size size2 = Size(T_theta, rhomax);
        int codec = CV_FOURCC('M', 'J', 'P', 'G');
        VideoWriter writer2("videos/video_.avi", codec, 1.0, size2, false);
        writer2.open("videos/video_.avi", codec, 1.0, size2, false);
        std::list<vint2> interestedPoints;
        box2d domain = vpp::make_box2d(T_theta,rhomax);
        //video_extruder_ctx ctx = video_extruder_init(domain);
        image2d<unsigned char> prev_frame(domain);
        bool first = true;
        int nframes = 0;
        int us_cpt = 0;
        while(ctrl && ranked<100){
            if(frame.channels()>1)
                cv::cvtColor(frame, bv, cv::COLOR_BGR2GRAY);
            else
                bv = frame;
            Image img = (from_opencv<vuchar1>(bv));

            std::vector<float> t_accumulator(rhomax*T_theta,0);
            //Hough_Accumulator_Video(img,hough_parallel,T_theta,t_accumulator,rhomax);
            //

            // writer2.write(Hough_Accumulator_Video_Map_and_Clusters(img,hough_parallel,T_theta,t_accumulator,interestedPoints,rhomax));
            string filename = "result";
            if(ranked<10)
                filename = filename +"0"+ std::to_string(ranked);
            else
                filename = filename +std::to_string(ranked);
            //cv::imwrite("images/"+filename+".bmp", Hough_Accumulator_Video_Map_and_Clusters(img,hough_parallel,T_theta,t_accumulator,interestedPoints,rhomax));
            //cv::imwrite("images/"+filename+".bmp", Hough_Accumulator_Video_Clusters(img,hough_parallel,T_theta,t_accumulator,interestedPoints,t_accumulator_point,rhomax));
            ctrl = cap.read(frame);
            ranked++;
        }
    }
    else if(mode==mode_capture_try)
    {
        int cp=0;
        bool first = true;
        int nframes = 0;
        //MultiPointTracker multitracking(vint2(0,0),20, 0.1,10, 0.2,0.5,20, 0.1,2);
        cv::VideoWriter output_video;
        image2d<uchar> prev_frame(make_box2d(1,1));
        image2d<uchar> frame_gl(make_box2d(1,1));
        video_extruder_hough_ctx ctx= video_extruder_hough_init(make_box2d(1,1));
        keypoint_container<keypoint<int>, int> listes_clusters(make_box2d(1,1));
        int us_cpt = 0;
        foreach_videoframe1(link)| [&] (const image2d<vuchar3>& frame_cv){
            if(nframes%1==0)
            {
                auto frame = rgb_to_graylevel<vuchar1>(frame_cv);
                image2d<vuchar3> dr(frame.domain());
                vpp::copy(dr, frame_cv);
                timer t;
                t.start();
                int ncols = frame.ncols();
                int nrows = frame.nrows();
                int rhomax = int(sqrt(pow(ncols,2)+pow(nrows,2)));
                box2d domain = make_box2d(rhomax,T_theta);
                if(first)
                {
                    ctx = video_extruder_hough_init(domain);
                    keypoint_container<keypoint<int>, int> temp_container(domain);
                    listes_clusters = temp_container;
                    //listes_clusters(make_box2d(1,1));
                    image2d<uchar> prev_frame_temp(domain);
                    prev_frame = prev_frame_temp;
                    image2d<uchar> frame_gl_temp(domain);
                    frame_gl = frame_gl_temp;
                    output_video.open("videos/video_.avi", cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 30.f,
                                      cv::Size(T_theta,rhomax), true);
                    first = false;
                }
                else
                {
                    video_extruder_hough_update(ctx, prev_frame, frame_gl, frame,int(type_video),
                                                _detector_th = 10,
                                                _keypoint_spacing = 10,
                                                _detector_period = 1,
                                                _max_trajectory_length = 100);
                }
                t.end();

                us_cpt += t.us();
                if (!(nframes%10))
                {
                    std::cout << "Tracker time: " << (us_cpt / 10000.f) << " ms/frame. " << ctx.trajectories.size() << " particles." << std::endl;
                    us_cpt = 0;
                }
                vpp::copy(frame_gl, prev_frame);
                auto display = graylevel_to_rgb<vuchar3>(frame_gl);
                draw::draw_trajectories(display, ctx.trajectories, 3/*,T_theta,nrows,ncols*/);
                cout << " frame no " << nframes << endl;
                /*string filename = "result";
                if(nframes<10)
                    filename = filename +"0"+ std::to_string(nframes);
                else
                    filename = filename +std::to_string(nframes);
                imwrite(filename+".bmp",to_opencv(dr));*/
                if (output_video.isOpened())
                    output_video << to_opencv(dr);
            }

            nframes++;
        };
    }
    else if(mode == mode_capture_vid_try)
    {
        box2d domain = videocapture_domain(link);
        video_extruder_ctx ctx = video_extruder_init(domain);

        cv::VideoWriter output_video;

        output_video.open("videos/video_.avi", cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 30.f,
                          cv::Size(domain.ncols(), domain.nrows()), true);


        image2d<unsigned char> prev_frame(domain);

        bool first = true;
        int nframes = 0;

        int us_cpt = 0;
        foreach_videoframe(link) | [&] (const image2d<vuchar3>& frame_cv)
        {
            auto frame = clone(frame_cv, _border = 3);
            fill_border_mirror(frame);
            auto frame_gl = rgb_to_graylevel<unsigned char>(frame);
            timer t;
            t.start();
            if (!first)
                video_extruder_update(ctx, prev_frame, frame_gl,
                                      _detector_th = 10,
                                      _keypoint_spacing = 10,
                                      _detector_period = 1,
                                      _max_trajectory_length = 100);
            else first = false;
            t.end();

            us_cpt += t.us();
            if (!(nframes%100))
            {
                std::cout << "Tracker time: " << (us_cpt / 100000.f) << " ms/frame. " << ctx.trajectories.size() << " particles." << std::endl;
                us_cpt = 0;
            }

            vpp::copy(frame_gl, prev_frame);
            auto display = clone(frame);
            //draw::draw_trajectories(display, ctx.trajectories, 200);
            //cv::imshow("Trajectories", to_opencv(display));
            //cv::waitKey(1);

            if (output_video.isOpened())
                output_video << to_opencv(display);

            nframes++;
        };
    }
    else if(mode==mode_capture_webcam)
    {
        cv::VideoCapture cap(0); // open the default camera
        if(!cap.isOpened()){
            cout<<"Camera could not load..."<<endl;
            return;
        }
        else
        {
            cout << " it's okay " << endl;
        }
        while(1){
            Mat frame,bv;
            bool ctrl = cap.read(frame);
            if(frame.channels()>1)
                cv::cvtColor(frame, bv, cv::COLOR_BGR2GRAY);
            else
                bv = frame;
            Image img = clone((from_opencv<vuchar1>(bv)), vpp::_border = 3);
            Hough_Accumulator(img,hough_parallel_map,T_theta);
            imshow("webcam",frame);
            if(waitKey(0) == 27){
                cout<<"The app is ended..."<<endl;
                break;
            }
        }
        namedWindow("webcam",CV_WINDOW_AUTOSIZE);
    }
}

}



#endif // HOUGH_IMAGE_HPP
