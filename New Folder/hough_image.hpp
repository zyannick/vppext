
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


///namespace vppx {
///
///
/// */
///
///
float Hough_Lines_Parallel(image2d<vuchar1> img, std::vector<float>& t_accumulator, std::list<vint2>& interestedPoints, std::vector<vshort4> &t_accumulator_point, int Theta_max)
{
    typedef vfloat3 F;
    typedef vuchar3 V;
    interestedPoints.clear();
    int ncols = img.ncols();
    int nrows = img.nrows();
    int rhomax = int(sqrt(pow(ncols,2)+pow(nrows,2)));
    Matrix<float,3,3> GxSobel;
    GxSobel << -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1;
    GxSobel = GxSobel/4;
    Matrix<float,3,3> GySobel;
    GySobel << 1, 2, 1,
            0, 0, 0,
            -1,-2,-1;
    GySobel = GySobel/4;
    Matrix<float,3,3> Gaus3x3;
    Gaus3x3 <<  1, 2, 1,
            2, 4, 2,
            1, 2, 1;
    float T_theta = Theta_max;
    image2d<vuchar1> out(img.domain());

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
                    t_accumulator[index_rho*T_theta + index_theta] += vote_total*poids_rho*poids_theta;
                    vshort4 vect_temp = t_accumulator_point[index_rho*T_theta + index_theta];
                    if(x > vect_temp[0])
                    {
                        vect_temp[0] = x;
                        vect_temp[1] = y;
                    }
                    if(x < vect_temp[2])
                    {
                        vect_temp[2] = x;
                        vect_temp[3] = y;
                    }
                    t_accumulator_point[index_rho*T_theta + index_theta] = vect_temp;
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
    float big_max = 0;

    for ( auto& x : list_temp )
    {
        if(lines_drawn==0)
            big_max = x[0];
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
            int x1,x2,y1,y2;
            x1=x2=y1=y2=0;
            vshort4 vect_temp = t_accumulator_point[rho*T_theta + theta];
            x1 = vect_temp[0];
            y1 = vect_temp[1];
            x2 = vect_temp[2];
            y2 = vect_temp[3];
            cv::line(result, cv::Point(x1, y1), cv::Point(x2, y2), (0,255,255),1);
            lines_drawn++;
        }
    }

    cout << "nombre " << lines_drawn << endl;
    cv::imwrite("okay.bmp", result);
    return big_max;
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
    int nb_max = int(taille_map*0.4) > 300 ? 300 : int(taille_map*0.4);
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

        if(found==0 && x[1]>500)
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
    std::vector<float> t_accumulator(rhomax*thetamax);
    std::vector<vshort4> t_accumulator_point(rhomax*T_theta, vshort4(0,0,ncols,nrows));
    std::list<vint2> interestedPoints;
    for(int i = 0 ; i < rhomax*thetamax ; i++)
    {
        t_accumulator[i]=0;
    }

    if(mode==hough_parallel)
    {
        cout << "Parallel" << endl;
        Hough_Lines_Parallel(img,t_accumulator,interestedPoints,t_accumulator_point,thetamax);
    }
}

Mat Hough_Accumulator_Video_Map_and_Clusters(image2d<vuchar1> img, int mode, int T_theta, std::vector<float>& t_accumulator, std::list<vint2> &interestedPoints,std::vector<vshort4> &t_accumulator_point, float rhomax)
{
    typedef vfloat3 F;
    typedef vuchar3 V;


    if(mode==hough_parallel)
    {
        cout << "Parallel" << endl;
       return accumulatorToFrame(t_accumulator,Hough_Lines_Parallel(img,t_accumulator,interestedPoints,t_accumulator_point,T_theta),rhomax,T_theta);
    }
}

cv::Mat Hough_Accumulator_Video_Clusters(image2d<vuchar1> img, int mode , int T_theta, std::vector<float>& t_accumulator, std::list<vint2>& interestedPoints, std::vector<vshort4> &t_accumulator_point, float rhomax)
{
    typedef vfloat3 F;
    typedef vuchar3 V;


    if(mode==hough_parallel)
    {
        cout << "Parallel" << endl;
        Hough_Lines_Parallel(img,t_accumulator,interestedPoints,t_accumulator_point,T_theta);
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

cv::Mat accumulatorToFrame(std::vector<float> t_accumulator,float big_max, float rhomax,int T_theta)
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

cv::Mat accumulatorToFrame(std::list<vint2> interestedPoints, float rhomax, int T_theta)
{
    Mat T = Mat(int(rhomax),int(T_theta),CV_8UC1,cvScalar(0));
    for(auto& ip : interestedPoints)
    {
       circle(T,cv::Point(ip[1],ip[0]),1,Scalar(255),CV_FILLED,8,0);
       //break;
    }
    return T;
}

void Capture_Image(int mode, Theta_max discr)
{
    typedef image2d<vuchar1> Image;
    int T_theta = getThetaMax(discr);
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
        video_extruder_ctx ctx = video_extruder_init(domain);
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

            std::vector<float> t_accumulator(rhomax*T_theta);
            std::vector<vshort4> t_accumulator_point(rhomax*T_theta, vshort4(0,0,ncols,nrows));
            for(int i = 0 ; i < rhomax*T_theta ; i++)
            {
                t_accumulator[i]=0;
            }
            //Hough_Accumulator_Video(img,hough_parallel,T_theta,t_accumulator,rhomax);
            //

           // writer2.write(Hough_Accumulator_Video_Map_and_Clusters(img,hough_parallel,T_theta,t_accumulator,interestedPoints,rhomax));
            string filename = "result";
            if(ranked<10)
            filename = filename +"0"+ std::to_string(ranked);
            else
               filename = filename +std::to_string(ranked);
            //cv::imwrite("images/"+filename+".bmp", Hough_Accumulator_Video_Map_and_Clusters(img,hough_parallel,T_theta,t_accumulator,interestedPoints,rhomax));
            cv::imwrite("images/"+filename+".bmp", Hough_Accumulator_Video_Clusters(img,hough_parallel,T_theta,t_accumulator,interestedPoints,t_accumulator_point,rhomax));
            ctrl = cap.read(frame);
            ranked++;
        }
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
