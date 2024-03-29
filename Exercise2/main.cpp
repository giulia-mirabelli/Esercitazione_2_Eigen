#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>
using namespace std;
using namespace Eigen;

Vector2d QR (Matrix2d A,Vector2d b)
{
    Vector2d x;
    HouseholderQR<Matrix2d> qr(A);
    x=qr.solve(b);
    return x;
}

Vector2d PALU (Matrix2d A,Vector2d b)
{
    Vector2d x;
    PartialPivLU<Matrix2d> lu(A);
    x=lu.solve(b);
    return x;
}

double relative_err (const Vector2d& x, const Vector2d& y)
{
    double err = (x-y).norm();

    return err/y.norm();


}


int main()
{
    Matrix2d A1, A2, A3;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01, -9.992887623566787e-01;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01, -8.320502947645361e-01;

    Vector2d b1, b2, b3, x_corretto, x1qr, x2qr, x3qr, x1lu, x2lu, x3lu;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    x_corretto << -1.0e+0, -1.0e+00;

    //QR

    x1qr<<QR(A1,b1);
    x2qr<<QR(A2,b2);
    x3qr<<QR(A3,b3);

    double err_rel_x1_qr, err_rel_x2_qr, err_rel_x3_qr;

    err_rel_x1_qr = relative_err(x1qr,x_corretto);
    err_rel_x2_qr = relative_err(x2qr,x_corretto);
    err_rel_x3_qr = relative_err(x3qr,x_corretto);


    cout<<"Risoluzione tramite decomposizione QR: "
         <<"\nx1 = "<<x1qr.transpose()<<"  err.rel = "<<err_rel_x1_qr
         <<"\nx2 = "<<x2qr.transpose()<<"  err.rel = "<<err_rel_x2_qr
         <<"\nx3 = "<<x3qr.transpose()<<"  err.rel = "<<err_rel_x3_qr
         <<endl;

    //PALU

    x1lu<<PALU(A1,b1);
    x2lu<<PALU(A2,b2);
    x3lu<<PALU(A3,b3);

    double err_rel_x1_lu, err_rel_x2_lu, err_rel_x3_lu;

    err_rel_x1_lu = relative_err(x1lu,x_corretto);
    err_rel_x2_lu = relative_err(x2lu,x_corretto);
    err_rel_x3_lu = relative_err(x3lu,x_corretto);

    cout<<"\nRisoluzione tramite decomposizione PA=LU: "
         <<"\nx1 = "<<x1lu.transpose()<<"  err.rel = "<<err_rel_x1_lu
         <<"\nx2 = "<<x2lu.transpose()<<"  err.rel = "<<err_rel_x2_lu
         <<"\nx3 = "<<x3lu.transpose()<<"  err.rel = "<<err_rel_x3_lu
         <<endl;

    return 0;
}
