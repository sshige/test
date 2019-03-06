#include <iostream>
#include <ctime>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/AutoDiff>
#include "least_square_problem.h"


using namespace Eigen;

template<typename _Scalar>
struct Functor
{
  typedef _Scalar Scalar;
  enum {
    InputsAtCompileTime = Dynamic,
    ValuesAtCompileTime = Dynamic
  };
  typedef Matrix<Scalar,InputsAtCompileTime,1> InputType;
  typedef Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
  typedef Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

  const int inputs_, values_;

  Functor() : inputs_(InputsAtCompileTime), values_(ValuesAtCompileTime) {}
  Functor(int inputs, int values) : inputs_(inputs), values_(values) {}

  int inputs() const { return inputs_; }
  int values() const { return values_; }
};

// Specialized functor
struct LeastSquareProblemFunctor : Functor<double>
{
  typedef Eigen::AutoDiffScalar<Eigen::Matrix<Scalar,Eigen::Dynamic,1> > ADS;
  typedef Eigen::Matrix<ADS, Eigen::Dynamic, 1> VectorXad;

  LeastSquareProblemFunctor(const LeastSquareProblem &lsp):
    Functor<double>(lsp.designVariableDim(), lsp.datasetNum()),
    lsp_(lsp)
  {};

  // Compute the function value into fvec for the current solution x
  int operator()(const VectorXd &var, VectorXd &fvec)
  {
    lsp_.setDesignVariable(var);
    for (unsigned int i = 0; i < lsp_.datasetNum(); i++) {
      fvec(i) = lsp_.y_(i) - lsp_.func_ptr_->operator()(lsp_.x_(i));
    }
    return 0;
  }

  // Compute the jacobian into fjac for the current solution x
  int df(const VectorXd &var, MatrixXd &fjac)
  {
    VectorXad coeff(lsp_.designVariableDim());
    for(unsigned int i = 0; i < coeff.rows(); i++) {
      coeff(i) = ADS(var(i), coeff.rows(), i);
    }

    lsp_.setDesignVariable(var);
    VectorXad fvalue(1);
    for (unsigned int i = 0; i < lsp_.datasetNum(); i++) {
      fvalue = lsp_.y_(i) - lsp_.func_ptr_->operator()(lsp_.x_(i));
      fjac.row(i) = fvalue(0).derivatives();
    }

    return 0;
  }

  LeastSquareProblem lsp_;
};

int main()
{
  using namespace std;

  // 1. setup problem
  LeastSquareProblemPtr lsp_ptr;
  VectorXd true_coeff;
  initialize_sample_polynomial_lsp(lsp_ptr, true_coeff);

  // 2. solve problem
  clock_t begin = clock();

  LeastSquareProblemFunctor func(*lsp_ptr);
  LevenbergMarquardt<LeastSquareProblemFunctor> lm(func);

  lm.parameters.ftol *= 1e-2;
  lm.parameters.xtol *= 1e-2;
  lm.parameters.maxfev = 2000;

  Eigen::VectorXd x = VectorXd::Zero(lsp_ptr->designVariableDim());
  int status = lm.minimize(x);

  clock_t end = clock();
  double optimization_time = double(end - begin) / CLOCKS_PER_SEC;

  // 3. print result
  cout << "Optimization status:" << status << endl;
  cout << "  - status: " << status << endl;
  cout << "  - number of function evaluation: " << lm.nfev << endl;
  cout << "  - number of jacobian evaluation: " << lm.njev << endl;

  cout << "optimization computation time: " << optimization_time << " [sec]" << endl;
  cout << "solution error: " << (true_coeff - x).norm() << endl;
  // cout << "coeff error: " << true_coeff - x << endl;
  // cout << "true coeff: " << endl << true_coeff << endl;
  // cout << "optimal coeff: " << endl << x << endl;

  return 0;
}
