#include <iostream>
#include <sys/time.h>
#include <boost/program_options.hpp>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include "least_square_problem.h"


using namespace Eigen;
using namespace opt_benchmark;

/**
 * \brief Base class of functor for Levenberg-Marquardt method.
 */
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

/**
 * \brief Class of functor for Levenberg-Marquardt method with least square problem.
 */
struct LeastSquareProblemFunctor : Functor<double>
{
  LeastSquareProblemFunctor(const LeastSquareProblemPtr &lsp_ptr):
    Functor<double>(lsp_ptr->designVariableDim(), lsp_ptr->datasetNum()),
    lsp_ptr_(lsp_ptr)
  {};

  // Compute the function value into fvec for the current solution var
  int operator()(const VectorXd &var, VectorXd &fvec) const
  {
    lsp_ptr_->eval(var, fvec);
    return 0;
  }

  // Compute the jacobian into fjac for the current solution var
  int df(const VectorXd &var, MatrixXd &fjac) const
  {
    lsp_ptr_->evalJacobi(var, fjac);
    return 0;
  }

  LeastSquareProblemPtr lsp_ptr_;
};

/**
 * \brief Solve.
 *
 * template type _Functor is LeastSquareProblemFunctor or NumericalDiff<LeastSquareProblemFunctor>
 */
template <typename _Functor>
int solve(_Functor &func, const VectorXd &true_coeff)
{
  using namespace std;

  // solve problem
  struct timeval start, end;
  gettimeofday(&start, NULL);

  LevenbergMarquardt<_Functor> lm(func);
  lm.parameters.ftol *= 1e-2;
  lm.parameters.xtol *= 1e-2;
  lm.parameters.maxfev = 2000;

  VectorXd x = VectorXd::Ones(func.lsp_ptr_->designVariableDim());
  int status = lm.minimize(x);

  gettimeofday(&end, NULL);
  double optimization_time = (end.tv_sec  - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

  // print result
  cout << "Optimization status:" << endl;
  cout << "  - status: " << status << endl;
  cout << "  - number of function evaluation: " << lm.nfev << endl;
  cout << "  - number of jacobian evaluation: " << lm.njev << endl;

  cout << "optimization computation time: " << optimization_time << " [sec]" << endl;
  cout << "solution error: " << (true_coeff - x).norm() << endl;
  // cout << "coeff error: " << endl << true_coeff - x << endl;
  // cout << "true coeff: " << endl << true_coeff << endl;
  // cout << "optimal coeff: " << endl << x << endl;

  return 0;
}

int main(int argc, char** argv)
{
  using namespace std;
  namespace po = boost::program_options;

  po::options_description desc("Allowed options");
  desc.add_options()("mode", po::value<std::string>(), "derivative mode (default, ad, nd)");
  desc.add_options()("func", po::value<std::string>(), "function type (poly, sin)");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  std::string mode = "default";
  std::string func_type = "poly";
  if(vm.count("mode")) {
    mode = vm["mode"].as<std::string>();
  }
  if(vm.count("func")) {
    func_type = vm["func"].as<std::string>();
  }
  cout << "derivative mode: " << mode << endl;
  cout << "function type: " << func_type << endl;
  cout << "number of threads: " << Eigen::nbThreads() << endl;

  // setup problem
  LeastSquareProblemPtr lsp_ptr;
  VectorXd true_coeff;
  if (func_type == "sin") {
    initialize_sample_sine_lsp(lsp_ptr, true_coeff, mode);
  } else {
    initialize_sample_polynomial_lsp(lsp_ptr, true_coeff, mode);
  }
  LeastSquareProblemFunctor func(lsp_ptr);
  if (mode == "nd") {
    NumericalDiff<LeastSquareProblemFunctor> func_nd(func);
    return solve<NumericalDiff<LeastSquareProblemFunctor> >(func_nd, true_coeff);
  } else {
    return solve<LeastSquareProblemFunctor >(func, true_coeff);
  }
}
