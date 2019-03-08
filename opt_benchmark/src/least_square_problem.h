#ifndef __LEAST_SQUARE_PROBLEM_H__
#define __LEAST_SQUARE_PROBLEM_H__

#include <memory>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>


namespace opt_benchmark
{
  typedef Eigen::AutoDiffScalar<Eigen::Matrix<double,Eigen::Dynamic,1> > ADS;
  typedef Eigen::Matrix<ADS, Eigen::Dynamic, 1> VectorXad;

  template<typename Scalar = double>
  class ScalarFuncWithCoeff
  {
  public:
    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> VectorXS;

    ScalarFuncWithCoeff(const VectorXS &coeff):
      coeff_(coeff)
    {}

    virtual Scalar operator()(const double x) = 0;
    virtual Eigen::VectorXd derivative_with_coeff(const double x) = 0;

    VectorXS coeff_;
  };
  template <typename Scalar>
  using ScalarFuncWithCoeffPtr = std::shared_ptr<ScalarFuncWithCoeff<Scalar>>;

  template<typename Scalar = double>
  class PolynomialFunc: public ScalarFuncWithCoeff<Scalar>
  {
  public:
    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> VectorXS;
    typedef ScalarFuncWithCoeff<Scalar> inherited;

    PolynomialFunc(const unsigned int order, const VectorXS &coeff):
      ScalarFuncWithCoeff<Scalar>(coeff),
      order_(order)
    {
      assert(inherited::coeff_.size() == order_+1);
    }

    Scalar operator()(const double x)
    {
      Scalar ret = 0.0;
      for (unsigned int i = 0; i < order_+1; i++) {
        ret += inherited::coeff_(i) * pow(x, (double)(order_ - i));
      }
      return ret;
    }

    Eigen::VectorXd derivative_with_coeff(const double x)
    {
      Eigen::VectorXd x_powed(order_+1);
      for (unsigned int i = 0; i < order_+1; i++) {
        x_powed(i) = pow(x, (double)(order_ - i));
      }
      return x_powed;
    }

    const unsigned int order_;
  };
  template <typename Scalar>
  using PolynomialFuncPtr = std::shared_ptr<PolynomialFunc<Scalar>>;

  class LeastSquareProblem
  {
  public:
    LeastSquareProblem(const ScalarFuncWithCoeffPtr<double> &func_ptr,
                       unsigned int data_num):
      func_ptr_(func_ptr)
    {
      x_ = Eigen::VectorXd::Random(data_num);
      y_ = Eigen::VectorXd(data_num);

#pragma omp parallel for
      for (unsigned int i = 0; i < data_num; i++) {
        y_(i) = func_ptr_->operator()(x_(i));
      }
    }

    unsigned int designVariableDim() const
    {
      return func_ptr_->coeff_.size();
    }

    unsigned int datasetNum() const
    {
      return x_.size();
    }

    Eigen::VectorXd designVariable()
    {
      return func_ptr_->coeff_;
    }

    void setDesignVariable(const Eigen::VectorXd var)
    {
      func_ptr_->coeff_ = var;
    }

    void eval(const Eigen::VectorXd &var, Eigen::VectorXd &fvec)
    {
      setDesignVariable(var);
#pragma omp parallel for
      for (unsigned int i = 0; i < datasetNum(); i++) {
        fvec(i) = y_(i) - func_ptr_->operator()(x_(i));
      }
      // The following is same:
      // fvec = y_ - x_.unaryExpr([&](double x) { return func_ptr_->operator()(x); });
    }

    virtual void evalJacobi(const Eigen::VectorXd &var, Eigen::MatrixXd &fjac)
    {
      setDesignVariable(var);
#pragma omp parallel for
      for (unsigned int i = 0; i < datasetNum(); i++) {
        fjac.row(i) = - func_ptr_->derivative_with_coeff(x_(i));
      }
    }

    void printBasicInfo()
    {
      std::cout << "LeastSquareProblem:" << std::endl;
      std::cout << "  - dim of design variable: " << designVariableDim() << std::endl;
      std::cout << "  - number of data: " << datasetNum() << std::endl;
    }

    void printXY()
    {
      std::cout << "x:" << std::endl << x_ << std::endl;
      std::cout << "y:" << std::endl << y_ << std::endl;
    }

    ScalarFuncWithCoeffPtr<double> func_ptr_;
    Eigen::VectorXd x_;
    Eigen::VectorXd y_;
  };
  typedef std::shared_ptr<LeastSquareProblem> LeastSquareProblemPtr;

  class LeastSquareProblemAD: public LeastSquareProblem
  {
  public:
    LeastSquareProblemAD(const ScalarFuncWithCoeffPtr<double> &func_ptr,
                         const ScalarFuncWithCoeffPtr<ADS> &func_ad_ptr,
                         unsigned int data_num):
      LeastSquareProblem(func_ptr, data_num),
      func_ad_ptr_(func_ad_ptr)
    {
    }

    virtual void evalJacobi(const Eigen::VectorXd &var, Eigen::MatrixXd &fjac)
    {
      // initialize coeff as auto diff variable
      for(unsigned int i = 0; i < designVariableDim(); i++) {
        func_ad_ptr_->coeff_(i) = ADS(func_ptr_->coeff_(i), designVariableDim(), i);
      }

      VectorXad fvec_ad(datasetNum());
      for (unsigned int i = 0; i < datasetNum(); i++) {
        fvec_ad(i) = y_(i) - func_ad_ptr_->operator()(x_(i));
        fjac.row(i) = fvec_ad(i).derivatives();
      }
    }

    ScalarFuncWithCoeffPtr<ADS> func_ad_ptr_;
  };
  typedef std::shared_ptr<LeastSquareProblemAD> LeastSquareProblemADPtr;


  void initialize_sample_polynomial_lsp(LeastSquareProblemPtr &lsp_ptr,
                                        Eigen::VectorXd &true_coeff,
                                        std::string mode = "default",
                                        unsigned int order = 20,
                                        unsigned int data_num = 1000
                                        )
  {
    true_coeff = Eigen::VectorXd::Random(order+1);
    PolynomialFuncPtr<double> poly_ptr = std::make_shared<PolynomialFunc<double>>(order, true_coeff);
    if (mode == "ad") {
      PolynomialFuncPtr<ADS> poly_ad_ptr = std::make_shared<PolynomialFunc<ADS>>(order, VectorXad(true_coeff.size()));
      lsp_ptr.reset(new LeastSquareProblemAD(poly_ptr, poly_ad_ptr, data_num));
    } else {
      lsp_ptr.reset(new LeastSquareProblem(poly_ptr, data_num));
    }
    lsp_ptr->designVariable().setZero(); // unset true coeff
    lsp_ptr->printBasicInfo();
    // lsp_ptr->printXY();
  }
}

#endif
