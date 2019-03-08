#ifndef __LEAST_SQUARE_PROBLEM_H__
#define __LEAST_SQUARE_PROBLEM_H__

#include <memory>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>


namespace opt_benchmark
{
  typedef Eigen::AutoDiffScalar<Eigen::Matrix<double,Eigen::Dynamic,1> > ADS;
  typedef Eigen::Matrix<ADS, Eigen::Dynamic, 1> VectorXad;

  /**
   * \brief Base class for scalar-input, scalar-output function, which has parameter as coefficient.
   *
   * y = f(x)
   */
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

  /**
   * \brief Class for polynomial function.
   *
   * y = c_0 x^n + c_1 x^{n-1} + ... + c_{n-1} x + c_n
   * coefficient is (c_0, c_1, ..., c_n)
   */
  template<typename Scalar = double>
  class PolynomialFunc: public ScalarFuncWithCoeff<Scalar>
  {
  public:
    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> VectorXS;
    typedef ScalarFuncWithCoeff<Scalar> inherited;

    PolynomialFunc(const unsigned int order, const VectorXS &coeff):
      ScalarFuncWithCoeff<Scalar>(coeff),
      order_(order),
      order_vec_(Eigen::VectorXd::LinSpaced(order+1, order, 0))
    {
      assert(inherited::coeff_.size() == order_+1);
    }

    Scalar operator()(const double x)
    {
#if EIGEN_VERSION_AT_LEAST(3,3,9)
      return inherited::coeff_.dot(pow(x, order_vec_.array()).matrix());
#else
#warning "Eigen is not latest, so we cannot use eigen multiplication between matrix of different scalar type."
      Scalar ret = 0;
      Eigen::VectorXd x_powed = pow(x, order_vec_.array());
      for (unsigned int i = 0; i < order_+1; i++) {
        ret += inherited::coeff_(i) * x_powed(i);
      }
      return ret;
#endif
    }

    Eigen::VectorXd derivative_with_coeff(const double x)
    {
      /* derivative with coefficient (c_0, c_1, ..., c_n) is (x^n, x^{n-1}, ..., x, 1) */
      return pow(x, order_vec_.array());
    }

    const unsigned int order_;
    const Eigen::VectorXd order_vec_;
  };
  template <typename Scalar>
  using PolynomialFuncPtr = std::shared_ptr<PolynomialFunc<Scalar>>;

  /**
   * \brief Class for sine function.
   *
   * y = \sum_{i=0}^{base_num-1} c_{3i} sin(c_{3i+1} x + c_{3i+2})
   * coefficient is (c_0, c_1, ..., c_{3 base_num})
   */
  template<typename Scalar = double>
  class SineFunc: public ScalarFuncWithCoeff<Scalar>
  {
  public:
    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> VectorXS;
    typedef ScalarFuncWithCoeff<Scalar> inherited;

    SineFunc(const unsigned int base_num, const VectorXS &coeff):
      ScalarFuncWithCoeff<Scalar>(coeff),
      base_num_(base_num)
    {
      assert(inherited::coeff_.size() == 3*base_num_);
    }

    Scalar operator()(const double x)
    {
      Scalar ret = 0;
      for (unsigned int i = 0; i < base_num_; i++) {
        ret += inherited::coeff_(3*i) * sin(inherited::coeff_(3*i+1) * x + inherited::coeff_(3*i+2));
      }
      return ret;
    }

    Eigen::VectorXd derivative_with_coeff(const double x)
    {
      Eigen::VectorXd der(3*base_num_);
#pragma omp parallel for
      for (unsigned int i = 0; i < base_num_; i++) {
        /* derivative with coefficient (c_{3i}, c_{3i+1}, c_{3i+2}) is
           ( sin(c_{3i+1} x + c_{3i+2}),
           c_{3i} x cos(c_{3i+1} x + c_{3i+2}),
           c_{3i} cos(c_{3i+1} x + c_{3i+2}) ) */
        der(3*i) = ((ADS)sin(inherited::coeff_(3*i+1) * x + inherited::coeff_(3*i+2))).value();
        der(3*i+1) = ((ADS)(inherited::coeff_(3*i) * x * cos(inherited::coeff_(3*i+1) * x + inherited::coeff_(3*i+2)))).value();
        der(3*i+2) = ((ADS)(inherited::coeff_(3*i) * cos(inherited::coeff_(3*i+1) * x + inherited::coeff_(3*i+2)))).value();
      }
      return der;
    }

    const unsigned int base_num_;
  };
  template <typename Scalar>
  using SineFuncPtr = std::shared_ptr<SineFunc<Scalar>>;

  /**
   * \brief Class for least square problem.
   */
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

  /**
   * \brief Class for least square problem with automatic differentiation.
   */
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


  /**
   * \brief Initialize the least square problem with polynomial function.
   */
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

  /**
   * \brief Initialize the least square problem with sine function.
   */
  void initialize_sample_sine_lsp(LeastSquareProblemPtr &lsp_ptr,
                                  Eigen::VectorXd &true_coeff,
                                  std::string mode = "default",
                                  unsigned int base_num = 2,
                                  unsigned int data_num = 1000
                                  )
  {
    true_coeff = Eigen::VectorXd::Random(3*base_num).array().abs();
    SineFuncPtr<double> poly_ptr = std::make_shared<SineFunc<double>>(base_num, true_coeff);
    if (mode == "ad") {
      SineFuncPtr<ADS> poly_ad_ptr = std::make_shared<SineFunc<ADS>>(base_num, VectorXad(true_coeff.size()));
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
