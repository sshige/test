#ifndef __LEAST_SQUARE_PROBLEM_H__
#define __LEAST_SQUARE_PROBLEM_H__

#include <memory>
#include <Eigen/Dense>

namespace opt_benchmark
{
  class ScalarFunc
  {
  public:
    ScalarFunc(const Eigen::VectorXd &coeff):
      coeff_(coeff)
    {}

    virtual double operator()(const double x) = 0;
    virtual Eigen::VectorXd derivative(const double x) = 0;

    Eigen::VectorXd coeff_;
  };
  typedef std::shared_ptr<ScalarFunc> ScalarFuncPtr;

  class PolynomialFunc: public ScalarFunc
  {
  public:
    PolynomialFunc(const unsigned int order, const Eigen::VectorXd &coeff):
      ScalarFunc(coeff),
      order_(order)
    {
      assert(coeff_.size() == order_+1);
    }

    double operator()(const double x)
    {
      Eigen::VectorXd poly_x(coeff_.size());
      for (unsigned int i = 0; i < coeff_.size(); i++) {
        poly_x(i) = pow(x, (double)(order_ - i));
      }
      return coeff_.dot(poly_x);
    }

    Eigen::VectorXd derivative(const double x)
    {
      Eigen::VectorXd poly_x(coeff_.size());
      for (unsigned int i = 0; i < coeff_.size(); i++) {
        poly_x(i) = pow(x, (double)(order_ - i));
      }

      return poly_x;
    }

    unsigned int order_;
  };
  typedef std::shared_ptr<PolynomialFunc> PolynomialFuncPtr;

  class LeastSquareProblem
  {
  public:
    LeastSquareProblem(const ScalarFuncPtr &func_ptr, unsigned int data_num=100):
      func_ptr_(func_ptr)
    {
      x_ = Eigen::VectorXd::Random(data_num);
      y_ = Eigen::VectorXd(data_num);

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

    ScalarFuncPtr func_ptr_;
    Eigen::VectorXd x_;
    Eigen::VectorXd y_;
  };
  typedef std::shared_ptr<LeastSquareProblem> LeastSquareProblemPtr;


  void initialize_sample_polynomial_lsp(LeastSquareProblemPtr &lsp_ptr,
                                        Eigen::VectorXd &true_coeff,
                                        unsigned int order = 20,
                                        unsigned int data_num = 1000)
  {
    true_coeff = Eigen::VectorXd::Random(order+1);
    PolynomialFuncPtr poly_ptr = std::make_shared<PolynomialFunc>(order, true_coeff);

    lsp_ptr.reset(new LeastSquareProblem(poly_ptr, data_num));
    lsp_ptr->func_ptr_->coeff_.setZero(); // unset true coeff

    lsp_ptr->printBasicInfo();
    // lsp_ptr->printXY();
  }
}

#endif
