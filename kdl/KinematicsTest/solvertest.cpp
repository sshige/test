#include <chain.hpp>
#include <chainfksolverpos_recursive.hpp>
#include <chainfksolvervel_recursive.hpp>
#include <chainiksolvervel_pinv.hpp>
#include <chainiksolvervel_pinv_givens.hpp>
#include <chainiksolvervel_pinv_nso.hpp>
#include <chainiksolvervel_wdls.hpp>
#include <chainiksolverpos_nr.hpp>
#include <chainiksolverpos_lma.hpp>
#include <chainiksolverpos_nr_jl.hpp>
#include <chainjnttojacsolver.hpp>
#include <chainjnttojacdotsolver.hpp>
#include <chainidsolver_vereshchagin.hpp>
#include <chainidsolver_recursive_newton_euler.hpp>
#include <chaindynparam.hpp>

#include <cmath>
#include <frames_io.hpp>
#include <framevel_io.hpp>
#include <kinfam_io.hpp>
#include <time.h>


using namespace KDL;

int main()
{
  Chain chain1;

  chain1.addSegment(Segment("Segment 1", Joint("Joint 1", Joint::RotZ),
                            Frame(Vector(1.0,0.0,0.0))));
  chain1.addSegment(Segment("Segment 2", Joint("Joint 2", Joint::RotZ),
                            Frame(Vector(1.0,0.0,0.0))));

  ChainFkSolverPos_recursive fksolver1(chain1);
  ChainIkSolverVel_pinv iksolver1v(chain1);
  ChainIkSolverPos_NR iksolver1(chain1, fksolver1, iksolver1v);
  // ChainIkSolverVel_pinv_givens iksolverv_pinv_givens1(chain1);
  // ChainIkSolverPos_NR iksolver1_givens(chain1, fksolver1, iksolverv_pinv_givens1, 1000);

  JntArray q1(chain1.getNrOfJoints());
  Frame F1;
  q1(0) = M_PI_2;
  q1(1) = M_PI_2;
  fksolver1.JntToCart(q1, F1);
  std::cout << "=== FK test ===" << std::endl;
  std::cout << "q1: " << q1 << std::endl;
  std::cout << "F1: " << std::endl << F1 << std::endl;

  JntArray q_init(chain1.getNrOfJoints());
  JntArray q_solved(chain1.getNrOfJoints());
  Frame F_solved;
  random(q_init(0));
  random(q_init(1));
  iksolver1.CartToJnt(q_init, F1, q_solved);
  std::cout << "=== IK test ===" << std::endl;
  std::cout << "q_init: " << q_init << std::endl;
  std::cout << "q_solved: " << q_solved << std::endl;
  std::cout << "F1: " << std::endl << F1 << std::endl;
  fksolver1.JntToCart(q_solved, F_solved);
  std::cout << "F_solved: " << std::endl << F_solved << std::endl;

  return 0;
}
