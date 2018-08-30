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

#define deg2rad(deg) (((deg)/360)*2*M_PI)
#define rad2deg(rad) (((rad)/2/M_PI)*360)


using namespace KDL;

int main()
{
  // generate tree
  Tree tree("root");
  tree.addSegment(Segment("Segment_0", Joint("Joint_0", Joint::None),
                          Frame(Vector(1.0, 0.0, 0.0))),
                  "root");
  tree.addSegment(Segment("Segment_1_1", Joint("Joint_1_1", Joint::RotZ),
                          Frame(Vector(1.0, 0.5, 0.0))),
                  "Segment_0");
  tree.addSegment(Segment("Segment_1_2", Joint("Joint_2_1", Joint::RotZ),
                          Frame(Vector(1.0, 0.0, 0.0))),
                  "Segment_1_1");
  tree.addSegment(Segment("Segment_2_1", Joint("Joint_1_1", Joint::RotZ),
                          Frame(Vector(1.0, -0.5, 0.0))),
                  "Segment_0");
  tree.addSegment(Segment("Segment_2_2", Joint("Joint_2_1", Joint::RotZ),
                          Frame(Vector(1.0, 0.0, 0.0))),
                  "Segment_2_1");

  // check tree
  unsigned int num_joint = tree.getNrOfJoints();
  unsigned int num_segment = tree.getNrOfSegments();
  printf("# Tree\n");
  printf("## Basic Info\n");
  printf("num_joint: %d,  num_segment: %d\n", num_joint, num_segment);

  std::map<std::string,TreeElement> tree_segments;
  tree_segments = tree.getSegments();
  printf("## Segments\n");
  for (std::map<std::string,TreeElement>::iterator itr = tree_segments.begin(); itr != tree_segments.end(); ++itr) {
    printf("  %s\n", itr->first.c_str());
  }

  // genrate chain
  Chain chain;
  tree.getChain("Segment_1_2","Segment_2_2",chain);

  // check chain
  printf("# Chain\n");
  printf("## Basic Info\n");
  num_joint = chain.getNrOfJoints();
  num_segment = chain.getNrOfSegments();
  printf("num_joint: %d,  num_segment: %d\n", num_joint, num_segment);

  std::vector<Segment> chain_segments = chain.segments;
  printf("## Segments\n");
  for (std::vector<Segment>::iterator itr = chain_segments.begin(); itr != chain_segments.end(); ++itr) {
    printf("  %s\n", itr->getName().c_str());
  }

  // solve FK and IK
  ChainFkSolverPos_recursive fksolver(chain);
  ChainIkSolverVel_pinv iksolverv(chain);
  ChainIkSolverPos_NR iksolver(chain, fksolver, iksolverv);
  // ChainIkSolverVel_pinv_givens iksolverv_pinv_givens1(chain);
  // ChainIkSolverPos_NR iksolver_givens(chain, fksolver, iksolverv_pinv_givens1, 1000);

  //// FK
  JntArray q1(chain.getNrOfJoints());
  Frame F1;
  q1(0) = deg2rad(30.0);
  q1(1) = deg2rad(60.0);
  q1(2) = deg2rad(-30.0);
  q1(3) = deg2rad(-60.0);
  fksolver.JntToCart(q1, F1);
  std::cout << "=== FK test ===" << std::endl;
  std::cout << "q1: " << q1 << std::endl;
  std::cout << "F1.p: " << std::endl << F1.p << std::endl;
  std::cout << "F1.M: " << std::endl << F1.M << std::endl;

  //// IK
  JntArray q_init(chain.getNrOfJoints());
  JntArray q_solved(chain.getNrOfJoints());
  Frame F_solved;
  random(q_init(0));
  random(q_init(1));
  random(q_init(2));
  random(q_init(3));
  iksolver.CartToJnt(q_init, F1, q_solved);
  std::cout << "=== IK test ===" << std::endl;
  std::cout << "q_init: " << q_init << std::endl;
  std::cout << "q_solved: " << q_solved << std::endl;
  std::cout << "F1.p: " << std::endl << F1.p << std::endl;
  fksolver.JntToCart(q_solved, F_solved);
  std::cout << "F_solved.p: " << std::endl << F_solved.p << std::endl;
  std::cout << "F1.inv*F_solved: " << std::endl << F1.Inverse()*F_solved << std::endl;

  return 0;
}
