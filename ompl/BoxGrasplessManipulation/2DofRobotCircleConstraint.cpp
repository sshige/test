/* Author: Masaki Murooka */

// ompl
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/prm/PRM.h>
#include <ompl/geometric/SimpleSetup.h>

// kdl
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
#include <frames_io.hpp>
#include <framevel_io.hpp>
#include <kinfam_io.hpp>

#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>


#define deg_to_rad(deg) (((deg)/360.0)*2*M_PI)
#define rad_to_deg(rad) (((rad)/2.0/M_PI)*360)


namespace ob = ompl::base;
namespace og = ompl::geometric;

bool checkIKonCircleTrajectory(double* theta)
{
  KDL::Chain chain1;

  chain1.addSegment(KDL::Segment("Segment 1", KDL::Joint("Joint 1", KDL::Joint::RotZ),
                                 KDL::Frame(KDL::Vector(1.0,0.0,0.0))));
  chain1.addSegment(KDL::Segment("Segment 2", KDL::Joint("Joint 2", KDL::Joint::RotZ),
                                 KDL::Frame(KDL::Vector(1.0,0.0,0.0))));
  chain1.addSegment(KDL::Segment("Segment 3", KDL::Joint("Joint 3", KDL::Joint::RotZ),
                                 KDL::Frame(KDL::Vector(0.0,0.0,0.0))));

  KDL::ChainFkSolverPos_recursive fksolver1(chain1);
  KDL::ChainIkSolverVel_pinv iksolver1v(chain1);
  KDL::ChainIkSolverPos_NR iksolver1(chain1, fksolver1, iksolver1v);

  KDL::JntArray q_init(chain1.getNrOfJoints());
  KDL::Frame F_init;
  for (int i=0; i<3; i++) {
    q_init(i) = theta[i];
  }
  fksolver1.JntToCart(q_init, F_init);

  double norm = sqrt((F_init.p[0] - 0.5) * (F_init.p[0] - 0.5) + F_init.p[1] * F_init.p[1]);

  std::cout << "=== FK test ===" << std::endl;
  std::cout << "q_init: " << q_init << std::endl;
  std::cout << "F_init: " << std::endl << F_init << std::endl;
  std::cout << "norm: " << norm << std::endl;

  return (fabs(norm - 0.5) < 0.1);
}

bool solveIKonCircleTrajectory(double* theta)
{
  KDL::Chain chain1;

  chain1.addSegment(KDL::Segment("Segment 1", KDL::Joint("Joint 1", KDL::Joint::RotZ),
                                 KDL::Frame(KDL::Vector(1.0,0.0,0.0))));
  chain1.addSegment(KDL::Segment("Segment 2", KDL::Joint("Joint 2", KDL::Joint::RotZ),
                                 KDL::Frame(KDL::Vector(1.0,0.0,0.0))));
  chain1.addSegment(KDL::Segment("Segment 3", KDL::Joint("Joint 3", KDL::Joint::RotZ),
                                 KDL::Frame(KDL::Vector(0.0,0.0,0.0))));

  KDL::ChainFkSolverPos_recursive fksolver1(chain1);
  KDL::ChainIkSolverVel_pinv iksolver1v(chain1);
  KDL::ChainIkSolverPos_NR iksolver1(chain1, fksolver1, iksolver1v);

  KDL::JntArray q_init(chain1.getNrOfJoints());
  KDL::Frame F_init;
  for (int i=0; i<3; i++) {
    q_init(i) = theta[i];
  }
  fksolver1.JntToCart(q_init, F_init);
  // std::cout << "=== FK test ===" << std::endl;
  // std::cout << "q_init: " << q_init << std::endl;
  // std::cout << "F_init: " << std::endl << F_init << std::endl;

  double norm = sqrt((F_init.p[0] - 0.5) * (F_init.p[0] - 0.5) + F_init.p[1] * F_init.p[1]);
  F_init.p[0] = (F_init.p[0] - 0.5) / (2.0 * norm) + 0.5;
  F_init.p[1] = F_init.p[1] / (2.0 * norm);

  KDL::JntArray q_solved(chain1.getNrOfJoints());
  KDL::Frame F_solved;
  iksolver1.CartToJnt(q_init, F_init, q_solved);
  // std::cout << "=== IK test ===" << std::endl;
  // std::cout << "q_init: " << q_init << std::endl;
  // std::cout << "q_solved: " << q_solved << std::endl;
  // std::cout << "F1: " << std::endl << F1 << std::endl;
  // fksolver1.JntToCart(q_solved, F_solved);
  // std::cout << "F_solved: " << std::endl << F_solved << std::endl;
  for (int i=0; i<3; i++) {
    theta[i] = q_solved(i);
  }

  return checkIKonCircleTrajectory(theta);
}

class MyValidStateSampler : public ob::ValidStateSampler
{
public:
  MyValidStateSampler(const ob::SpaceInformation *si) : ValidStateSampler(si)
  {
    name_ = "my sampler";
  }

  bool sample(ob::State *state) override
  {
    double* val = static_cast<ob::RealVectorStateSpace::StateType*>(state)->values;
    return solveIKonCircleTrajectory(val);
  }

  // We don't need this in the example below.
  bool sampleNear(ob::State* /*state*/, const ob::State* /*near*/, const double /*distance*/) override
  {
    throw ompl::Exception("MyValidStateSampler::sampleNear", "not implemented");
    return true;
  }
protected:
  ompl::RNG rng_;
};

ob::ValidStateSamplerPtr allocMyValidStateSampler(const ob::SpaceInformation *si)
{
  return std::make_shared<MyValidStateSampler>(si);
}

bool isStateValid(const ob::State *state)
{
  const double* val = static_cast<const ob::RealVectorStateSpace::StateType*>(state)->values;

  double val_tmp[3];
  for(int i = 0; i < 3; i++) {
    val_tmp[i] = val[i];
  }
  return checkIKonCircleTrajectory(val_tmp);
}

void plan()
{
    // construct the state space we are planning in
    auto space(std::make_shared<ob::RealVectorStateSpace>(3));

    // set the bounds
    ob::RealVectorBounds bounds(3);
    bounds.setLow(- 2 * M_PI);
    bounds.setHigh(2 * M_PI);
    space->setBounds(bounds);

    // define a simple setup class
    og::SimpleSetup ss(space);

    // set state validity checking for this space
    ss.setStateValidityChecker(isStateValid);

    // create a random start state
    ob::ScopedState<ob::RealVectorStateSpace> start(space);
    start[0] = deg_to_rad(60.0);
    start[1] = deg_to_rad(-120.0);
    start[2] = 0.0;

    // create a random goal state
    ob::ScopedState<ob::RealVectorStateSpace> goal(space);
    goal[0] = deg_to_rad(90);
    goal[1] = deg_to_rad(-180);
    goal[2] = 0.0;

    // set the start and goal states
    ss.setStartAndGoalStates(start, goal);

    // set sampler
    ss.getSpaceInformation()->setValidStateSamplerAllocator(allocMyValidStateSampler);

    // set planner
    auto planner(std::make_shared<og::PRM>(ss.getSpaceInformation()));
    ss.setPlanner(planner);

    // this call is optional, but we put it in to get more output information
    //ss.setup();
    //ss.print();

    // attempt to solve the problem within one second of planning time
    ob::PlannerStatus solved = ss.solve(1.0);

    if (solved)
    {
        std::cout << "Found solution:" << std::endl;
        // print the path to screen
        ss.simplifySolution();

        og::PathGeometric& path(ss.getSolutionPath());
        path.interpolate();
        // path.print(std::cout);
        path.printAsMatrix(std::cout);

        std::string filename = "/tmp/ompl_path.txt";
        std::ofstream file_stream;
        file_stream.open(filename, std::ios::out);
        ss.getSolutionPath().printAsMatrix(file_stream);
        std::cout << "Saved the path to " << filename << std::endl;
    }
    else
        std::cout << "No solution found" << std::endl;
}

int main(int /*argc*/, char ** /*argv*/)
{
    std::cout << "OMPL version: " << OMPL_VERSION << std::endl;

    plan();

    return 0;
}
