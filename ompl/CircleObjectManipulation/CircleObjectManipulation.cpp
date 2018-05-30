/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2010, Rice University
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Rice University nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Author: Ioan Sucan */

#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/spaces/DiscreteStateSpace.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/SimpleSetup.h>

#include <ompl/config.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>

namespace ob = ompl::base;
namespace og = ompl::geometric;

double* calcGripperCenter(unsigned int idx, double* obj_center, double obj_theta, double obj_radius, double gripper_radius)
{
  double gripper_angle = M_PI/2*(idx-1);
  double *gripper_center = new double[2];

  gripper_center[0] = (obj_radius + gripper_radius) * cos(obj_theta+gripper_angle) + obj_center[0];
  gripper_center[1] = (obj_radius + gripper_radius) * sin(obj_theta+gripper_angle) + obj_center[1];

  return gripper_center;
}

bool checkCollisionWithCircleObstacle(double* target_center, double target_radius, double* obst_center, double obst_radius, double padding=1.05)
{
  return
    (target_center[0] - obst_center[0]) * (target_center[0] - obst_center[0]) + (target_center[1] - obst_center[1]) * (target_center[1] - obst_center[1])
    < ((target_radius + obst_radius) * padding) * ((target_radius + obst_radius) * padding);
}

bool isStateValid(const ob::State *state)
{
    // get each element state value
    const auto *object_pose_state = state->as<ob::CompoundState>()->as<ob::SE2StateSpace::StateType>(0);
    const auto *contact1_state = state->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(1);
    const auto *contact2_state = state->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(2);
    const auto *contact3_state = state->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(3);
    const auto *contact4_state = state->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(4);

    double obj_x = object_pose_state->getX();
    double obj_y = object_pose_state->getY();
    double obj_theta = object_pose_state->getYaw();
    int contact1 = contact1_state->value;
    int contact2 = contact2_state->value;
    int contact3 = contact3_state->value;
    int contact4 = contact4_state->value;
    // std::cout << obj_x << " " << obj_y << " " << obj_theta << " " << contact1 << " " << contact2 << " " << contact3 << " " << contact4 << std::endl;

    // geometric settings
    double obj_center[2] = { obj_x, obj_y };
    double obj_radius = 0.2;
    double obst1_center[2] = { 0.0, 1.0 };
    double obst1_radius = 0.75;
    double obst2_center[2] = { 0.0, -1.0 };
    double obst2_radius = 0.75;
    double gripper_radius = 0.1;

    // check collision between object and environment
    bool is_obj_collision;
    is_obj_collision =
      checkCollisionWithCircleObstacle(obj_center, obj_radius, obst1_center, obst1_radius)
      or checkCollisionWithCircleObstacle(obj_center, obj_radius, obst2_center, obst2_radius);

    // check collision between gripper and environment
    bool is_gripper_collision;
    {
      double *gripper1_center, *gripper2_center, *gripper3_center, *gripper4_center;
      gripper1_center = calcGripperCenter(1, obj_center, obj_theta, obj_radius, gripper_radius);
      gripper2_center = calcGripperCenter(2, obj_center, obj_theta, obj_radius, gripper_radius);
      gripper3_center = calcGripperCenter(3, obj_center, obj_theta, obj_radius, gripper_radius);
      gripper4_center = calcGripperCenter(4, obj_center, obj_theta, obj_radius, gripper_radius);

      bool is_gripper1_collision =
        contact1 and
        (checkCollisionWithCircleObstacle(gripper1_center, gripper_radius, obst1_center, obst1_radius)
         or checkCollisionWithCircleObstacle(gripper1_center, gripper_radius, obst2_center, obst2_radius));
      bool is_gripper2_collision =
        contact2 and
        (checkCollisionWithCircleObstacle(gripper2_center, gripper_radius, obst1_center, obst1_radius)
         or checkCollisionWithCircleObstacle(gripper2_center, gripper_radius, obst2_center, obst2_radius));
      bool is_gripper3_collision =
        contact3 and
        (checkCollisionWithCircleObstacle(gripper3_center, gripper_radius, obst1_center, obst1_radius)
         or checkCollisionWithCircleObstacle(gripper3_center, gripper_radius, obst2_center, obst2_radius));
      bool is_gripper4_collision =
        contact4 and
        (checkCollisionWithCircleObstacle(gripper4_center, gripper_radius, obst1_center, obst1_radius)
         or checkCollisionWithCircleObstacle(gripper4_center, gripper_radius, obst2_center, obst2_radius));

      is_gripper_collision = is_gripper1_collision or is_gripper2_collision or is_gripper3_collision or is_gripper4_collision;

      delete gripper1_center;
      delete gripper2_center;
      delete gripper3_center;
      delete gripper4_center;
    }

    // check grasping condition to move object
    bool is_grasping = (contact1 and contact3) or (contact2 and contact4);

    return (not is_obj_collision) and (not is_gripper_collision) and is_grasping;
}

void plan()
{
    // construct the state space we are planning in
    auto object_pose_space(std::make_shared<ob::SE2StateSpace>());
    auto contact1_space(std::make_shared<ob::DiscreteStateSpace>(0,1));
    auto contact2_space(std::make_shared<ob::DiscreteStateSpace>(0,1));
    auto contact3_space(std::make_shared<ob::DiscreteStateSpace>(0,1));
    auto contact4_space(std::make_shared<ob::DiscreteStateSpace>(0,1));
    ob::StateSpacePtr space = object_pose_space + contact1_space + contact2_space + contact3_space + contact4_space;

    // set the bounds for the R^2 part of SE(2)
    ob::RealVectorBounds bounds(2);
    bounds.setLow(0, -2.0);
    bounds.setLow(1, -1.5);
    bounds.setHigh(0, 2.0);
    bounds.setHigh(1, 1.5);
    object_pose_space->setBounds(bounds);

    // create a start state
    ob::ScopedState<> start(space);
    start[0] = -1.5;
    start[1] = 1.0;
    start[2] = 0.0;
    start->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(1)->value = 1;
    start->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(2)->value = 1;
    start->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(3)->value = 1;
    start->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(4)->value = 1;

    // create a goal state
    ob::ScopedState<> goal(space);
    goal[0] = 1.5;
    goal[1] = 1.0;
    goal[2] = 0.0;
    goal->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(1)->value = 1;
    goal->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(2)->value = 1;
    goal->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(3)->value = 1;
    goal->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(4)->value = 1;

    // define a simple setup class
    og::SimpleSetup ss(space);

    // set state validity checking for this space
    ss.setStateValidityChecker([](const ob::State *state) { return isStateValid(state); });

    // set the start and goal states
    ss.setStartAndGoalStates(start, goal);

    // this call is optional, but we put it in to get more output information
    ss.setup();
    ss.print();

    // attempt to solve the problem within the specified planning time
    ob::PlannerStatus solved = ss.solve(10.0);

    if (solved)
    {
        std::cout << "Found solution:" << std::endl;
        // print the path to screen
        ss.simplifySolution();

        // ss.getSolutionPath().print(std::cout);
        // ss.getSolutionPath().printAsMatrix(std::cout);

        std::string filename = "/tmp/ompl_path.txt";
        std::ofstream file_stream;
        file_stream.open(filename, std::ios::out);
        // ss.getSolutionPath().printAsMatrix(file_stream);
        std::cout << "Saved the path to " << filename << std::endl;

        og::PathGeometric& path(ss.getSolutionPath());
        for(unsigned int i=0; i<path.getStateCount(); ++i) {
            // get each element state value
            const ob::State* state = path.getState(i);
            const auto *object_pose_state = state->as<ob::CompoundState>()->as<ob::SE2StateSpace::StateType>(0);
            const auto *contact1_state = state->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(1);
            const auto *contact2_state = state->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(2);
            const auto *contact3_state = state->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(3);
            const auto *contact4_state = state->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(4);

            double obj_x = object_pose_state->getX();
            double obj_y = object_pose_state->getY();
            double obj_theta = object_pose_state->getYaw();
            int contact1 = contact1_state->value;
            int contact2 = contact2_state->value;
            int contact3 = contact3_state->value;
            int contact4 = contact4_state->value;

            std::stringstream output_oneline;
            output_oneline << obj_x << " " << obj_y << " " << obj_theta << " " << contact1 << " " << contact2 << " " << contact3 << " " << contact4 << std::endl;
            std::cout << output_oneline.str();
            file_stream << output_oneline.str();
        }
    }
    else {
        std::cout << "No solution found" << std::endl;
    }
}

int main(int /*argc*/, char ** /*argv*/)
{
    std::cout << "OMPL version: " << OMPL_VERSION << std::endl;

    plan();

    return 0;
}
