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
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/control/spaces/DiscreteControlSpace.h>
#include <ompl/control/SimpleSetup.h>
#include <ompl/config.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>

namespace ob = ompl::base;
namespace oc = ompl::control;

void propagate(const oc::SpaceInformation *si, const ob::State *state,
               const oc::Control* control, const double duration, ob::State *result)
{
  static double timeStep = 0.1;
  int nsteps = ceil(duration / timeStep);
  double dt = duration / nsteps;
  const double *u = control->as<oc::RealVectorControlSpace::ControlType>()->values;

  // copy current state to result
  si->getStateSpace()->copyState(result, state);

  // get each element state value
  auto *object_pose_state = result->as<ob::CompoundState>()->as<ob::SE2StateSpace::StateType>(0);
  auto *contact1_state = result->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(1);
  auto *contact2_state = result->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(2);
  auto *contact3_state = result->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(3);
  auto *contact4_state = result->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(4);

  double obj_x = object_pose_state->getX();
  double obj_y = object_pose_state->getY();
  double obj_theta = object_pose_state->getYaw();
  int contact1 = contact1_state->value;
  int contact2 = contact2_state->value;
  int contact3 = contact3_state->value;
  int contact4 = contact4_state->value;

  // update contact
  bool change_contact = false;
  double switch_contact_prob = 0.1;
  if (u[3] < switch_contact_prob) {
    contact1_state->value = (contact1_state->value == 0 ? 1 : 0);
    change_contact = true;
  }
  if (u[4] < switch_contact_prob) {
    contact2_state->value = (contact2_state->value == 0 ? 1 : 0);
    change_contact = true;
  }
  if (u[5] < switch_contact_prob) {
    contact3_state->value = (contact3_state->value == 0 ? 1 : 0);
    change_contact = true;
  }
  if (u[6] < switch_contact_prob) {
    contact4_state->value = (contact4_state->value == 0 ? 1 : 0);
    change_contact = true;
  }

  // update object pose
  bool is_grasping = (contact1 and contact3) or (contact2 and contact4);
  if ((not change_contact) and is_grasping) {
    for(int i=0; i<nsteps; i++) {
      object_pose_state->setX(obj_x + dt * u[0]);
      object_pose_state->setY(obj_y + dt * u[1]);
      object_pose_state->setYaw(obj_theta + dt * u[2]);

      if (!si->satisfiesBounds(result))
        return;
    }
  }
}

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

bool isStateValid(const oc::SpaceInformation *si, const ob::State *state)
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

  return si->satisfiesBounds(state) and (not is_obj_collision) and (not is_gripper_collision);
}

int main(int /*argc*/, char ** /*argv*/)
{
  std::cout << "OMPL version: " << OMPL_VERSION << std::endl;

  // construct the state space we are planning in
  auto object_pose_space(std::make_shared<ob::SE2StateSpace>());
  auto contact1_space(std::make_shared<ob::DiscreteStateSpace>(0,1));
  auto contact2_space(std::make_shared<ob::DiscreteStateSpace>(0,1));
  auto contact3_space(std::make_shared<ob::DiscreteStateSpace>(0,1));
  auto contact4_space(std::make_shared<ob::DiscreteStateSpace>(0,1));
  ob::StateSpacePtr space = object_pose_space + contact1_space + contact2_space + contact3_space + contact4_space;

  // set the bounds for object x, y
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
  start->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(1)->value = 0;
  start->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(2)->value = 0;
  start->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(3)->value = 0;
  start->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(4)->value = 0;

  // create a goal state
  ob::ScopedState<> goal(space);
  goal[0] = 1.5;
  goal[1] = 1.0;
  goal[2] = 0.0;
  goal->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(1)->value = 0;
  goal->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(2)->value = 0;
  goal->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(3)->value = 0;
  goal->as<ob::CompoundState>()->as<ob::DiscreteStateSpace::StateType>(4)->value = 0;

  // define a control manifold
  // [ToDo] control space which compounds RealVectorControlSpace and DiscreteControlSpace should be used
  oc::ControlSpacePtr cmanifold(std::make_shared<oc::RealVectorControlSpace>(space, 7));

  // set the bounds for the control manifold
  ob::RealVectorBounds cbounds(7);
  // bounds for x, y
  cbounds.setLow(0, -0.1);
  cbounds.setHigh(0, 0.1);
  cbounds.setLow(1, -0.1);
  cbounds.setHigh(1, 0.1);
  // bounds for theta
  cbounds.setLow(2, -0.1);
  cbounds.setHigh(2, 0.1);
  // bounds for contact
  cbounds.setLow(3, 0.0);
  cbounds.setHigh(3, 1.0);
  cbounds.setLow(4, 0.0);
  cbounds.setHigh(4, 1.0);
  cbounds.setLow(5, 0.0);
  cbounds.setHigh(5, 1.0);
  cbounds.setLow(6, 0.0);
  cbounds.setHigh(6, 1.0);
  cmanifold->as<oc::RealVectorControlSpace>()->setBounds(cbounds);

  // define a simple setup class
  oc::SimpleSetup ss(cmanifold);

  // set the start and goal states
  ss.setStartAndGoalStates(start, goal);

  // set state validity checking for this space
  const oc::SpaceInformation *si = ss.getSpaceInformation().get();
  ss.setStateValidityChecker([si](const ob::State *state)
                             {
                               return isStateValid(si, state);
                             });
  ss.setStatePropagator([si](const ob::State *state, const oc::Control* control,
                             const double duration, ob::State *result)
                        {
                          propagate(si, state, control, duration, result);
                        });
  ss.getSpaceInformation()->setPropagationStepSize(0.1);
  ss.getSpaceInformation()->setMinMaxControlDuration(1, 10);

  // attempt to solve the problem within the specified planning time
  ob::PlannerStatus solved = ss.solve(30.0);

  if (solved) {
    std::cout << "Found solution:" << std::endl;

    std::string filename = "/tmp/ompl_path.txt";
    std::ofstream file_stream;
    file_stream.open(filename, std::ios::out);
    std::cout << "Saved the path to " << filename << std::endl;

    oc::PathControl& path(ss.getSolutionPath());
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

    if (!ss.haveExactSolutionPath()) {
      std::cout << "Solution is approximate. Distance to actual goal is " <<
        ss.getProblemDefinition()->getSolutionDifference() << std::endl;
    }
  }
  else {
    std::cout << "No solution found" << std::endl;
  }
}
