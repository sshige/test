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
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/SimpleSetup.h>

#include <ompl/config.h>
#include <iostream>
#include <fstream>

namespace ob = ompl::base;
namespace og = ompl::geometric;

bool isStateValid(const ob::State *state)
{
    // cast the abstract state type to the type we expect
    const auto *se2state = state->as<ob::SE2StateSpace::StateType>();

    // extract the first component of the state and cast it to what we expect
    const auto *pos = se2state->as<ob::RealVectorStateSpace::StateType>(0);
    double pos_x = pos->values[0];
    double pos_y = pos->values[1];

    // check validity of state defined by pos & rot
    double obj_radius = 0.2;
    double obst1_center[2] = { 0.0, 1.0 };
    double obst1_radius = 0.75;
    bool is_obst1_collision = (pos_x - obst1_center[0]) * (pos_x - obst1_center[0]) + (pos_y - obst1_center[1]) * (pos_y - obst1_center[1]) < (obst1_radius + obj_radius) * (obst1_radius + obj_radius);
    double obst2_center[2] = { 0.0, -1.0 };
    double obst2_radius = 0.75;
    bool is_obst2_collision = (pos_x - obst2_center[0]) * (pos_x - obst2_center[0]) + (pos_y - obst2_center[1]) * (pos_y - obst2_center[1]) < (obst2_radius + obj_radius) * (obst2_radius + obj_radius);
    // std::cout << "obst1: " << is_obst1_collision << "  obst2: " << is_obst2_collision << std::endl;

    // return a value that is always true but uses the two variables we define, so we avoid compiler warnings
    return (not is_obst1_collision) and (not is_obst2_collision);
}

void plan()
{
    // construct the state space we are planning in
    auto space(std::make_shared<ob::SE2StateSpace>());

    // set the bounds for the R^2 part of SE(2)
    ob::RealVectorBounds bounds(2);
    bounds.setLow(0, -2.0);
    bounds.setLow(1, -1.5);
    bounds.setHigh(0, 2.0);
    bounds.setHigh(1, 1.5);

    space->setBounds(bounds);

    // define a simple setup class
    og::SimpleSetup ss(space);

    // set state validity checking for this space
    ss.setStateValidityChecker([](const ob::State *state) { return isStateValid(state); });

    // create a random start state
    ob::ScopedState<ob::SE2StateSpace> start(space);
    start->setXY(-1.5, 1.0);
    start->setYaw(0.0);

    // create a random goal state
    ob::ScopedState<ob::SE2StateSpace> goal(space);
    goal->setXY(1.5, 1.0);
    goal->setYaw(0.0);
    // goal.random();

    // set the start and goal states
    ss.setStartAndGoalStates(start, goal);

    // this call is optional, but we put it in to get more output information
    ss.setup();
    ss.print();

    // attempt to solve the problem within one second of planning time
    ob::PlannerStatus solved = ss.solve(1.0);

    if (solved)
    {
        std::cout << "Found solution:" << std::endl;
        // print the path to screen
        ss.simplifySolution();

        // ss.getSolutionPath().print(std::cout);
        ss.getSolutionPath().printAsMatrix(std::cout);

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
