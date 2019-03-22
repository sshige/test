#include <iostream>
#include <cnoid/Body>
#include <cnoid/JointPath>
#include <cnoid/ExecutablePath>
#include <cnoid/BodyLoader>
#include <cnoid/Link>
#include <cnoid/EigenUtil>
#include <cnoid/FileUtil>


using namespace std;
using namespace cnoid;
using namespace Eigen;

int main(int argc, char** argv)
{

  // load robot model
  // string modelfile("/home/murooka/ros/rhp_ws/src/openhrp3/sample/model/sample1.wrl");
  string modelfile("/home/murooka/src/choreonoid/share/model/PA10/PA10.body");
  BodyLoader loader;
  BodyPtr robot = loader.load(modelfile);
  if(!robot){
    cout << modelfile << " cannot be loaded." << endl;
    return 1;
  }

  // print robot information
  cout << "dof: " << robot->numJoints() << endl;
  cout << "base link name: " << robot->rootLink()->name() << endl;
  cout << "base link pos: \n" << robot->rootLink()->p() << endl;
  cout << "base link rot: \n" << robot->rootLink()->R() << endl;

  JointPathPtr jointPathRootToEnd = getCustomJointPath(robot, robot->rootLink(), robot->link("J7"));
  Link *linkEnd = jointPathRootToEnd->endLink();

  jointPathRootToEnd->joint(0)->q() = 0.5;
  jointPathRootToEnd->joint(1)->q() = 0.5;
  jointPathRootToEnd->joint(2)->q() = 0.5;
  jointPathRootToEnd->joint(3)->q() = 0.5;
  jointPathRootToEnd->joint(4)->q() = 0.5;
  jointPathRootToEnd->joint(5)->q() = 0.5;

  // calculate FK
  jointPathRootToEnd->calcForwardKinematics();
  Vector3d currentEndPos = linkEnd->p();
  Matrix3d currentEndRot = linkEnd->R();
  cout << "currentEndPos: \n" << currentEndPos << endl;
  cout << "currentEndRot: \n" << currentEndRot << endl;

  // calculate IK
  currentEndPos.z() -= 0.01;
  if (jointPathRootToEnd->calcInverseKinematics(currentEndPos, currentEndRot)) {
    for (int i = 0; i < jointPathRootToEnd->numJoints(); i++) {
      cout << "joint(" << i << ")->q(): " << jointPathRootToEnd->joint(i)->q() << endl;
    }
  } else {
    cout << "failed to solve IK" << endl;
  }

  return 0;
}
