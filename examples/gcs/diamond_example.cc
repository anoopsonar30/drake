#include <chrono>
#include <thread>
#include <vector>
#include <cmath>
#include <iostream>

#include "drake/common/proto/call_python.h"
#include "drake/common/pointer_cast.h"
#include "drake/common/scope_exit.h"
#include "drake/common/symbolic/decompose.h"

#include "drake/common/trajectories/bezier_curve.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/solve.h"
#include "drake/solvers/solver_options.h"
#include "drake/solvers/decision_variable.h"

#include "drake/geometry/optimization/graph_of_convex_sets.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/point.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/planning/trajectory_optimization/gcs_trajectory_optimization.h"


namespace drake {
namespace examples {
namespace gcs{
namespace {

using Eigen::Vector2d;
using geometry::optimization::ConvexSets;
using geometry::optimization::GraphOfConvexSetsOptions;
using geometry::optimization::HPolyhedron;
using geometry::optimization::MakeConvexSets;
using geometry::optimization::Point;
using geometry::optimization::VPolytope;
using planning::trajectory_optimization::GcsTrajectoryOptimization;
using solvers::MathematicalProgram;
using symbolic::Expression;
using symbolic::MakeMatrixContinuousVariable;
using symbolic::MakeVectorContinuousVariable;
using solvers::VectorXDecisionVariable;
using trajectories::BezierCurve;
using trajectories::CompositeTrajectory;
using trajectories::Trajectory;

void PlotResults(const Eigen::Matrix<double, 1, Eigen::Dynamic> xData, const Eigen::Matrix<double, 1, Eigen::Dynamic> yData) {
  using common::CallPython;
  using common::ToPythonTuple;

  Eigen::Matrix<int, 1, 4> diamondX {1, 0, -1, 0};
  Eigen::Matrix<int, 1, 4> diamondY {0, 1, 0, -1};
  Vector2d start(0.0, -2.0), goal(0.0, 2.0);
  
  CallPython("figure", 1);
  CallPython("clf");

  CallPython("plot", start[0], start[1], "b*");
  CallPython("plot", goal[0], goal[1], "g*");

  CallPython("plot", xData.transpose(), yData.transpose(), "b");
  CallPython("fill", diamondX.transpose(), diamondY.transpose(), "r");
  CallPython("axis", "equal");

  // Give time for Python to plot.
  std::this_thread::sleep_for(std::chrono::seconds(1));
}


bool GurobiOrMosekSolverAvailable() {
  return (solvers::MosekSolver::is_available() &&
          solvers::MosekSolver::is_enabled()) ||
         (solvers::GurobiSolver::is_available() &&
          solvers::GurobiSolver::is_enabled());
}

void GCSDiamondExample(){
  const int kDimension = 2;
  GcsTrajectoryOptimization gcs(kDimension);

  Vector2d start(0.0, -2.0), goal(0.0, 2.0);

  Eigen::Matrix<double, 3, 2> A_bl;
  A_bl << -1, 0, 0, -1, 1, 1;
  Eigen::Matrix<double, 3, 2> A_br;
  A_br << 1, 0, 0, -1, -1, 1;
  Eigen::Matrix<double, 3, 2> A_tl;
  A_tl << -1, 0, 0, 1, 1, -1;
  Eigen::Matrix<double, 3, 2> A_tr;
  A_tr << 1, 0, 0, 1, -1, -1;
  Eigen::Vector3d b(3, 3, -1);

  HPolyhedron region_1(A_bl, b);
  HPolyhedron region_2(A_br, b);
  HPolyhedron region_3(A_tl, b);
  HPolyhedron region_4(A_tr, b);

  ConvexSets regions_ = MakeConvexSets(region_1, region_2, region_3, region_4);
  // ConvexSets regions_ = MakeConvexSets(region_1, region_3);

  auto& regions = gcs.AddRegions(regions_, 1);
  auto& source = gcs.AddRegions(MakeConvexSets(Point(start)), 0);
  auto& target = gcs.AddRegions(MakeConvexSets(Point(goal)), 0);

  gcs.AddEdges(source, regions);
  gcs.AddEdges(regions, target);

  gcs.AddPathLengthCost(1.0);
  gcs.AddTimeCost(1.0);

  GraphOfConvexSetsOptions options;
  options.max_rounded_paths = 3;

 auto [traj, result] = gcs.SolvePath(source, target, options);

  std::cout << "Number of segments : " << traj.get_number_of_segments() << std::endl;

  //////////////////////////////////////////////////////
  /////// PLOTTING CODE 
  //////////////////////////////////////////////////////

  // Remove the first and last segments, i.e. the source and target segment since those are trivial.
  int numberOfSegments = traj.get_number_of_segments() - 2;

  Eigen::Matrix<double, 1, Eigen::Dynamic> x_data;
  Eigen::Matrix<double, 1, Eigen::Dynamic> y_data;

  int numPoints = 50;
  int numIntervals = numPoints - 1;

  x_data.resize(1, numPoints * numberOfSegments);
  y_data.resize(1, numPoints * numberOfSegments);

  for (int segmentID = 0; segmentID < numberOfSegments; segmentID++)
  {
    double start_time = traj.segment(segmentID + 1).start_time();
    double end_time = traj.segment(segmentID + 1).end_time();
    double timeStepSize = (end_time - start_time) / numIntervals;

    for (int i = 0; i < numPoints; i++)
    {
      double timeStep = start_time + timeStepSize * i;
      auto coords = traj.segment(segmentID + 1).value(timeStep);

      x_data(1, segmentID * numPoints + i) = coords(0);
      y_data(1, segmentID * numPoints + i) = coords(1);
    }
  }

  PlotResults(x_data, y_data);
  return;
}

void DiamondWithSingleBezierSegment(const int order=2, const double minimum_distance=1.5){
  const int kDimension = 2;
  // const int order = 2;
  // const double minimum_distance = 1.0;
  
  Vector2d start(0.0, -2.0), goal(0.0, 2.0);

  // Create a MathematicalProgram.
  MathematicalProgram prog;

  MatrixX<symbolic::Variable> segment_control = prog.NewContinuousVariables(kDimension, order + 1, "xu");
  auto segment_trajectory = BezierCurve<Expression>(0, 1, segment_control.cast<drake::symbolic::Expression>());

  Eigen::MatrixXd start_ = VectorX<double>::Zero(kDimension);
  start_(0) = start(0);
  start_(1) = start(1);
  prog.AddLinearEqualityConstraint(segment_control.col(0) == start_);

  Eigen::MatrixXd goal_ = VectorX<double>::Zero(kDimension);
  goal_(0) = goal(0);
  goal_(1) = goal(1);
  prog.AddLinearEqualityConstraint(segment_control.col(order) == goal_);

  // Print statements for debugging
  // std::cout << "Norm : " << segment_trajectory.value(0).norm() << std::endl;
  // std::cout << "Element : " << (drake::symbolic::pow(segment_trajectory.value(0.5)(0), 2) + drake::symbolic::pow(segment_trajectory.value(0.5)(1), 2)) << std::endl;
  // std::cout << "Norm squared : " << drake::symbolic::pow(segment_trajectory.value(0).norm(), 2) << std::endl;
  // prog.AddConstraint((drake::symbolic::pow(segment_trajectory.value(0.5)(0), 2) + drake::symbolic::pow(segment_trajectory.value(0.5)(1), 2)) >= (minimum_distance * minimum_distance));
  // prog.AddConstraint((drake::symbolic::pow(segment_trajectory.value(0.66)(0), 2) + drake::symbolic::pow(segment_trajectory.value(0.6)(1), 2)) >= (minimum_distance * minimum_distance));

  // Constraints for distance from (0, 0)
  for (int j = 0; j < 21; j++) {
      double t = static_cast<double>(j) / 20.0;
      // prog.AddConstraint((drake::symbolic::pow(segment_trajectory.value(t)(0), 2) + drake::symbolic::pow(segment_trajectory.value(t)(1), 2)) >= (minimum_distance * minimum_distance));
      prog.AddConstraint(segment_trajectory.value(t).norm() >= minimum_distance);
  }

  Eigen::MatrixXd guess = VectorX<double>::Zero(kDimension);
  guess(0) = 0.1;
  guess(1) = 0.0;
  prog.SetInitialGuess(segment_control.col(1), guess);

  // Solve the program
  drake::solvers::SnoptSolver solver;
  drake::solvers::SolverOptions options_;
  prog.SetSolverOption(drake::solvers::SnoptSolver::id(), "Major iterations limit", 100000);

  drake::solvers::MathematicalProgramResult result = solver.Solve(prog);

  Eigen::Matrix<double, 1, Eigen::Dynamic> x_data;
  Eigen::Matrix<double, 1, Eigen::Dynamic> y_data;

  int numPoints = 101;
  x_data.resize(1, numPoints);
  y_data.resize(1, numPoints);

  if (result.is_success()) {
      // Fetch the optimal coefficients
      auto optimal_coeffs = result.GetSolution(segment_control);

      // Construct the Bezier curve using the optimal coefficients
      auto optimal_trajectory = BezierCurve<double>(0, 1, optimal_coeffs);

      int index = 0;
      // Now you can use the `optimal_trajectory` as needed, for example:
      for (double t = 0; t < 1.01; t += 0.01) {
          auto point = optimal_trajectory.value(t);
          std::cout << "t: " << t << ", point: " << point.transpose() << std::endl;

          x_data(1, index) = point(0);
          y_data(1, index) = point(1);
          index++;
      }

      PlotResults(x_data, y_data);

  } else {
      std::cout << "No solution found." << std::endl;
  }
}

void DiamondWithMultipleBezierSegments(const int numSegments = 2, const int order=2, const double minimum_distance=1.5){
  const int kDimension = 2;
  // const int order = 2;
  // const double minimum_distance = 1.0;

  Vector2d start(0.0, -2.0), goal(0.0, 2.0);
  // Create a MathematicalProgram.
  MathematicalProgram prog;

  std::vector<MatrixX<symbolic::Variable>> segment_controls(numSegments);
  std::vector<drake::trajectories::BezierCurve<drake::symbolic::Expression>> segment_trajectories(numSegments);
  for (int i = 0; i < numSegments; i++)
  {
    segment_controls[i] = prog.NewContinuousVariables(kDimension, order + 1, "xu_" + std::to_string(i));
    segment_trajectories[i] = BezierCurve<Expression>(0, 1, segment_controls[i].cast<drake::symbolic::Expression>());
  }

  Eigen::MatrixXd start_ = VectorX<double>::Zero(kDimension);
  start_(0) = start(0);
  start_(1) = start(1);
  prog.AddLinearEqualityConstraint(segment_controls[0].col(0) == start_);

  Eigen::MatrixXd goal_ = VectorX<double>::Zero(kDimension);
  goal_(0) = goal(0);
  goal_(1) = goal(1);
  prog.AddLinearEqualityConstraint(segment_controls[numSegments - 1].col(order) == goal_);

  // Print statements for debugging
  // std::cout << "Norm : " << segment_trajectory.value(0).norm() << std::endl;
  // std::cout << "Element : " << (drake::symbolic::pow(segment_trajectory.value(0.5)(0), 2) + drake::symbolic::pow(segment_trajectory.value(0.5)(1), 2)) << std::endl;
  // std::cout << "Norm squared : " << drake::symbolic::pow(segment_trajectory.value(0).norm(), 2) << std::endl;
  // prog.AddConstraint((drake::symbolic::pow(segment_trajectory.value(0.5)(0), 2) + drake::symbolic::pow(segment_trajectory.value(0.5)(1), 2)) >= (minimum_distance * minimum_distance));
  // prog.AddConstraint((drake::symbolic::pow(segment_trajectory.value(0.66)(0), 2) + drake::symbolic::pow(segment_trajectory.value(0.6)(1), 2)) >= (minimum_distance * minimum_distance));

  // Constraint to ensure that end of one segment is beginning of the next segment
  for (int i = 0; i < (numSegments - 1); i++)
  {
    prog.AddLinearEqualityConstraint(segment_controls[i].col(order) == segment_controls[i + 1].col(0));
  }

  // Constraints for distance from (0, 0)
  for (int i = 0; i < numSegments; i++){
    for (int j = 0; j < 6; j++) {
        double t = static_cast<double>(j) / 5.0;
        // prog.AddConstraint((drake::symbolic::pow(segment_trajectory.value(t)(0), 2) + drake::symbolic::pow(segment_trajectory.value(t)(1), 2)) >= (minimum_distance * minimum_distance));
        prog.AddConstraint(segment_trajectories[i].value(t).norm() >= minimum_distance);
    }
  }

  Eigen::MatrixXd guess = VectorX<double>::Zero(kDimension);
  guess(0) = 0.1; // Set to something non-zero so the solver converges to a good solution.
  guess(1) = 0.0;
  prog.SetInitialGuess(segment_controls[0].col(1), guess);
  prog.SetInitialGuess(segment_controls[1].col(0), guess);

  // Solve the program
  drake::solvers::SnoptSolver solver;
  drake::solvers::SolverOptions options_;
  prog.SetSolverOption(drake::solvers::SnoptSolver::id(), "Major iterations limit", 100000);

  drake::solvers::MathematicalProgramResult result = solver.Solve(prog);

  Eigen::Matrix<double, 1, Eigen::Dynamic> x_data;
  Eigen::Matrix<double, 1, Eigen::Dynamic> y_data;

  int numPoints = 101;
  x_data.resize(1, numPoints * numSegments);
  y_data.resize(1, numPoints * numSegments);

  if (result.is_success()) {
      // Fetch the optimal coefficients
      int index = 0;
      for (int i = 0; i < numSegments; i++)
      {
        auto optimal_coeffs = result.GetSolution(segment_controls[i]);
        // Construct the Bezier curve using the optimal coefficients
        auto optimal_trajectory = BezierCurve<double>(0, 1, optimal_coeffs);
      
        // Now you can use the `optimal_trajectory` as needed, for example:
        for (double t = 0; t < 1.01; t += 0.01) {
            auto point = optimal_trajectory.value(t);
            std::cout << "t: " << t << ", point: " << point.transpose() << std::endl;

            x_data(1, index) = point(0);
            y_data(1, index) = point(1);
            index++;
        }
      }
      
      PlotResults(x_data, y_data);

  } else {
      std::cout << "No solution found." << std::endl;
  }
}


void do_main() {
  if (!GurobiOrMosekSolverAvailable()) {
    drake::log()->info("Cannot find Gurobi or Mosek!, skipping execution...");
    return;
  }

  std::string exampleToRun = "DiamondWithSingleBezierSegment";

  // Decide which example to run based on input
  if (exampleToRun == "GCSDiamond")
  {
    // Generates two non-trivial segments using GCS.
    // Uses path length cost and time cost. No chance or distance constraints.
    GCSDiamondExample();
  }
  else if (exampleToRun == "DiamondWithSingleBezierSegment")
  {
    // Generates single segment that follows a distance constraint.
    // No cost used. No GCS.
    DiamondWithSingleBezierSegment(2, 1.5);
  }
  else if (exampleToRun == "DiamondWithMultipleBezierSegments")
  {
    // Generates a sequence of segments that follow distance constraints.
    // No cost used. No GCS.
    // DiamondWithMultipleBezierSegments(1, 1.5);
    DiamondWithMultipleBezierSegments(2, 1, 1.5);
  }
  // TODO : Modify the 3D obstacle course URDF made by Charles to look just like the diamond obstacle scenario (from a top view) with a point object and plot it in 2D.
  // TODO : SingleBezierSegment for diamond example with CHANCE + distance constraints in 2D
  // TODO : MultipleBezierSegment for diamond example with CHANCE + distance constraints in 3D with point object.
  
  // TODO : Diamond example with distance constraint WITH GCS (STILL NEED TO FIGURE OUT HOW TO MODIFY THE UNDERLYING PROGRAM CORRECTLY)
  // TODO : Diamond example with CHANCE + distance constraint WITH GCS
  else
  {
    std::cout << "Invalid example requested." << std::endl;
  }
}

}  // namespace
}  // namespace gcs
}  // namespace examples
}  // namespace drake

int main() {
  drake::examples::gcs::do_main();
  return 0;
}