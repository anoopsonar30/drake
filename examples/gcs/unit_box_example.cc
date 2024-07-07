#include <chrono>
#include <thread>
#include <vector>
#include <cmath>
#include <iostream>

#include "drake/common/proto/call_python.h"

#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/gurobi_solver.h"
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

void PlotResults(const Eigen::Matrix<double, 1, Eigen::Dynamic> xData, const Eigen::Matrix<double, 1, Eigen::Dynamic> yData, const std::string xlabel = "x-axis", const std::string ylabel = "y-axis") {
  using common::CallPython;
  using common::ToPythonTuple;

  CallPython("figure", 1);
  CallPython("clf");
//   CallPython("subplot", 2, 1, 1);
  CallPython("plot", xData.transpose(), yData.transpose(), "r");
//   CallPython("plot", traj.time.transpose(), traj.x.row(0).transpose(), "c");
  CallPython("xlabel", xlabel);
  CallPython("ylabel", ylabel);
//   CallPython("legend", ToPythonTuple("desired zmp", "planned com",
//                                      "planned cop", "actual com"));

  // Give time for Python to plot.
  std::this_thread::sleep_for(std::chrono::seconds(1));
}

void do_main() {
  std::cout << solvers::MosekSolver::is_available() << std::endl;

  const int kDimension = 2;
  GcsTrajectoryOptimization gcs(kDimension);

  // Add a single region (the unit box), and plan a line segment inside that
  // box.
  Vector2d start(-0.5, -0.5), goal(0.5, 0.5);
  
  auto& regions = gcs.AddRegions(MakeConvexSets(HPolyhedron::MakeUnitBox(kDimension)), 1);
  auto& source = gcs.AddRegions(MakeConvexSets(Point(start)), 0);
  auto& target = gcs.AddRegions(MakeConvexSets(Point(goal)), 0);

  gcs.AddEdges(source, regions);
  gcs.AddEdges(regions, target);

  auto [traj, traj_control_points, result] = gcs.SolvePath(source, target);

  std::cout << "Result is success : " << result.is_success() << std::endl;
  std::cout << "Rows : " << traj.rows() << ", Cols : " << traj.cols() << std::endl;
  // std::cout << traj.segment(1).value(0)(0) << ", " << traj.segment(1).value(1)(0) << std::endl;

  int numPoints = 50;
  int numIntervals = numPoints - 1;

  Eigen::Matrix<double, 1, Eigen::Dynamic> x_data;
  Eigen::Matrix<double, 1, Eigen::Dynamic> y_data;
  x_data.resize(1, numPoints);
  y_data.resize(1, numPoints);

  double start_time = traj.segment(1).start_time();
  double end_time = traj.segment(1).end_time();
  double timeStepSize = (end_time - start_time) / numIntervals;

  for (int i = 0; i < numPoints; i++)
  {
    double timeStep = start_time + timeStepSize * i;
    auto coords = traj.segment(1).value(timeStep);

    x_data(1, i) = coords(0);
    y_data(1, i) = coords(1);
  }

  PlotResults(x_data, y_data);
}

}  // namespace
}  // namespace gcs
}  // namespace examples
}  // namespace drake

int main() {
  drake::examples::gcs::do_main();
  return 0;
}


