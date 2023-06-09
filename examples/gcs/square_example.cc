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

void PlotResults(const Eigen::Matrix<double, 1, Eigen::Dynamic> xData, const Eigen::Matrix<double, 1, Eigen::Dynamic> yData) {
  using common::CallPython;
  using common::ToPythonTuple;

  Eigen::Matrix<int, 1, 4> squareX {1, 1, -1, -1};
  Eigen::Matrix<int, 1, 4> squareY {-1, 1, 1, -1};
  Vector2d start(0.0, -2.0), goal(0.0, 2.0);
  
  CallPython("figure", 1);
  CallPython("clf");

  CallPython("plot", start[0], start[1], "b*");
  CallPython("plot", goal[0], goal[1], "g*");

  CallPython("plot", xData.transpose(), yData.transpose(), "b");
  CallPython("fill", squareX.transpose(), squareY.transpose(), "r");
  CallPython("axis", "equal");

  // Give time for Python to plot.
  std::this_thread::sleep_for(std::chrono::seconds(1));
}

void do_main() {
  const int kDimension = 2;
  GcsTrajectoryOptimization gcs(kDimension);

  Vector2d start(0.0, -2.0), goal(0.0, 2.0);

  HPolyhedron region_1 = HPolyhedron::MakeBox(Vector2d(-4, -4), Vector2d(4, -1));
  HPolyhedron region_2 = HPolyhedron::MakeBox(Vector2d(-4, -4), Vector2d(-1, 4));
  HPolyhedron region_3 = HPolyhedron::MakeBox(Vector2d(1, -4), Vector2d(4, 4));
  HPolyhedron region_4 = HPolyhedron::MakeBox(Vector2d(-4, 1), Vector2d(4, 4));

  ConvexSets regions_ = MakeConvexSets(region_1, region_2, region_3, region_4);

  auto& regions = gcs.AddRegions(regions_, 1);
  auto& source = gcs.AddRegions(MakeConvexSets(Point(start)), 0);
  auto& target = gcs.AddRegions(MakeConvexSets(Point(goal)), 0);

  gcs.AddEdges(source, regions);
  gcs.AddEdges(regions, target);

  GraphOfConvexSetsOptions options;
  options.max_rounded_paths = 3;
  
  auto [traj, result] = gcs.SolvePath(source, target, options);

  std::cout << "Result is success : " << result.is_success() << std::endl;
  std::cout << "Rows : " << traj.rows() << ", Cols : " << traj.cols() << std::endl;
  std::cout << "Number of segments : " << traj.get_number_of_segments() << std::endl;

  ////////////////////////////////////////////////////////
  ///////// PLOTTING CODE 
  ////////////////////////////////////////////////////////
  int numberOfSegments = traj.get_number_of_segments() - 2; // Remove the source and target segment since those are trivial.

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
}

}  // namespace
}  // namespace gcs
}  // namespace examples
}  // namespace drake

int main() {
  drake::examples::gcs::do_main();
  return 0;
}