// Copyright 2021 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "./csg_tree.h"
#include "./impl.h"
#include "./parallel.h"
#include "manifold/polygon.h"

namespace manifold {
/**
 * Constructs a smooth version of the input mesh by creating tangents; this
 * method will throw if you have supplied tangents with your mesh already. The
 * actual triangle resolution is unchanged; use the Refine() method to
 * interpolate to a higher-resolution curve.
 *
 * By default, every edge is calculated for maximum smoothness (very much
 * approximately), attempting to minimize the maximum mean Curvature magnitude.
 * No higher-order derivatives are considered, as the interpolation is
 * independent per triangle, only sharing constraints on their boundaries.
 *
 * @param meshGL input MeshGL.
 * @param sharpenedEdges If desired, you can supply a vector of sharpened
 * halfedges, which should in general be a small subset of all halfedges. Order
 * of entries doesn't matter, as each one specifies the desired smoothness
 * (between zero and one, with one the default for all unspecified halfedges)
 * and the halfedge index (3 * triangle index + [0,1,2] where 0 is the edge
 * between triVert 0 and 1, etc).
 *
 * At a smoothness value of zero, a sharp crease is made. The smoothness is
 * interpolated along each edge, so the specified value should be thought of as
 * an average. Where exactly two sharpened edges meet at a vertex, their
 * tangents are rotated to be colinear so that the sharpened edge can be
 * continuous. Vertices with only one sharpened edge are completely smooth,
 * allowing sharpened edges to smoothly vanish at termination. A single vertex
 * can be sharpened by sharping all edges that are incident on it, allowing
 * cones to be formed.
 */
Manifold Manifold::Smooth(const MeshGL& meshGL,
                          const std::vector<Smoothness>& sharpenedEdges) {
  std::shared_ptr<Impl> impl = std::make_shared<Impl>(meshGL);
  impl->CreateTangents(impl->UpdateSharpenedEdges(sharpenedEdges));
  return Manifold(impl);
}

/**
 * Constructs a smooth version of the input mesh by creating tangents; this
 * method will throw if you have supplied tangents with your mesh already. The
 * actual triangle resolution is unchanged; use the Refine() method to
 * interpolate to a higher-resolution curve.
 *
 * By default, every edge is calculated for maximum smoothness (very much
 * approximately), attempting to minimize the maximum mean Curvature magnitude.
 * No higher-order derivatives are considered, as the interpolation is
 * independent per triangle, only sharing constraints on their boundaries.
 *
 * @param meshGL64 input MeshGL64.
 * @param sharpenedEdges If desired, you can supply a vector of sharpened
 * halfedges, which should in general be a small subset of all halfedges. Order
 * of entries doesn't matter, as each one specifies the desired smoothness
 * (between zero and one, with one the default for all unspecified halfedges)
 * and the halfedge index (3 * triangle index + [0,1,2] where 0 is the edge
 * between triVert 0 and 1, etc).
 *
 * At a smoothness value of zero, a sharp crease is made. The smoothness is
 * interpolated along each edge, so the specified value should be thought of as
 * an average. Where exactly two sharpened edges meet at a vertex, their
 * tangents are rotated to be colinear so that the sharpened edge can be
 * continuous. Vertices with only one sharpened edge are completely smooth,
 * allowing sharpened edges to smoothly vanish at termination. A single vertex
 * can be sharpened by sharping all edges that are incident on it, allowing
 * cones to be formed.
 */
Manifold Manifold::Smooth(const MeshGL64& meshGL64,
                          const std::vector<Smoothness>& sharpenedEdges) {
  std::shared_ptr<Impl> impl = std::make_shared<Impl>(meshGL64);
  impl->CreateTangents(impl->UpdateSharpenedEdges(sharpenedEdges));
  return Manifold(impl);
}

/**
 * Constructs a tetrahedron centered at the origin with one vertex at (1,1,1)
 * and the rest at similarly symmetric points.
 */
Manifold Manifold::Tetrahedron() {
  return Manifold(std::make_shared<Impl>(Impl::Shape::Tetrahedron));
}

/**
 * Constructs a unit cube (edge lengths all one), by default in the first
 * octant, touching the origin. If any dimensions in size are negative, or if
 * all are zero, an empty Manifold will be returned.
 *
 * @param size The X, Y, and Z dimensions of the box.
 * @param center Set to true to shift the center to the origin.
 */
Manifold Manifold::Cube(vec3 size, bool center) {
  if (size.x < 0.0 || size.y < 0.0 || size.z < 0.0 || la::length(size) == 0.) {
    return Invalid();
  }
  mat3x4 m({{size.x, 0.0, 0.0}, {0.0, size.y, 0.0}, {0.0, 0.0, size.z}},
           center ? (-size / 2.0) : vec3(0.0));
  return Manifold(std::make_shared<Impl>(Manifold::Impl::Shape::Cube, m));
}

/**
 * Constructs a new manifold from a vector of other manifolds. This is a purely
 * topological operation, so care should be taken to avoid creating
 * overlapping results. It is the inverse operation of Decompose().
 *
 * @param manifolds A vector of Manifolds to lazy-union together.
 */
Manifold Manifold::Compose(const std::vector<Manifold>& manifolds) {
  std::vector<std::shared_ptr<CsgLeafNode>> children;
  for (const auto& manifold : manifolds) {
    children.push_back(manifold.pNode_->ToLeafNode());
  }
  return Manifold(CsgLeafNode::Compose(children));
}

/**
 * This operation returns a vector of Manifolds that are topologically
 * disconnected. If everything is connected, the vector is length one,
 * containing a copy of the original. It is the inverse operation of Compose().
 */
std::vector<Manifold> Manifold::Decompose() const {
  UnionFind<> uf(NumVert());
  // Graph graph;
  auto pImpl_ = GetCsgLeafNode().GetImpl();
  for (const Halfedge& halfedge : pImpl_->halfedge_) {
    if (halfedge.IsForward()) uf.unionXY(halfedge.startVert, halfedge.endVert);
  }
  std::vector<int> componentIndices;
  const int numComponents = uf.connectedComponents(componentIndices);

  if (numComponents == 1) {
    std::vector<Manifold> meshes(1);
    meshes[0] = *this;
    return meshes;
  }
  Vec<int> vertLabel(componentIndices);

  const int numVert = NumVert();
  std::vector<Manifold> meshes;
  for (int i = 0; i < numComponents; ++i) {
    auto impl = std::make_shared<Impl>();
    // inherit original object's precision
    impl->epsilon_ = pImpl_->epsilon_;
    impl->tolerance_ = pImpl_->tolerance_;

    Vec<int> vertNew2Old(numVert);
    const int nVert =
        std::copy_if(countAt(0), countAt(numVert), vertNew2Old.begin(),
                     [i, &vertLabel](int v) { return vertLabel[v] == i; }) -
        vertNew2Old.begin();
    impl->vertPos_.resize(nVert);
    vertNew2Old.resize(nVert);
    gather(vertNew2Old.begin(), vertNew2Old.end(), pImpl_->vertPos_.begin(),
           impl->vertPos_.begin());

    Vec<int> faceNew2Old(NumTri());
    const auto& halfedge = pImpl_->halfedge_;
    const int nFace =
        std::copy_if(countAt(0_uz), countAt(NumTri()), faceNew2Old.begin(),
                     [i, &vertLabel, &halfedge](int face) {
                       return vertLabel[halfedge[3 * face].startVert] == i;
                     }) -
        faceNew2Old.begin();

    if (nFace == 0) continue;
    faceNew2Old.resize(nFace);

    impl->GatherFaces(*pImpl_, faceNew2Old);
    impl->ReindexVerts(vertNew2Old, pImpl_->NumVert());
    impl->Finish();

    meshes.push_back(Manifold(impl));
  }
  return meshes;
}
}  // namespace manifold
