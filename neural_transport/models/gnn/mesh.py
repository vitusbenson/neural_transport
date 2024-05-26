# Adapted from https://github.com/google-deepmind/graphcast/blob/main/graphcast/icosahedral_mesh.py with the following license:
# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import numpy as np
import scipy
import trimesh
import xarray as xr

from neural_transport.datasets.grids import CARBOSCOPE_79_LONLAT, OBSPACK_287_LONLAT


def xyz_to_latlon(x, y, z):
    """Converts 3-D vector to lat-lon coordinates. Lat in [-90, 90], Lon in [0, 360]."""
    lat = 90 - np.degrees(np.arccos(z))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def latlon_to_xyz(lat, lon):
    """Converts lat-lon coordinates to 3-D vector. Lat in [-90, 90], Lon in [0, 360]."""
    x = np.sin(np.radians(90 - lat)) * np.cos(np.radians(lon))
    y = np.sin(np.radians(90 - lat)) * np.sin(np.radians(lon))
    z = np.cos(np.radians(90 - lat))
    return x, y, z


def x_rotation(vector, theta):
    """Rotates 3-D vector around x-axis"""

    theta = np.radians(theta)
    one = np.ones_like(theta)
    zero = np.zeros_like(theta)

    R = np.array(
        [
            [one, zero, zero],
            [zero, np.cos(theta), -np.sin(theta)],
            [zero, np.sin(theta), np.cos(theta)],
        ]
    ).transpose(2, 0, 1)
    return np.einsum("bji,bi->bj", R, vector)  # np.dot(R, vector)


def y_rotation(vector, theta):
    """Rotates 3-D vector around y-axis"""
    theta = np.radians(theta)
    one = np.ones_like(theta)
    zero = np.zeros_like(theta)

    R = np.array(
        [
            [np.cos(theta), zero, np.sin(theta)],
            [zero, one, zero],
            [-np.sin(theta), zero, np.cos(theta)],
        ]
    ).transpose(2, 0, 1)
    return np.einsum("bji,bi->bj", R, vector)  # np.dot(R, vector)


def z_rotation(vector, theta):
    """Rotates 3-D vector around z-axis"""
    theta = np.radians(theta)
    one = np.ones_like(theta)
    zero = np.zeros_like(theta)

    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), zero],
            [np.sin(theta), np.cos(theta), zero],
            [zero, zero, one],
        ]
    ).transpose(2, 0, 1)
    return np.einsum("bji,bi->bj", R, vector)  # np.dot(R, vector)


def pos_in_local_grid(lat_receiver, lon_receiver, lat_sender, lon_sender):
    """Returns the position of the sender in a relative local grid of the receiver"""
    pos = y_rotation(
        z_rotation(
            np.stack(latlon_to_xyz(90 - lat_sender, lon_sender), axis=-1), -lon_receiver
        ),
        90 - lat_receiver,
    )
    return pos[:, 0], pos[:, 1], pos[:, 2]


def great_circle_distance(lat_receiver, lon_receiver, lat_sender, lon_sender, radius=1):
    """Returns the great circle distance between two points on a sphere"""
    lat1, lng1 = np.radians(90 - lat_receiver), np.radians(lon_receiver)
    lat2, lng2 = np.radians(90 - lat_sender), np.radians(lon_sender)

    sin_lat1, cos_lat1 = np.sin(lat1), np.cos(lat1)
    sin_lat2, cos_lat2 = np.sin(lat2), np.cos(lat2)

    delta_lng = np.subtract(lng2, lng1)
    cos_delta_lng, sin_delta_lng = np.cos(delta_lng), np.sin(delta_lng)

    d = np.arctan2(
        np.sqrt(
            (cos_lat2 * sin_delta_lng) ** 2
            + (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lng) ** 2
        ),
        sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng,
    )

    return radius * d


def spherical_area(a, b, c):
    """Returns the area of a spherical triangle"""
    t = abs(np.inner(a, np.cross(b, c)))
    t /= 1 + np.inner(a, b) + np.inner(b, c) + np.inner(a, c)
    return 2 * np.arctan(t)


class ICONGrid:
    """Class for generating an ICON grid"""

    def __init__(self, min_level, max_level, resolved_locations=None):
        """Creates a new ICON grid

        Level 1 ~2560km resolution
        Level 2 ~1280km resolution
        Level 3 ~640km resolution
        Level 4 ~320km resolution
        Level 5 ~160km resolution
        Level 6 ~80km resolution

        """
        self.min_level = min_level
        self.max_level = max_level
        self.resolved_locations = (
            np.array(sorted(resolved_locations))
            if resolved_locations is not None
            else None
        )
        self.flexible_grid = resolved_locations is not None

        self.hex_vertices = []
        self.hex_vertices_mapping = {}
        self.tri_vertices = []
        self.hex_faces = []
        self.tri_faces = []

    @classmethod
    def create(cls, min_level, max_level, resolved_locations=None):
        """Creates a new ICON grid"""
        self = cls(min_level, max_level, resolved_locations=resolved_locations)

        self._generate()

        return self

    def to_netcdf(self, path=None):
        """Writes the triangular grid to a netcdf file that can be used by CDO for regridding"""
        clon_vertices = np.array(
            [[self.hex_vertices[i][0] for i in triangle] for triangle in self.tri_faces]
        )
        clat_vertices = np.array(
            [[self.hex_vertices[i][1] for i in triangle] for triangle in self.tri_faces]
        )

        cell_area = self._compute_tri_area()

        ds = xr.Dataset(
            {
                "clon_vertices": (("cell", "nv"), clon_vertices),
                "clat_vertices": (("cell", "nv"), clat_vertices),
                "cell_index": (
                    ("cell",),
                    np.arange(1, len(self.tri_vertices) + 1, dtype="int32"),
                ),
                "cell_area": (("cell",), cell_area),
            },
            coords={
                "clon": (("cell",), np.array(self.tri_vertices)[:, 0]),
                "clat": (("cell",), np.array(self.tri_vertices)[:, 1]),
            },
        )

        ds.clon.attrs = {
            "standard_name": "longitude",
            "long_name": "center longitude",
            "units": "degree",
            "bounds": "clon_vertices",
        }
        ds.clat.attrs = {
            "standard_name": "latitude",
            "long_name": "center latitude",
            "units": "degree",
            "bounds": "clat_vertices",
        }
        ds.clon_vertices.attrs = {"units": "degree"}
        ds.clat_vertices.attrs = {"units": "degree"}
        ds.cell_index.attrs = {"long_name": "cell index"}

        if path is not None:
            path = Path(path)
            path.unlink(missing_ok=True)
            ds.to_netcdf(path)

        return ds

    def to_mesh(self, path=None, hex_not_tri=False):
        """Writes the grid to a netcdf file that can be used as a mesh in a graph neural network."""
        if hex_not_tri:
            cell_area = self._compute_hex_area()
            nodes = np.array(self.hex_vertices)
        else:
            cell_area = self._compute_tri_area()
            nodes = np.array(self.tri_vertices)

        cos_lat = np.cos(np.radians(nodes[:, 1]))
        cos_lon = np.cos(np.radians(nodes[:, 0]))
        sin_lon = np.sin(np.radians(nodes[:, 0]))

        node_features = np.stack([cell_area, cos_lat, cos_lon, sin_lon], axis=-1)

        edge_idxs = self._faces_to_edges(
            self.tri_faces if hex_not_tri else self.hex_faces
        )

        edge_length = great_circle_distance(
            nodes[edge_idxs[:, 1], 1],
            nodes[edge_idxs[:, 1], 0],
            nodes[edge_idxs[:, 0], 1],
            nodes[edge_idxs[:, 0], 0],
            radius=1 / np.pi,
        )

        dual_edge_length = self._compute_dual_edge_length(
            edge_idxs, hex_not_tri=hex_not_tri
        )

        # dual_edge_idxs = self._faces_to_edges(self.hex_faces)

        x_rel, y_rel, z_rel = self._compute_rel_pos(nodes, edge_idxs)

        edge_features = np.stack(
            [edge_length, dual_edge_length, x_rel, y_rel, z_rel], axis=-1
        )

        ds = xr.Dataset(
            {
                "node_features": (("n_nodes", "n_node_feats"), node_features),
                "edge_features": (("n_edges", "n_edge_feats"), edge_features),
                "edge_idxs": (("n_edges", "nv"), edge_idxs),
            },
            coords={
                "n_nodes": (("n_nodes",), np.arange(len(nodes), dtype="int32")),
                "n_edges": (("n_edges",), np.arange(len(edge_idxs), dtype="int32")),
                "n_node_feats": (
                    ("n_node_feats",),
                    ["cell_area", "cos_lat", "cos_lon", "sin_lon"],
                ),
                "n_edge_feats": (
                    ("n_edge_feats",),
                    ["edge_length", "dual_edge_length", "x_rel", "y_rel", "z_rel"],
                ),
            },
        )
        if path is not None:
            path = Path(path)
            path.unlink(missing_ok=True)
            ds.to_netcdf(path)

        return ds  # node_features, edge_features, edge_idxs

    def generate_g2m_mesh(self, grid, path=None, hex_not_tri=False, radius=1.0):
        # TODO: THIS IS CURRENTLY BROKEN!!
        ## NEED TO CHECK WHY IT DOES NOT BUILD EDGES USING ALL GRID NODES !!!
        ## L5 grid should have ~0.07 max radius...
        grid_latitude = grid.lat.values
        grid_longitude = grid.lon.values

        grid_positions = _grid_lat_lon_to_coordinates(
            grid_latitude, grid_longitude
        ).reshape([-1, 3])

        mesh_latitude = np.array(
            self.hex_vertices if hex_not_tri else self.tri_vertices
        )[:, 1]
        mesh_longitude = np.array(
            self.hex_vertices if hex_not_tri else self.tri_vertices
        )[:, 0]

        grid_edge_indices, mesh_edge_indices = radius_query_indices(
            grid_positions=grid_positions,
            mesh_latitude=mesh_latitude,
            mesh_longitude=mesh_longitude,
            radius=radius,
        )

        grid_nodes = np.stack(
            xyz_to_latlon(
                grid_positions[:, 0], grid_positions[:, 1], grid_positions[:, 2]
            ),
            axis=-1,
        )

        grid_cos_lat = np.cos(np.radians(grid_nodes[:, 0]))
        grid_cos_lon = np.cos(np.radians(grid_nodes[:, 1]))
        grid_sin_lon = np.sin(np.radians(grid_nodes[:, 1]))

        grid_lat_res = np.abs(np.diff(grid_latitude)).mean()
        grid_lon_res = np.abs(np.diff(grid_longitude)).mean()

        grid_cell_area = (
            np.pi
            * (
                np.sin(np.radians(grid_nodes[:, 0] + grid_lat_res / 2))
                - np.sin(np.radians(grid_nodes[:, 0] - grid_lat_res / 2))
            )
            * grid_lon_res
        )

        grid_node_features = np.stack(
            [grid_cell_area, grid_cos_lat, grid_cos_lon, grid_sin_lon], axis=-1
        )

        edge_idxs = np.stack([grid_edge_indices, mesh_edge_indices], axis=-1)

        mesh_nodes = np.array(self.tri_vertices if hex_not_tri else self.hex_vertices)

        edge_length = great_circle_distance(
            mesh_nodes[edge_idxs[:, 1], 1],
            mesh_nodes[edge_idxs[:, 1], 0],
            grid_nodes[edge_idxs[:, 0], 1],
            grid_nodes[edge_idxs[:, 0], 0],
            radius=1 / np.pi,
        )

        x_rel, y_rel, z_rel = self._compute_rel_pos(
            grid_nodes, edge_idxs, receiver_nodes=mesh_nodes
        )

        edge_features = np.stack([edge_length, x_rel, y_rel, z_rel], axis=-1)

        ds = xr.Dataset(
            {
                "grid_node_features": (
                    ("n_grid_nodes", "n_node_feats"),
                    grid_node_features,
                ),
                "edge_features": (("n_edges", "n_edge_feats"), edge_features),
                "edge_idxs": (("n_edges", "nv"), edge_idxs),
            },
            coords={
                "n_grid_nodes": (
                    ("n_grid_nodes",),
                    np.arange(len(grid_nodes), dtype="int32"),
                ),
                "n_edges": (("n_edges",), np.arange(len(edge_idxs), dtype="int32")),
                "n_node_feats": (
                    ("n_node_feats",),
                    ["cell_area", "cos_lat", "cos_lon", "sin_lon"],
                ),
                "n_edge_feats": (
                    ("n_edge_feats",),
                    ["edge_length", "x_rel", "y_rel", "z_rel"],
                ),
            },
        )
        if path is not None:
            path = Path(path)
            path.unlink(missing_ok=True)
            ds.to_netcdf(path)

        return ds

    def generate_m2g_mesh(self, grid, path=None, hex_not_tri=False, radius=1.0):
        if not hex_not_tri:
            raise NotImplementedError

        grid_latitude = grid.lat.values
        grid_longitude = grid.lon.values

        grid_positions = _grid_lat_lon_to_coordinates(
            grid_latitude, grid_longitude
        ).reshape([-1, 3])

        grid_edge_indices, mesh_edge_indices = in_mesh_triangle_indices(
            grid_positions=grid_positions, mesh=self
        )

        grid_nodes = np.stack(
            xyz_to_latlon(
                grid_positions[:, 0], grid_positions[:, 1], grid_positions[:, 2]
            ),
            axis=-1,
        )

        grid_cos_lat = np.cos(np.radians(grid_nodes[:, 0]))
        grid_cos_lon = np.cos(np.radians(grid_nodes[:, 1]))
        grid_sin_lon = np.sin(np.radians(grid_nodes[:, 1]))

        grid_lat_res = np.abs(np.diff(grid_latitude)).mean()
        grid_lon_res = np.abs(np.diff(grid_longitude)).mean()

        grid_cell_area = (
            np.pi
            * (
                np.sin(np.radians(grid_nodes[:, 0] + grid_lat_res / 2))
                - np.sin(np.radians(grid_nodes[:, 0] - grid_lat_res / 2))
            )
            * grid_lon_res
        )

        grid_node_features = np.stack(
            [grid_cell_area, grid_cos_lat, grid_cos_lon, grid_sin_lon], axis=-1
        )

        edge_idxs = np.stack([mesh_edge_indices, grid_edge_indices], axis=-1)

        mesh_nodes = np.array(self.tri_vertices if hex_not_tri else self.hex_vertices)

        edge_length = great_circle_distance(
            grid_nodes[edge_idxs[:, 1], 1],
            grid_nodes[edge_idxs[:, 1], 0],
            mesh_nodes[edge_idxs[:, 0], 1],
            mesh_nodes[edge_idxs[:, 0], 0],
            radius=1 / np.pi,
        )

        x_rel, y_rel, z_rel = self._compute_rel_pos(
            mesh_nodes, edge_idxs, receiver_nodes=grid_nodes
        )

        edge_features = np.stack([edge_length, x_rel, y_rel, z_rel], axis=-1)

        ds = xr.Dataset(
            {
                "grid_node_features": (
                    ("n_grid_nodes", "n_node_feats"),
                    grid_node_features,
                ),
                "edge_features": (("n_edges", "n_edge_feats"), edge_features),
                "edge_idxs": (("n_edges", "nv"), edge_idxs),
            },
            coords={
                "n_grid_nodes": (
                    ("n_grid_nodes",),
                    np.arange(len(grid_nodes), dtype="int32"),
                ),
                "n_edges": (("n_edges",), np.arange(len(edge_idxs), dtype="int32")),
                "n_node_feats": (
                    ("n_node_feats",),
                    ["cell_area", "cos_lat", "cos_lon", "sin_lon"],
                ),
                "n_edge_feats": (
                    ("n_edge_feats",),
                    ["edge_length", "x_rel", "y_rel", "z_rel"],
                ),
            },
        )
        if path is not None:
            path = Path(path)
            path.unlink(missing_ok=True)
            ds.to_netcdf(path)

        return ds

    def _compute_hex_area(self):
        """Computes the area of each hexagon in the grid"""
        hex_areas = []
        for i, face in enumerate(self.hex_faces):
            area = 0
            for f1, f2 in zip(face, face[1:] + face[:1]):
                area += self._compute_triangle_area(
                    self.hex_vertices[i], self.tri_vertices[f1], self.tri_vertices[f2]
                )

            hex_areas.append(area)

        return np.array(hex_areas)

    def _compute_tri_area(self):
        """Computes the area of each triangle in the grid"""
        tri_areas = []
        for face in self.tri_faces:
            tri_areas.append(
                self._compute_triangle_area(*[self.hex_vertices[i] for i in face])
            )

        return np.array(tri_areas)

    def _compute_triangle_area(self, v1, v2, v3):
        """Computes the area of a triangle on a sphere"""
        # Calculate Area of Triangle on Sphere

        a = np.array(latlon_to_xyz(v1[1], v1[0]))
        b = np.array(latlon_to_xyz(v2[1], v2[0]))
        c = np.array(latlon_to_xyz(v3[1], v3[0]))

        return spherical_area(a, b, c)

    def _faces_to_edges(self, faces):
        """Converts a list of faces to a list of edges"""
        edges = []
        for face in faces:
            for f1, f2 in zip(face, face[1:] + face[:1]):
                edges.append([f1, f2])

        return np.array(edges)

    def _compute_dual_edge_length(self, edge_idxs, hex_not_tri=False):
        """Computes the length of the dual edge for each edge, this is the length of the cell boundary that the edge is intersecting."""
        dual_edge_length = []
        for edge in edge_idxs:
            face1 = self.hex_faces[edge[0]] if hex_not_tri else self.tri_faces[edge[0]]
            face2 = self.hex_faces[edge[1]] if hex_not_tri else self.tri_faces[edge[1]]

            dual_edge = list(set(face1).intersection(set(face2)))

            if len(dual_edge) != 2:
                if len(dual_edge) == 1:
                    i1 = dual_edge[0]

                    candidates = list(set(face1).union(set(face2)).difference({i1}))

                    v = (
                        np.array(self.tri_vertices)[candidates]
                        if hex_not_tri
                        else np.array(self.hex_vertices)[candidates]
                    )

                    c1 = (
                        self.hex_vertices[edge[0]]
                        if hex_not_tri
                        else self.tri_vertices[edge[0]]
                    )
                    c2 = (
                        self.hex_vertices[edge[1]]
                        if hex_not_tri
                        else self.tri_vertices[edge[1]]
                    )

                    dist1 = great_circle_distance(v[:, 1], v[:, 0], c1[1], c1[0])
                    dist2 = great_circle_distance(v[:, 1], v[:, 0], c2[1], c2[0])

                    i2 = candidates[np.argmin(dist1 + dist2)]

                else:
                    candidates = list(set(face1).union(set(face2)))

                    v = (
                        np.array(self.tri_vertices)[candidates]
                        if hex_not_tri
                        else np.array(self.hex_vertices)[candidates]
                    )

                    c1 = (
                        self.hex_vertices[edge[0]]
                        if hex_not_tri
                        else self.tri_vertices[edge[0]]
                    )
                    c2 = (
                        self.hex_vertices[edge[1]]
                        if hex_not_tri
                        else self.tri_vertices[edge[1]]
                    )

                    dist1 = great_circle_distance(v[:, 1], v[:, 0], c1[1], c1[0])
                    dist2 = great_circle_distance(v[:, 1], v[:, 0], c2[1], c2[0])

                    idx1, idx2 = np.argpartition(dist1 + dist2, 2)[:2]

                    i1, i2 = candidates[idx1], candidates[idx2]

                dual_edge = [i1, i2]

            v1 = (
                self.tri_vertices[dual_edge[0]]
                if hex_not_tri
                else self.hex_vertices[dual_edge[0]]
            )
            v2 = (
                self.tri_vertices[dual_edge[1]]
                if hex_not_tri
                else self.hex_vertices[dual_edge[1]]
            )

            dual_edge_length.append(
                great_circle_distance(v1[1], v1[0], v2[1], v2[0], radius=1 / np.pi)
            )

        return np.array(dual_edge_length)

    def _compute_rel_pos(self, nodes, edge_idxs, receiver_nodes=None):
        """Computes the relative position of the sender in a local grid of the receiver for all nodes"""
        # x_rel = []
        # y_rel = []
        # z_rel = []

        # for edge in edge_idxs:
        #     v1 = nodes[edge[0]]
        #     v2 = nodes[edge[1]] if receiver_nodes is None else receiver_nodes[edge[1]]

        #     x, y, z = pos_in_local_grid(v2[1], v2[0], v1[1], v1[0])

        #     x_rel.append(x)
        #     y_rel.append(y)
        #     z_rel.append(z)

        v1 = nodes[edge_idxs[:, 0]]
        v2 = (
            nodes[edge_idxs[:, 1]]
            if receiver_nodes is None
            else receiver_nodes[edge_idxs[:, 1]]
        )

        x_rel, y_rel, z_rel = pos_in_local_grid(v2[:, 1], v2[:, 0], v1[:, 1], v1[:, 0])

        return np.array(x_rel), np.array(y_rel), np.array(z_rel)

    def _generate(self):
        """Generates the grid"""

        self._get_icosahedron()

        for l in range(self.max_level):
            self._refine(
                refine_only_resolved=(self.flexible_grid and l >= self.min_level)
            )

    def _get_icosahedron(self):
        """Creates the initial icosahedron"""

        x_offset = 36.0
        atan_half = np.degrees(np.arctan(0.5))

        x_vertex = [
            (i + x_offset) % 360
            for i in [
                0.0,
                0.0,
                36.0,
                72.0,
                108.0,
                144.0,
                180.0,
                216.0,
                252.0,
                288.0,
                324.0,
                0.0,
            ]
        ]
        y_vertex = [
            90.0,
            atan_half,
            -atan_half,
            atan_half,
            -atan_half,
            atan_half,
            -atan_half,
            atan_half,
            -atan_half,
            atan_half,
            -atan_half,
            -90.0,
        ]

        self.hex_vertices = [(lon, lat) for lon, lat in zip(x_vertex, y_vertex)]

        self.tri_faces = [
            [0, 1, 3],
            [0, 3, 5],
            [0, 5, 7],
            [0, 7, 9],
            [0, 9, 1],
            [1, 10, 2],
            [1, 2, 3],
            [3, 2, 4],
            [3, 4, 5],
            [5, 4, 6],
            [5, 6, 7],
            [7, 6, 8],
            [7, 8, 9],
            [9, 8, 10],
            [9, 10, 1],
            [11, 4, 2],
            [11, 6, 4],
            [11, 8, 6],
            [11, 10, 8],
            [11, 2, 10],
        ]

        self.tri_vertices = [
            self._calc_midpoint([self.hex_vertices[i] for i in triangle])
            for triangle in self.tri_faces
        ]

        self.hex_faces = [
            [0, 1, 2, 3, 4],
            [0, 4, 14, 5, 6],
            [7, 6, 5, 19, 15],
            [1, 0, 6, 7, 8],
            [9, 8, 7, 15, 16],
            [2, 1, 8, 9, 10],
            [11, 10, 9, 16, 17],
            [3, 2, 10, 11, 12],
            [13, 12, 11, 17, 18],
            [4, 3, 12, 13, 14],
            [5, 14, 13, 18, 19],
            [15, 16, 17, 18, 19],
        ]

    def _refine(self, refine_only_resolved=False):
        """Refines the grid one time"""

        for i in range(len(self.tri_faces)):
            if refine_only_resolved:
                if not self._resolved_loc_in_triangle(i):
                    continue

            self._refine_triangle(i)

    def _resolved_loc_in_triangle(self, i):
        """Checks if any of the resolved locations is in the triangle"""

        i1, i2, i3 = self.tri_faces[i]
        v1, v2, v3 = (
            self.hex_vertices[i1],
            self.hex_vertices[i2],
            self.hex_vertices[i3],
        )

        # Calculate Area of Triangle

        A = self._calc_area_latlon_safe(v1[0], v2[0], v3[0], v1[1], v2[1], v3[1])

        # Vectorize Calculate Areas of new triangles for all Locations

        x = self.resolved_locations[:, 0]
        y = self.resolved_locations[:, 1]

        A1 = self._calc_area_latlon_safe(x, v2[0], v3[0], y, v2[1], v3[1])
        A2 = self._calc_area_latlon_safe(v1[0], x, v3[0], v1[1], y, v3[1])
        A3 = self._calc_area_latlon_safe(v1[0], v2[0], x, v1[1], v2[1], y)

        # Caluclate sum of vector areas
        # Check if any of the sums is close to triangle area

        return np.isclose((A1 + A2 + A3), A).any()

    def _calc_area_latlon_safe(self, x1, x2, x3, y1, y2, y3):
        """Calculates the area of a triangle on a 2D lat-lon plane, but makes sure that the triangle is in the middle of the plane, not on the edge"""

        large_lon_mask = (x1 > 270) | (x2 > 270) | (x3 > 270)
        x1 = np.where(large_lon_mask & (x1 < 90), x1 + 360, x1)
        x2 = np.where(large_lon_mask & (x2 < 90), x2 + 360, x2)
        x3 = np.where(large_lon_mask & (x3 < 90), x3 + 360, x3)

        A = abs((y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2)) / 2.0)

        x1 = x1 % 360
        x2 = x2 % 360
        x3 = x3 % 360

        return A

    def _refine_triangle(self, i):
        """Refines a single triangle"""

        i1, i2, i3 = self.tri_faces[i]

        # FIRST NEED NEW HEX VERTEX LOCATIONS

        v1, v12, v2, v23, v3, v31 = self._calc_new_hex_vertices(i1, i2, i3)

        # THEN NEED NEW TRI VERTEX LOCATIONS

        c, c1, c2, c3 = self._calc_new_tri_vertices(v1, v12, v2, v23, v3, v31)

        # print(c, self.tri_vertices[i])
        # if not np.isclose(c, self.tri_vertices[i]).all():
        #     print(c, self.tri_vertices[i], i)
        c = self.tri_vertices[i]

        # THEN GET NEW TRI VERTEX IDXS (register)

        ic = i
        ic1, ic2, ic3 = self._register_tri_vertices(c1, c2, c3)

        # THEN GET NEW HEX VERTEX IDXS (register) + Update/register new Hex faces

        i12 = self._register_hex_vertex_face(i1, i2, v12, ic, ic1, ic2)
        i23 = self._register_hex_vertex_face(i2, i3, v23, ic, ic2, ic3)
        i31 = self._register_hex_vertex_face(i3, i1, v31, ic, ic3, ic1)

        # THEN UPDATE OLD HEX FACES

        self._update_hex_face(i1, ic, ic1)
        self._update_hex_face(i2, ic, ic2)
        self._update_hex_face(i3, ic, ic3)

        # THEN GET NEW TRI FACES

        self._register_tri_face(ic, [i12, i23, i31])
        self._register_tri_face(ic1, [i1, i12, i31])
        self._register_tri_face(ic2, [i12, i2, i23])
        self._register_tri_face(ic3, [i23, i3, i31])

    def _calc_new_hex_vertices(self, i1, i2, i3):
        """Calculates the new hex vertices for a triangle"""

        v1, v2, v3 = (
            self.hex_vertices[i1],
            self.hex_vertices[i2],
            self.hex_vertices[i3],
        )
        v12 = self._calc_midpoint((v1, v2))
        v23 = self._calc_midpoint((v2, v3))
        v31 = self._calc_midpoint((v3, v1))

        return v1, v12, v2, v23, v3, v31

    def _calc_new_tri_vertices(self, v1, v12, v2, v23, v3, v31):
        """Calculates the new tri vertices for a triangle"""

        c = self._calc_midpoint((v1, v2, v3))
        c1 = self._calc_midpoint((v1, v12, v31))
        c2 = self._calc_midpoint((v12, v2, v23))
        c3 = self._calc_midpoint((v23, v3, v31))

        return c, c1, c2, c3

    def _register_tri_vertices(self, c1, c2, c3):
        """Registers the new tri vertices and returns their indices"""

        ic1 = len(self.tri_vertices)
        ic2, ic3 = (ic1 + 1, ic1 + 2)

        self.tri_vertices.extend([c1, c2, c3])

        return ic1, ic2, ic3

    def _register_hex_vertex_face(self, i1, i2, v12, ic, ic1, ic2):
        """Registers the new hex vertex and returns its index"""

        key = tuple(sorted((i1, i2)))

        if key not in self.hex_vertices_mapping:
            self.hex_vertices_mapping[key] = len(self.hex_vertices)
            self.hex_vertices.append(v12)

            # HERE: Add new hex face
            # [Outside Triangle, 3 new small triangles]
            left_hex_face = self.hex_faces[i1]
            right_hex_face = self.hex_faces[i2]

            inter_i = [
                i
                for i in set(left_hex_face).intersection(set(right_hex_face))
                if i != ic
            ][0]

            self.hex_faces.append([inter_i, ic2, ic, ic1])

        else:
            i12 = self.hex_vertices_mapping[key]

            hex_face = self.hex_faces[i12]

            new_hex_face = []
            for f in hex_face:
                if f == ic:
                    new_hex_face.extend([ic2, ic, ic1])
                else:
                    new_hex_face.append(f)

            self.hex_faces[i12] = new_hex_face

        return self.hex_vertices_mapping[key]

    def _update_hex_face(self, i1, ic, ic1):
        """Updates the hex face"""

        hex_face = self.hex_faces[i1]
        new_hex_face = []
        for f in hex_face:
            if f == ic:
                new_hex_face.append(ic1)
            else:
                new_hex_face.append(f)

        self.hex_faces[i1] = new_hex_face

    def _register_tri_face(self, ic, tri_face):
        """Registers the new tri face"""

        if ic < len(self.tri_faces):
            self.tri_faces[ic] = tri_face
        else:
            assert ic == len(self.tri_faces)
            self.tri_faces.append(tri_face)

    def _calc_midpoint(self, points):
        """Calculates the midpoint of a list of points"""

        data = []

        for point in points:
            x, y, z = latlon_to_xyz(point[1], point[0])

            data.append([x, y, z])

        data = np.array(data).mean(0)

        data /= np.linalg.norm(data)

        x, y, z = data

        lat, lon = xyz_to_latlon(x, y, z)

        return [lon % 360, lat]


# %%


def _grid_lat_lon_to_coordinates(
    grid_latitude: np.ndarray, grid_longitude: np.ndarray
) -> np.ndarray:
    """Lat [num_lat] lon [num_lon] to 3d coordinates [num_lat, num_lon, 3]."""
    # Copyright 2023 DeepMind Technologies Limited.

    # Convert to spherical coordinates phi and theta defined in the grid.
    # Each [num_latitude_points, num_longitude_points]
    phi_grid, theta_grid = np.meshgrid(
        np.deg2rad(grid_longitude), np.deg2rad(90 - grid_latitude)
    )

    # [num_latitude_points, num_longitude_points, 3]
    # Note this assumes unit radius, since for now we model the earth as a
    # sphere of unit radius, and keep any vertical dimension as a regular grid.
    return np.stack(
        [
            np.cos(phi_grid) * np.sin(theta_grid),
            np.sin(phi_grid) * np.sin(theta_grid),
            np.cos(theta_grid),
        ],
        axis=-1,
    )


def radius_query_indices(
    *,
    grid_positions: np.ndarray,
    mesh_latitude: np.ndarray,
    mesh_longitude: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns mesh-grid edge indices for radius query.

    Args:
    grid_latitude: Latitude values for the grid [num_lat_points]
    grid_longitude: Longitude values for the grid [num_lon_points]
    mesh_positions: .
    radius: Radius of connectivity in R3. for a sphere of unit radius.

    Returns:
    tuple with `grid_indices` and `mesh_indices` indicating edges between the
    grid and the mesh such that the distances in a straight line (not geodesic)
    are smaller than or equal to `radius`.
    * grid_indices: Indices of shape [num_edges], that index into a
        [num_lat_points, num_lon_points] grid, after flattening the leading axes.
    * mesh_indices: Indices of shape [num_edges], that index into mesh.vertices.
    """
    # Copyright 2023 DeepMind Technologies Limited.

    # [num_grid_points=num_lat_points * num_lon_points, 3]

    # [num_mesh_points, 3]
    mesh_positions = np.stack(latlon_to_xyz(mesh_latitude, mesh_longitude), axis=-1)
    kd_tree = scipy.spatial.cKDTree(mesh_positions)

    # [num_grid_points, num_mesh_points_per_grid_point]
    # Note `num_mesh_points_per_grid_point` is not constant, so this is a list
    # of arrays, rather than a 2d array.
    query_indices = kd_tree.query_ball_point(x=grid_positions, r=radius)

    grid_edge_indices = []
    mesh_edge_indices = []
    for grid_index, mesh_neighbors in enumerate(query_indices):
        grid_edge_indices.append(np.repeat(grid_index, len(mesh_neighbors)))
        mesh_edge_indices.append(mesh_neighbors)

    # [num_edges]
    grid_edge_indices = np.concatenate(grid_edge_indices, axis=0).astype(int)
    mesh_edge_indices = np.concatenate(mesh_edge_indices, axis=0).astype(int)

    return grid_edge_indices, mesh_edge_indices


def in_mesh_triangle_indices(
    *, grid_positions: np.ndarray, mesh: ICONGrid
) -> tuple[np.ndarray, np.ndarray]:
    """Returns mesh-grid edge indices for grid points contained in mesh triangles.

    Args:
    grid_latitude: Latitude values for the grid [num_lat_points]
    grid_longitude: Longitude values for the grid [num_lon_points]
    mesh: Mesh object.

    Returns:
    tuple with `grid_indices` and `mesh_indices` indicating edges between the
    grid and the mesh vertices of the triangle that contain each grid point.
    The number of edges is always num_lat_points * num_lon_points * 3
    * grid_indices: Indices of shape [num_edges], that index into a
        [num_lat_points, num_lon_points] grid, after flattening the leading axes.
    * mesh_indices: Indices of shape [num_edges], that index into mesh.vertices.
    """
    # Copyright 2023 DeepMind Technologies Limited.

    # [num_grid_points=num_lat_points * num_lon_points, 3]

    vertices = np.array(mesh.tri_vertices)
    vertices = np.stack(latlon_to_xyz(vertices[:, 1], vertices[:, 0]), axis=-1)

    mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=mesh.tri_faces)

    # [num_grid_points] with mesh face indices for each grid point.
    _, _, query_face_indices = trimesh.proximity.closest_point(
        mesh_trimesh, grid_positions
    )

    # [num_grid_points, 3] with mesh node indices for each grid point.
    mesh_edge_indices = np.array(mesh.tri_faces)[query_face_indices]

    # [num_grid_points, 3] with grid node indices, where every row simply contains
    # the row (grid_point) index.
    grid_indices = np.arange(grid_positions.shape[0])
    grid_edge_indices = np.tile(grid_indices.reshape([-1, 1]), [1, 3])

    # Flatten to get a regular list.
    # [num_edges=num_grid_points*3]
    mesh_edge_indices = mesh_edge_indices.reshape([-1])
    grid_edge_indices = grid_edge_indices.reshape([-1])

    return grid_edge_indices, mesh_edge_indices


def get_gridnc_from_grid(grid):
    grid_parts = grid.replace("icosa", "").split("-")
    if len(grid_parts) == 1:
        min_level = 0
        max_level = int(grid_parts[0][-1])
        resolved_locations = None
    elif len(grid_parts) == 2:
        min_level = int(grid_parts[1][-1])
        max_level = int(grid_parts[0][-1])
        resolved_locations = None
    elif len(grid_parts) == 3:
        min_level = int(grid_parts[1][-1])
        max_level = int(grid_parts[0][-1])
        resolved_locations = {
            "a287": OBSPACK_287_LONLAT,
            "a79": CARBOSCOPE_79_LONLAT,
        }[grid_parts[2]]
    else:
        raise ValueError(f"Invalid grid {grid}")

    ico = ICONGrid.create(
        min_level=min_level,
        max_level=max_level,
        resolved_locations=resolved_locations,
    )
    ds = ico.to_netcdf()
    return ds


# %%
if __name__ == "__main__":
    grid = ICONGrid.create(1, 4, resolved_locations=[[10, 10], [170, -35]])
