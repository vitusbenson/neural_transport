from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from torch_geometric.utils import scatter

from neural_transport.models.gnn.mesh import *
from neural_transport.models.regulargrid import ACTIVATIONS, RegularGridModel
from neural_transport.tools.conversion import *


class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out, layer_norm=True, act="swish"):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(n_in, n_hid, bias=True),
            ACTIVATIONS[act](),
            nn.Linear(n_hid, n_out, bias=(not layer_norm)),
        )

        self.norm = nn.LayerNorm(n_out) if layer_norm else nn.Identity()

    def forward(self, x):
        return self.norm(self.mlp(x))


class MessagePassing(nn.Module):
    def __init__(
        self,
        n_hid=128,
        layer_norm=True,
        act="swish",
        update_sender=False,
        edge_reduction="mean",
    ):
        super().__init__()

        self.edge_mlp = MLP(3 * n_hid, n_hid, n_hid, layer_norm=layer_norm, act=act)
        self.receiver_mlp = MLP(2 * n_hid, n_hid, n_hid, layer_norm=layer_norm, act=act)
        self.sender_mlp = (
            MLP(n_hid, n_hid, n_hid, layer_norm=layer_norm, act=act)
            if update_sender
            else None
        )

        self.edge_reduction = edge_reduction

    def forward(self, edge_attr, idx_sender, idx_receiver, x_sender, x_receiver=None):
        if x_receiver is None:
            x_receiver = x_sender

        edge_update = self.edge_mlp(
            torch.cat(
                [edge_attr, x_sender[:, idx_sender], x_receiver[:, idx_receiver]], dim=2
            )
        )

        edges_collated = scatter(
            edge_update, idx_receiver, dim=1, reduce=self.edge_reduction
        )

        node_update_receiver = self.receiver_mlp(
            torch.cat([x_receiver, edges_collated], dim=2)
        )

        edge_attr = edge_attr + edge_update
        x_receiver = x_receiver + node_update_receiver

        if self.sender_mlp:
            node_update_sender = self.sender_mlp(x_sender)
            x_sender = x_sender + node_update_sender
            return x_sender, x_receiver, edge_attr
        else:
            return x_receiver, edge_attr


class GraphCastGNN(nn.Module):
    def __init__(
        self,
        n_grid,
        n_mesh=3,
        n_g2m=4,
        n_mm=4,
        n_m2g=4,
        n_hid=128,
        n_layers=5,
        layer_norm=True,
        act="swish",
        edge_reduction="mean",
    ):
        super().__init__()

        self.embed_grid_nodes = MLP(
            n_grid, n_hid, n_hid, layer_norm=layer_norm, act=act
        )
        self.embed_mesh_nodes = MLP(
            n_mesh, n_hid, n_hid, layer_norm=layer_norm, act=act
        )
        self.embed_g2m_edges = MLP(n_g2m, n_hid, n_hid, layer_norm=layer_norm, act=act)
        self.embed_mm_edges = MLP(n_mm, n_hid, n_hid, layer_norm=layer_norm, act=act)
        self.embed_m2g_edges = MLP(n_m2g, n_hid, n_hid, layer_norm=layer_norm, act=act)

        self.encoder = MessagePassing(
            n_hid,
            layer_norm=layer_norm,
            act=act,
            update_sender=True,
            edge_reduction=edge_reduction,
        )
        self.processor_layers = nn.ModuleList(
            [
                MessagePassing(
                    n_hid,
                    layer_norm=layer_norm,
                    act=act,
                    update_sender=False,
                    edge_reduction=edge_reduction,
                )
                for i in range(n_layers)
            ]
        )
        self.decoder = MessagePassing(
            n_hid,
            layer_norm=layer_norm,
            act=act,
            update_sender=False,
            edge_reduction=edge_reduction,
        )

    def forward(
        self,
        x_grid,
        x_mesh,
        g2m_edge_attr,
        mm_edge_attr,
        m2g_edge_attr,
        g2m_edge_index,
        mm_edge_index,
        m2g_edge_index,
    ):
        B, N, C = x_grid.shape

        x_grid = self.embed_grid_nodes(x_grid)

        x_mesh = self.embed_mesh_nodes(x_mesh).expand(B, -1, -1)

        g2m_edge_attr = self.embed_g2m_edges(g2m_edge_attr).expand(B, -1, -1)
        mm_edge_attr = self.embed_mm_edges(mm_edge_attr).expand(B, -1, -1)
        m2g_edge_attr = self.embed_m2g_edges(m2g_edge_attr).expand(B, -1, -1)

        x_grid, x_mesh, g2m_edge_attr = self.encoder(
            g2m_edge_attr, g2m_edge_index[0], g2m_edge_index[1], x_grid, x_mesh
        )

        for processor_layer in self.processor_layers:
            x_mesh, mm_edge_attr = processor_layer(
                mm_edge_attr, mm_edge_index[0], mm_edge_index[1], x_mesh
            )

        x_grid, m2g_edge_attr = self.decoder(
            m2g_edge_attr, m2g_edge_index[0], m2g_edge_index[1], x_mesh, x_grid
        )

        return x_grid


class GraphCast(RegularGridModel):

    def init_model(
        self,
        in_chans=1,
        out_chans=19,
        embed_dim=256,
        num_layers=8,
        mesh_min_level=1,
        mesh_max_level=3,
        multimesh=True,
        gridname="latlon5.625",
        g2m_radius_fraction=0.6,
    ) -> None:

        self.init_mesh(
            mesh_min_level=mesh_min_level,
            mesh_max_level=mesh_max_level,
            multimesh=multimesh,
            gridname=gridname,
            g2m_radius_fraction=g2m_radius_fraction,
        )

        self.gnn = GraphCastGNN(
            n_grid=in_chans + 4,
            n_mesh=4,
            n_g2m=4,
            n_mm=5,
            n_m2g=4,
            n_hid=embed_dim,
            n_layers=num_layers,
            layer_norm=True,
            act="swish",
            edge_reduction="mean",
        )

        self.readout = MLP(
            embed_dim,
            embed_dim,
            out_chans,
            layer_norm=False,
            act="swish",
        )

    def init_mesh(
        self, mesh_min_level, mesh_max_level, multimesh, gridname, g2m_radius_fraction
    ) -> None:
        if multimesh:
            edge_features = []
            edge_idxs = []

            for i in range(mesh_min_level, mesh_max_level + 1):
                grid = ICONGrid.create(mesh_min_level, i, resolved_locations=None)
                ds = grid.to_mesh(hex_not_tri=True)
                edge_features.append(ds.edge_features.values)
                edge_idxs.append(ds.edge_idxs.values)

            edge_features = np.concatenate(edge_features, axis=0)
            edge_idxs = np.concatenate(edge_idxs, axis=0)
        else:
            grid = ICONGrid.create(
                mesh_min_level, mesh_max_level, resolved_locations=None
            )
            ds = grid.to_mesh(hex_not_tri=True)
            edge_features = ds.edge_features.values
            edge_idxs = ds.edge_idxs.values

        node_features = ds.node_features.values

        self.register_buffer(
            "mm_x",
            torch.from_numpy(node_features.astype("float32")),
        )
        self.register_buffer(
            "mm_edge_index",
            torch.from_numpy(edge_idxs.astype("int")).t().contiguous(),
        )
        self.register_buffer(
            "mm_edge_attr",
            torch.from_numpy(edge_features.astype("float32")).contiguous(),
        )

        v = np.array(grid.hex_vertices)
        nodes = np.stack(latlon_to_xyz(v[:, 1], v[:, 0]), axis=-1)
        edge_distances = np.linalg.norm(
            nodes[ds.edge_idxs.values[:, 0]] - nodes[ds.edge_idxs.values[:, 1]],
            axis=-1,
        )
        max_edge_length = (
            edge_distances.max()
        )  # grid.to_mesh(hex_not_tri=hex_not_tri).edge_features.values[:, 0].max()

        g2m = grid.generate_g2m_mesh(
            gridname,
            hex_not_tri=True,
            radius=g2m_radius_fraction * max_edge_length,
        )

        self.register_buffer(
            "grid_x",
            torch.from_numpy(g2m.grid_node_features.values.astype("float32")),
        )
        self.register_buffer(
            "g2m_edge_index",
            torch.from_numpy(g2m.edge_idxs.values.astype("int")).t().contiguous(),
        )
        self.register_buffer(
            "g2m_edge_attr",
            torch.from_numpy(g2m.edge_features.values.astype("float32")).contiguous(),
        )

        m2g = grid.generate_m2g_mesh(gridname, hex_not_tri=True)

        self.register_buffer(
            "m2g_edge_index",
            torch.from_numpy(m2g.edge_idxs.values.astype("int")).t().contiguous(),
        )
        self.register_buffer(
            "m2g_edge_attr",
            torch.from_numpy(m2g.edge_features.values.astype("float32")).contiguous(),
        )

    def model(self, x_in):

        B, C, n_lat, n_lon = x_in.shape

        x_grid = torch.cat(
            [
                x_in.reshape(B, C, n_lat * n_lon).permute(0, 2, 1),
                self.grid_x.expand(B, -1, -1),
            ],
            dim=-1,
        )

        g2m_edge_attr = self.g2m_edge_attr.expand(B, -1, -1)
        mm_edge_attr = self.mm_edge_attr.expand(B, -1, -1)
        m2g_edge_attr = self.m2g_edge_attr.expand(B, -1, -1)

        g2m_edge_index = self.g2m_edge_index
        mm_edge_index = self.mm_edge_index
        m2g_edge_index = self.m2g_edge_index

        x_mesh = self.mm_x.expand(B, -1, -1)

        x_feats = self.gnn(
            x_grid,
            x_mesh,
            g2m_edge_attr,
            mm_edge_attr,
            m2g_edge_attr,
            g2m_edge_index,
            mm_edge_index,
            m2g_edge_index,
        )

        x_out = self.readout(x_feats)

        x_out = x_out.reshape(B, n_lat, n_lon, -1).permute(0, 3, 1, 2)

        return x_out
