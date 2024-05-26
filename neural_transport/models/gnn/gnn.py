from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from torch_geometric.utils import scatter

from neural_transport.models.gnn.mesh import *
from neural_transport.tools.conversion import *

ACTIVATIONS = {
    "none": nn.Identity,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": nn.LeakyReLU,
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "hardsigmoid": nn.Hardsigmoid,
    "hardtanh": nn.Hardtanh,
    "swish": nn.SiLU,
}


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


class MultiMeshGNN(nn.Module):
    def __init__(
        self,
        n_in_nodes,
        n_in_edges,
        n_hid=128,
        n_layers=5,
        layer_norm=True,
        act="swish",
        edge_reduction="mean",
    ):
        super().__init__()

        self.embed_nodes = MLP(n_in_nodes, n_hid, n_hid, layer_norm=layer_norm, act=act)
        self.embed_edges = MLP(n_in_edges, n_hid, n_hid, layer_norm=layer_norm, act=act)

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

    def forward(self, x, edge_attr, edge_index):
        B, N, C = x.shape

        x = self.embed_nodes(x)

        edge_attr = self.embed_edges(edge_attr).expand(B, -1, -1)

        for processor_layer in self.processor_layers:
            x, edge_attr = processor_layer(edge_attr, edge_index[0], edge_index[1], x)

        return x


class MassFixer(nn.Module):
    def __init__(
        self,
        mass_correction="shift",
        density_not_massmix=True,
    ):
        super().__init__()

        self.density_not_massmix = density_not_massmix

        self.mass_correction = mass_correction

    def forward(
        self,
        x_density_delta,
        x_density,
        volume,
        volume_next,
        airdensity,
        airdensity_next,
        edge_index,
    ):
        if not self.density_not_massmix:
            x_density_next = massmix_to_density(
                x_density_delta + x_density, airdensity_next, ppm=True
            )
            x_density = massmix_to_density(x_density, airdensity, ppm=True)
            x_density_delta = x_density_next - x_density

        if self.mass_correction == "scale":
            volume_weights = volume / volume.sum((1, 2), keepdim=True)
            old_mean = (x_density * volume_weights).sum((1, 2), keepdim=True)

            volume_weights = volume_next / volume_next.sum((1, 2), keepdim=True)
            x_density_hat = x_density + x_density_delta
            new_mean = (x_density_hat * volume_weights).sum((1, 2), keepdim=True)

            x_density_delta = x_density_hat * (old_mean / new_mean) - x_density

        elif self.mass_correction == "shift":
            volume_weights = volume_next / volume_next.sum((1, 2), keepdim=True)
            x_density_delta = x_density_delta - (x_density_delta * volume_weights).sum(
                (1, 2), keepdim=True
            )

        if not self.density_not_massmix:
            x_density_next = density_to_massmix(
                x_density_delta + x_density, airdensity_next, ppm=True
            )
            x_density = density_to_massmix(x_density, airdensity, ppm=True)
            x_density_delta = x_density_next - x_density

        return x_density_delta


class ProjectionHead(nn.Module):
    def __init__(
        self,
        n_hid,
        n_out,
        layer_norm=False,
        act="swish",
        readout_act="none",
    ):
        super().__init__()

        self.readout = MLP(
            n_hid,
            n_hid,
            n_out,
            layer_norm=layer_norm,
            act=act,
        )
        self.readout_act = ACTIVATIONS[readout_act]()

    def forward(self, x_feats, x_grid, x_grid_delta_scale, x_grid_delta_offset):
        x_out = self.readout_act(self.readout(x_feats))

        x_out = x_out * x_grid_delta_scale + x_grid_delta_offset

        return x_out


class GraphCastHead(nn.Module):
    def __init__(
        self,
        n_hid,
        n_targ,
        dryairmass_weighting=False,
        rescale_to_co2diff=False,
        add_surfflux_posthoc=True,
        error_correction=True,
        error_correction_kwargs=dict(mass_correction="shift"),
        layer_norm=True,
        act="swish",
        readout_act="none",
    ):
        super().__init__()

        self.readout = MLP(
            n_hid,
            n_hid,
            n_targ,
            layer_norm=layer_norm,
            act=act,
        )
        self.readout_act = ACTIVATIONS[readout_act]()

        self.dryairmass_weighting = dryairmass_weighting

        self.rescale_to_co2diff = rescale_to_co2diff

        self.add_surfflux_posthoc = add_surfflux_posthoc

        if error_correction:
            self.error_correction = MassFixer(**error_correction_kwargs)
        else:
            self.error_correction = None

    def forward(
        self,
        x_feats,
        x_grid,
        surfflux_as_densitysource_prev,
        co2density_delta_offset,
        co2density_delta_scale,
        volume,
        volume_next,
        airdensity,
        airdensity_next,
        edge_index,
    ):
        x_out = self.readout_act(self.readout(x_feats))

        if self.rescale_to_co2diff:
            x_out = x_out * co2density_delta_scale + co2density_delta_offset

        if self.error_correction and (not self.training):
            x_out = self.error_correction(
                x_out,
                x_grid,
                volume,
                volume_next,
                airdensity,
                airdensity_next,
                edge_index,
            )

        if self.add_surfflux_posthoc:
            x_out[..., :1] = x_out[..., :1] + surfflux_as_densitysource_prev

        if not torch.isfinite(x_out).all():
            print("ATTENTION!", x_out.min(), x_out.max())
        x_out = torch.nan_to_num(x_out, nan=0.0, posinf=0.0, neginf=0.0)

        return x_out


def GraphCastStep(
    x_grid, x_grid_offset, x_grid_scale, x_aux, gnn, head, gnn_inputs, head_inputs
):
    x_feats = gnn(
        torch.cat([(x_grid - x_grid_offset) / x_grid_scale, x_aux], dim=-1), *gnn_inputs
    )
    x_out = head(x_feats, x_grid, *head_inputs)

    return x_out


def Euler(x_grid, step, step_inputs):
    return x_grid + step(x_grid, *step_inputs)


def SSP_RK2(x_grid, step, step_inputs):
    output_1 = x_grid + step(x_grid, *step_inputs)
    output_2 = step(output_1, *step_inputs)
    return 0.5 * x_grid + 0.5 * output_1 + 0.5 * output_2


def SSP_RK3(x_grid, step, step_inputs):
    output_1 = x_grid + step(x_grid, *step_inputs)
    output_2 = 0.75 * x_grid + 0.25 * (output_1 + step(output_1, *step_inputs))
    output_3 = output_2 + step(output_2, *step_inputs)
    return 1 / 3 * x_grid + 2 / 3 * output_3


def RK4(x_grid, step, step_inputs):
    output_1 = step(x_grid, *step_inputs)
    output_2 = step(x_grid + 0.5 * output_1, *step_inputs)
    output_3 = step(x_grid + 0.5 * output_2, *step_inputs)
    output_4 = step(x_grid + output_3, *step_inputs)

    return x_grid + (output_1 + 2 * output_2 + 2 * output_3 + output_4) / 6


class GraphTM(nn.Module):
    def __init__(
        self,
        mesh_is_grid=False,
        mesh_min_level=2,
        mesh_max_level=4,
        resolved_locations=None,
        multimesh=True,
        hex_not_tri=True,
        gridpath=None,
        g2m_radius_fraction=0.6,
        gnn_kwargs=dict(
            n_grid=7 + 19 + 1 + 6 * 19,
            n_mesh=3,
            n_g2m=4,
            n_mm=4,
            n_m2g=4,
            n_hid=128,
            n_layers=5,
            layer_norm=True,
            act="swish",
        ),
        head_kwargs=dict(
            n_hid=128,
            n_targ=19,
            flux_scheme=False,
            grid="carboscope_2D",
            layer_norm=True,
            act="swish",
            readout_act="none",
        ),
        integrator="euler",
        meteo_vars=[
            "u",
            "v",
            "omeg",
            "q",
            "r",
            "t",
            "gp",
            "airdensity",
            "airdensity_next",
        ],
        dt=6 * 60 * 60,
        density_not_massmix=True,
        surfflux_as_input=False,
        target_vars=[],
        forcing_vars=[],
        v2=True,
    ):
        super().__init__()

        self.v2 = v2
        self.target_vars = target_vars
        self.forcing_vars = forcing_vars

        self.meteo_vars = meteo_vars
        self.dt = dt
        self.density_not_massmix = density_not_massmix
        self.surfflux_as_input = surfflux_as_input

        self.mesh_is_grid = mesh_is_grid

        if multimesh:
            edge_features = []
            edge_idxs = []

            for i in range(mesh_min_level, mesh_max_level + 1):
                grid = ICONGrid.create(
                    mesh_min_level, i, resolved_locations=resolved_locations
                )
                ds = grid.to_mesh(hex_not_tri=hex_not_tri)
                edge_features.append(ds.edge_features.values)
                edge_idxs.append(ds.edge_idxs.values)

            edge_features = np.concatenate(edge_features, axis=0)
            edge_idxs = np.concatenate(edge_idxs, axis=0)
        else:
            grid = ICONGrid.create(
                mesh_min_level, mesh_max_level, resolved_locations=resolved_locations
            )
            ds = grid.to_mesh(hex_not_tri=hex_not_tri)
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

        if self.mesh_is_grid:
            self.gnn = MultiMeshGNN(**gnn_kwargs)
        else:
            v = np.array(grid.hex_vertices)
            nodes = np.stack(latlon_to_xyz(v[:, 1], v[:, 0]), axis=-1)
            edge_distances = np.linalg.norm(
                nodes[ds.edge_idxs.values[:, 0]] - nodes[ds.edge_idxs.values[:, 1]],
                axis=-1,
            )
            max_edge_length = (
                edge_distances.max()
            )  # grid.to_mesh(hex_not_tri=hex_not_tri).edge_features.values[:, 0].max()

            ds = xr.open_zarr(gridpath)

            g2m = grid.generate_g2m_mesh(
                ds,
                hex_not_tri=hex_not_tri,
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
                torch.from_numpy(
                    g2m.edge_features.values.astype("float32")
                ).contiguous(),
            )

            m2g = grid.generate_m2g_mesh(ds, hex_not_tri=hex_not_tri)

            self.register_buffer(
                "m2g_edge_index",
                torch.from_numpy(m2g.edge_idxs.values.astype("int")).t().contiguous(),
            )
            self.register_buffer(
                "m2g_edge_attr",
                torch.from_numpy(
                    m2g.edge_features.values.astype("float32")
                ).contiguous(),
            )

            self.gnn = GraphCastGNN(**gnn_kwargs)

        if self.v2:
            self.head = ProjectionHead(**head_kwargs)
        else:
            head_kwargs["error_correction_kwargs"][
                "density_not_massmix"
            ] = density_not_massmix
            self.head = GraphCastHead(**head_kwargs)

        self.integrator = {
            "euler": Euler,
            "ssp_rk2": SSP_RK2,
            "ssp_rk3": SSP_RK3,
            "rk4": RK4,
        }[integrator]

    def forward(self, batch):
        if self.v2:
            x_grid = torch.cat([batch[v] for v in self.target_vars], dim=-1)
            x_grid_offset = torch.cat(
                [(batch[f"{v}_offset"]).expand_as(batch[v]) for v in self.target_vars],
                dim=-1,
            )
            x_grid_scale = torch.cat(
                [(batch[f"{v}_scale"]).expand_as(batch[v]) for v in self.target_vars],
                dim=-1,
            )
            x_grid_delta_offset = torch.cat(
                [
                    (batch[f"{v}_delta_offset"]).expand_as(batch[v])
                    for v in self.target_vars
                ],
                dim=-1,
            )
            x_grid_delta_scale = torch.cat(
                [
                    (batch[f"{v}_delta_scale"]).expand_as(batch[v])
                    for v in self.target_vars
                ],
                dim=-1,
            )

            B, N, _ = x_grid.shape

            x_aux = torch.cat(
                [
                    (
                        self.mm_x.expand(B, -1, -1)
                        if self.mesh_is_grid
                        else self.grid_x.expand(B, -1, -1)
                    ),
                ]
                + [
                    (batch[v] - batch[f"{v}_offset"]) / batch[f"{v}_scale"]
                    for v in self.forcing_vars
                ],
                dim=-1,
            )

            if self.mesh_is_grid:
                edge_attr = self.mm_edge_attr.expand(B, -1, -1)

                edge_index = self.mm_edge_index

                gnn_inputs = [edge_attr, edge_index]

            else:
                g2m_edge_attr = self.g2m_edge_attr.expand(B, -1, -1)
                mm_edge_attr = self.mm_edge_attr.expand(B, -1, -1)
                m2g_edge_attr = self.m2g_edge_attr.expand(B, -1, -1)

                g2m_edge_index = self.g2m_edge_index
                mm_edge_index = self.mm_edge_index
                m2g_edge_index = self.m2g_edge_index

                x_mesh = self.mm_x.expand(B, -1, -1)

                gnn_inputs = [
                    x_mesh,
                    g2m_edge_attr,
                    mm_edge_attr,
                    m2g_edge_attr,
                    g2m_edge_index,
                    mm_edge_index,
                    m2g_edge_index,
                ]

            head_inputs = [
                x_grid_delta_offset,
                x_grid_delta_scale,
            ]

            step_inputs = [
                x_grid_offset,
                x_grid_scale,
                x_aux,
                self.gnn,
                self.head,
                gnn_inputs,
                head_inputs,
            ]

            x_grid = self.integrator(x_grid, GraphCastStep, step_inputs)

            preds = {}
            i = 0
            for v in self.target_vars:
                C = batch[v].shape[-1]
                preds[v] = x_grid[..., i : i + C]
                i += C

            for molecule in ["co2", "ch4"]:
                if (
                    f"{molecule}density" in preds
                    and (f"{molecule}massmix" not in preds)
                    # v.endswith("density")
                    # and v != "airdensity"
                    # and v.replace("density", "massmix") not in self.target_vars
                ):
                    preds[f"{molecule}massmix"] = density_to_massmix(
                        preds[f"{molecule}density"],
                        (
                            batch["airdensity_next"]
                            if not "airdensity" in self.target_vars
                            else preds["airdensity"]
                        ),
                        ppm=True,
                    )
                if (
                    f"{molecule}massmix" in preds
                    and (f"{molecule}density" not in preds)
                    # v.endswith("massmix")
                    # and v.replace("massmix", "density") not in self.target_vars
                ):
                    preds[f"{molecule}density"] = massmix_to_density(
                        preds[f"{molecule}massmix"],
                        (
                            batch["airdensity_next"]
                            if not "airdensity" in self.target_vars
                            else preds["airdensity"]
                        ),
                        ppm=True,
                    )

            return preds

        else:
            density_prev = batch["co2density"]  # b n h
            massmix_prev = batch["co2massmix"]  # b n h

            surfflux_as_densitysource_prev = mass_to_density(
                (batch["co2flux_land"] + batch["co2flux_ocean"] + batch["co2flux_subt"])
                * batch["cell_area"]
                * self.dt,
                batch["volume"][..., :1],
            )  # b n h

            if not self.density_not_massmix:
                # conc_t = density_to_massmix(conc_t, batch["airdensity"], ppm = True)
                surfflux_as_densitysource_prev = density_to_massmix(
                    surfflux_as_densitysource_prev,
                    batch["airdensity"][..., :1],
                    ppm=True,
                )

            meteo_norm_prev = torch.stack(
                [
                    (batch[v] - batch[f"{v}_offset"]) / batch[f"{v}_scale"]
                    for v in self.meteo_vars
                ],
                dim=-1,
            )  # b n h c

            if not torch.isfinite(density_prev).all():
                print("ATTENTION! density_prev", density_prev.min(), density_prev.max())
            if not torch.isfinite(massmix_prev).all():
                print("ATTENTION! massmix_prev", massmix_prev.min(), massmix_prev.max())
            if not torch.isfinite(surfflux_as_densitysource_prev).all():
                print(
                    "ATTENTION! surfflux_as_densitysource_prev",
                    surfflux_as_densitysource_prev.min(),
                    surfflux_as_densitysource_prev.max(),
                )
            if not torch.isfinite(meteo_norm_prev).all():
                print(
                    "ATTENTION! meteo_norm_prev",
                    meteo_norm_prev.min(),
                    meteo_norm_prev.max(),
                )

            B, n_cells, n_lev, C = meteo_norm_prev.shape

            x_grid = density_prev if self.density_not_massmix else massmix_prev
            x_aux = torch.cat(
                [
                    (
                        self.mm_x.expand(B, -1, -1)
                        if self.mesh_is_grid
                        else self.grid_x.expand(B, -1, -1)
                    ),
                    meteo_norm_prev.reshape(B, n_cells, n_lev * C),
                ],
                dim=2,
            )

            if self.surfflux_as_input:
                x_aux = torch.cat(
                    [x_aux]
                    + [
                        (batch[v] - batch[f"{v}_offset"]) / batch[f"{v}_scale"]
                        for v in ["co2flux_land", "co2flux_ocean", "co2flux_subt"]
                    ],
                    dim=-1,
                )

            if self.mesh_is_grid:
                edge_attr = self.mm_edge_attr.expand(B, -1, -1)

                edge_index = self.mm_edge_index

                gnn_inputs = [edge_attr, edge_index]

            else:
                g2m_edge_attr = self.g2m_edge_attr.expand(B, -1, -1)
                mm_edge_attr = self.mm_edge_attr.expand(B, -1, -1)
                m2g_edge_attr = self.m2g_edge_attr.expand(B, -1, -1)

                g2m_edge_index = self.g2m_edge_index
                mm_edge_index = self.mm_edge_index
                m2g_edge_index = self.m2g_edge_index

                x_mesh = self.mm_x.expand(B, -1, -1)

                gnn_inputs = [
                    x_mesh,
                    g2m_edge_attr,
                    mm_edge_attr,
                    m2g_edge_attr,
                    g2m_edge_index,
                    mm_edge_index,
                    m2g_edge_index,
                ]

            head_inputs = [
                surfflux_as_densitysource_prev,
                (
                    batch["co2density_delta_offset"]
                    if self.density_not_massmix
                    else batch["co2massmix_delta_offset"]
                ),
                (
                    batch["co2density_delta_scale"]
                    if self.density_not_massmix
                    else batch["co2massmix_delta_scale"]
                ),
                batch["volume"],
                batch["volume_next"],
                batch["airdensity"],
                batch["airdensity_next"],
                mm_edge_index,
            ]

            step_inputs = [
                (
                    batch["co2density_offset"]
                    if self.density_not_massmix
                    else batch["co2massmix_offset"]
                ),
                (
                    batch["co2density_scale"]
                    if self.density_not_massmix
                    else batch["co2massmix_scale"]
                ),
                x_aux,
                self.gnn,
                self.head,
                gnn_inputs,
                head_inputs,
            ]

            x_grid = self.integrator(x_grid, GraphCastStep, step_inputs)

            if self.density_not_massmix:
                density_pred = x_grid
                massmix_pred = density_to_massmix(
                    density_pred, batch["airdensity_next"], ppm=True
                )
            else:
                massmix_pred = x_grid
                density_pred = massmix_to_density(
                    massmix_pred, batch["airdensity_next"], ppm=True
                )

            return density_pred, massmix_pred
