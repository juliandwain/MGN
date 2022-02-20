import os
import pathlib
import pandas as pd
import numpy as np
import pickle
from glob import glob
import enum
import json

from numpy import concatenate

from torch import from_numpy, no_grad

from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import (
    RadiusGraph,
    Cartesian,
    Distance,
    Compose,
    KNNGraph,
    Delaunay,
    ToUndirected,
)
from torch_geometric.utils import to_networkx
import torch


from networkx import is_weakly_connected
from warnings import warn

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from ..utils.mgn_utils import process_node_window, get_sample


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


def squeeze_data_frame(data_frame):
    for k, v in data_frame.items():
        data_frame[k] = torch.squeeze(v, 0)
    return data_frame


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_targets(params):
    """Adds target and optionally history fields to dataframe."""
    fields = params["field"]
    add_history = params["history"]
    loss_type = params["loss_type"]

    def fn(trajectory):
        if loss_type == "deform":
            out = {}
            for key, val in trajectory.items():
                out[key] = val[0:-1]
                if key in fields:
                    out["target|" + key] = val[1:]
                if key == "stress":
                    out["target|stress"] = val[1:]
            return out
        elif loss_type == "cloth":
            out = {}
            for key, val in trajectory.items():
                out[key] = val[1:-1]
                if key in fields:
                    if add_history:
                        out["prev|" + key] = val[0:-2]
                    out["target|" + key] = val[2:]
            return out
        else:
            out = {}
            for key, val in trajectory.items():
                out[key] = val[1:-1]
                if key in fields:
                    if add_history:
                        out["prev|" + key] = val[0:-2]
                    out["target|" + key] = val[2:]
            return out

    return fn


def split_and_preprocess(params, model_type=None):
    """Splits trajectories into frames, and adds training noise."""
    noise_field = params["field"]
    noise_scale = params["noise"]
    noise_gamma = params["gamma"]

    def add_noise(frame):
        zero_size = torch.zeros(
            frame[noise_field].size(), dtype=torch.float32
        ).to(device)
        noise = torch.normal(zero_size, std=noise_scale).to(device)
        other = torch.Tensor([NodeType.NORMAL.value]).to(device)
        mask = torch.eq(frame["node_type"], other.int())[:, 0]
        mask_sequence = []
        for i in range(noise.shape[1]):
            mask_sequence.append(mask)
        mask = torch.stack(mask_sequence, dim=1)
        noise = torch.where(mask, noise, torch.zeros_like(noise))
        frame[noise_field] += noise
        frame["target|" + noise_field] += (1.0 - noise_gamma) * noise
        return frame

    def element_operation(trajectory):
        trajectory_steps = []
        for i in range(steps):
            trajectory_step = {}
            for key, value in trajectory.items():
                trajectory_step[key] = value[i]
            noisy_trajectory_step = add_noise(trajectory_step)
            trajectory_steps.append(noisy_trajectory_step)
        return trajectory_steps

    return element_operation


loaded_meta = False


def process_trajectory(
    trajectory_data,
    params,
    dataset_dir,
    add_targets_bool=False,
    split_and_preprocess_bool=False,
):
    global loaded_meta
    global shapes
    global dtypes
    global types
    global steps

    if not loaded_meta:
        try:
            with open(os.path.join(dataset_dir, "meta.json"), "r") as fp:
                meta = json.loads(fp.read())
            shapes = {}
            dtypes = {}
            types = {}
            if params["loss_type"] == "cloth":
                steps = meta["trajectory_length"] - 2
            elif params["loss_type"] == "deform":
                steps = meta["trajectory_length"] - 1
            else:
                steps = meta["trajectory_length"] - 2
            for key, field in meta["features"].items():
                shapes[key] = field["shape"]
                dtypes[key] = field["dtype"]
                types[key] = field["type"]
        except FileNotFoundError as e:
            print(e)
            quit()
    trajectory = {}
    # decode bytes into corresponding dtypes
    for key, value in trajectory_data.items():
        raw_data = value.numpy().tobytes()
        mature_data = np.frombuffer(raw_data, dtype=getattr(np, dtypes[key]))
        mature_data = torch.from_numpy(mature_data).to(device)
        reshaped_data = torch.reshape(mature_data, shapes[key])
        if types[key] == "static":
            reshaped_data = torch.tile(
                reshaped_data, (meta["trajectory_length"], 1, 1)
            )
        elif types[key] == "dynamic_varlen":
            pass
        elif types[key] != "dynamic":
            raise ValueError("invalid data format")
        trajectory[key] = reshaped_data

    if add_targets_bool:
        trajectory = add_targets(params)(trajectory)
    if split_and_preprocess_bool:
        trajectory = split_and_preprocess(params)(trajectory)
    _trajectory = {}
    for field in trajectory[0].keys():
        _trajectory[field] = (
            torch.cat([d[field][None] for d in trajectory], dim=0)
            .cpu()
            .numpy()
        )
    return _trajectory


def get_comsol_data(fn="/data/ccsi/cylinder_flow/cylinder_flow_comsol.csv"):

    """
    Preprocesses COMSOL cylinder flow simulation output.

    """

    D = pd.read_csv(fn)
    x = D["x"]
    y = D["y"]
    D = D.drop(columns=["x", "y"])

    X = D.values

    inds = np.arange(0, X.shape[1], 4)
    times = X[:, inds]
    t = times[0]

    inds = np.arange(1, X.shape[1], 4)
    vel_x = X[:, inds]

    inds = np.arange(2, X.shape[1], 4)
    vel_y = X[:, inds]

    inds = np.arange(3, X.shape[1], 4)
    p = X[:, inds]

    return x, y, t, vel_x, vel_y, p


def get_comsol_edges(
    node_coordinates,
    mesh_file="/data/ccsi/cylinder_flow/mesh_comsol_output.txt",
):

    """
    Preprocesses COMSOL cylinder flow mesh

    This function is necessary because the node coordinates and comsol mesh are in a different order
    Need to re-order the edge list from the mesh file
    """

    def splitFloatLine(line):
        return list(map(float, line.split()[:2]))

    def splitElementLine(line):
        return list(map(int, line.split()[:3]))

    def simplexToEdgeList(simp):
        edges = [(simp[0], simp[1]), (simp[1], simp[2]), (simp[2], simp[0])]
        r_edges = [(e[1], e[0]) for e in edges]
        return edges + r_edges

    with open(mesh_file) as fid:
        mesh = fid.readlines()

    # get nodes
    nodeLine = mesh[4]
    numNodes = int(nodeLine.split()[2])
    mesh_nodes = mesh[10 : (10 + numNodes)]
    mesh_nodes = np.array(list(map(splitFloatLine, mesh_nodes)))

    # get mesh elements
    mesh_elements = mesh[11 + numNodes :]
    mesh_elements = np.array(list(map(splitElementLine, mesh_elements)))
    mesh_elements = mesh_elements - 1  # comsol starts from 1 not 0.

    # match mesh and node coordinates
    Y = cdist(mesh_nodes, node_coordinates)
    index = np.argmin(Y, axis=1)
    simplex = index[mesh_elements]

    A = list(map(simplexToEdgeList, simplex))
    edge_list = [b for sublist in A for b in sublist]
    edge_list = np.unique(edge_list, axis=1)

    return edge_list


def simplexToEdgeList(simp):
    edges = [(simp[0], simp[1]), (simp[1], simp[2]), (simp[2], simp[0])]
    r_edges = [(e[1], e[0]) for e in edges]
    return edges + r_edges


class CylinderFlowDatasetNP(Dataset):

    # drop pressure prediction from CylinderFlowDataset2

    """
    CylinderFlow dataset (using one simulation)
    Pressure is not included as an input and output

    """

    def __init__(
        self,
        dataset_dir="/home/julian/ma/MGN/cylinder_flow_comsol.csv",
        center=[0.2, 0.2],
        R=0.05,
        output_type="velocity",  # acceleration, velocity, state
        window_length=1,
        noise=None,
        noise_gamma=0.1,
        apply_onehot=False,
        boundary_nodes=[
            1,
            3,
        ],  # list of integer node types corresponding to boundaries
        source_nodes=[
            2
        ],  # list of integer node types corresponding to sources
        normalize=True,
        **kwargs
    ):

        self.output_type = output_type

        assert (output_type != "acceleration") or (window_length >= 2)
        self.window_length = window_length

        data_dir = pathlib.Path("./data")
        fname = data_dir / "train_loader.pkl"
        with fname.open(mode="rb") as f:
            trajectory = pickle.load(f)
        params = {
            "loss_type": "cfd",
            "field": "velocity",
            "noise": 0.02,
            "gamma": 0.1,
            "history": False,
        }
        trajectory = process_trajectory(
            trajectory,
            params,
            dataset_dir,
            add_targets_bool=True,
            split_and_preprocess_bool=True,
        )
        data = trajectory["velocity"]

        # normalize data;
        # for larger datasets, replace with online normalization
        if normalize:
            self.mean = np.mean(data, axis=(0, 1))
            self.std = np.std(data, axis=(0, 1))
            data = (data - self.mean) / self.std

        # Find the boundary nodes
        # node_coordinates = np.vstack([x, y]).T
        node_coordinates = trajectory["mesh_pos"][0]
        x = node_coordinates[:, 0]
        mn, mx = np.min(node_coordinates, axis=0), np.max(
            node_coordinates, axis=0
        )
        source_inds = np.where(node_coordinates[:, 0] == mn[0])[0]
        bottom_inds = np.where(node_coordinates[:, 1] == mn[1])[0]
        top_inds = np.where(node_coordinates[:, 1] == mx[1])[0]
        right_inds = np.where(node_coordinates[:, 0] == mx[0])[0]
        non_source_boundary_inds = (
            set(bottom_inds)
            .union(set(right_inds))
            .union(set(top_inds))
            .difference(source_inds)
        )

        # cylinder
        center = np.array(center).reshape(1, 2)
        distFromCircleCenter = cdist(node_coordinates, center)

        interior_boundary_inds = np.where(distFromCircleCenter <= R)[0]
        boundary_inds = sorted(
            list(non_source_boundary_inds.union(interior_boundary_inds))
        )

        # save data, node types
        self.data = data
        self.node_types = np.zeros((len(x), 1), dtype="int")
        self.node_types[boundary_inds] = boundary_nodes[
            0
        ]  # top, bottom, interior
        self.node_types[right_inds] = boundary_nodes[
            1
        ]  # right #another 'source'?
        self.node_types[source_inds] = source_nodes[0]

        # indices of boundary/source nodes for this class, since there is only one simulation
        self.boundary_nodes = boundary_inds
        self.source_nodes = source_inds

        # one-hot
        apply_onehot_ = (
            np.min(self.node_types) >= 0
        )  # check if one-hot encoding can be applied
        onehot_dim = -1 if not apply_onehot_ else (np.max(self.node_types) + 1)

        if apply_onehot and not apply_onehot_:
            raise Exception(filename + ": cannot apply one-hot encoding")

        self.onehot_dim = onehot_dim
        self.apply_onehot = apply_onehot

        # graph construction
        transforms = [
            Cartesian(norm=False, cat=True),
            Distance(norm=False, cat=True),
        ]

        # edge_list = get_comsol_edges(node_coordinates, mesh_file)
        y = cdist(node_coordinates, node_coordinates)
        index = np.argmin(y, axis=1)
        simplex = index[[trajectory["cells"][0]]]

        A = list(map(simplexToEdgeList, simplex))
        edge_list = [b for sublist in A for b in sublist]
        edge_list = np.unique(edge_list, axis=1)

        graph = Data(
            pos=from_numpy(node_coordinates.astype(np.float32)),
            edge_index=from_numpy(edge_list.T),
        )

        transforms = Compose(transforms)
        graph = transforms(graph)

        # #remove other-to-source edges #keep?
        # sources = np.where(np.isin(self.node_types[-1].flatten(), source_nodes))[0]
        # drop_edges = np.isin(graph.edge_index[1].numpy(), sources)
        # graph.edge_index = graph.edge_index[:, ~drop_edges].contiguous()
        # graph.edge_attr = graph.edge_attr[~drop_edges].contiguous()

        if not is_weakly_connected(to_networkx(graph)):
            warn(filename + ": disconnected graph")

        self.graph = graph

        self.dataset_length = np.array(
            self.data.shape[0] - self.window_length, dtype=np.int64
        )
        self.output_dim = self.data.shape[-1]

        self.noise = noise  # to do: check none or length==output_dim
        self.noise_gamma = noise_gamma
        self.faces = trajectory["cells"]

        # update function; update momentum based on predicted change ('velocity'), predict pressure
        def update_function(
            mgn_output_np,
            output_type,
            current_state=None,
            previous_state=None,
            source_data=None,
        ):

            num_states = current_state.shape[-1]

            with no_grad():
                if output_type == "acceleration":
                    assert current_state is not None
                    assert previous_state is not None
                    next_state = (
                        2 * current_state - previous_state + mgn_output_np
                    )
                elif output_type == "velocity":
                    assert current_state is not None
                    next_state = current_state + mgn_output_np
                else:  # state
                    next_state = mgn_output_np.copy()

                if type(source_data) is dict:
                    for key in source_data:
                        next_state[key, :num_states] = source_data[key]
                elif type(source_data) is tuple:
                    next_state[source_data[0], :num_states] = source_data[1]
                # else: warning?

            return next_state

        self.update_function = update_function

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):

        node_data, outputs = get_sample(
            self.data,
            self.source_nodes,
            idx,
            self.window_length,
            self.output_type,
            self.noise,
            self.noise_gamma,
        )

        node_data = from_numpy(
            process_node_window(
                node_data,
                self.graph.pos.numpy(),
                self.node_types,
                self.apply_onehot,
                self.onehot_dim,
            ).astype(np.float32)
        )

        outputs = from_numpy(outputs.astype(np.float32))

        graph = Data(
            x=node_data,
            edge_index=self.graph.edge_index,
            edge_attr=self.graph.edge_attr,
            y=outputs,
            num_nodes=len(node_data),
        )

        return graph
