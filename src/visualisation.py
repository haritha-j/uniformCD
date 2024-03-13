# viusalize elements / relationships
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from OCC.Core.gp import gp_Pnt
from utils.JupyterIFCRenderer import JupyterIFCRenderer
import plotly.graph_objects as go
from chamferdist import ChamferDistance
from pythreejs import (
    LineSegments2,
    LineSegmentsGeometry,
    LineMaterial,
)

from src.utils import *


# visualize ifc model and point cloud simultaneously
def vis_ifc_and_cloud(ifc, clouds):
    viewer = JupyterIFCRenderer(ifc, size=(400, 300))
    colours = ["#ff7070", "#70ff70", "#7070ff"]
    for i, cloud in enumerate(clouds):
        if cloud is not None:
            gp_pnt_list = [gp_Pnt(k[0], k[1], k[2]) for k in cloud]
            # print("no points:", len(gp_pnt_list))
            col = i if i < len(colours) else 0
            viewer.DisplayShape(gp_pnt_list, vertex_color=colours[col])
    return viewer


# covert rgb to hex value
def rgb_to_hex(r, g, b):
    # Ensure that the RGB values are within the valid range (0-255)
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    # Convert RGB values to a hexadecimal color code
    hex_color = "#{:02X}{:02X}{:02X}".format(r, g, b)
    return hex_color


def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames = []

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(
            dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze)))))
        )
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1,
                    x=0.8,
                    xanchor="left",
                    yanchor="bottom",
                    pad=dict(t=45, r=10),
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=50, redraw=True),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        )
                    ],
                )
            ]
        ),
        frames=frames,
    )

    return fig


def pcshow(xs, ys, zs):
    data = [go.Scatter3d(x=xs, y=ys, z=zs, mode="markers")]
    fig = visualize_rotate(data)
    fig.update_traces(
        marker=dict(size=2, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    fig.show()


# add cloud to visualisation
def add_cloud(v, cloud, colour="#70ff70"):
    gp_pnt_list = [gp_Pnt(k[0], k[1], k[2]) for k in cloud]
    v.DisplayShape(gp_pnt_list, vertex_color=colour)


# draw lines connecting src and tgt points
def add_lines(v, src, tgt, pairs=None, k=1):
    lines = []
    if pairs is None: pairs = [i for i in range(len(src))]
    if k==1: pairs = [pairs]

    for j in range(k):
        positions = [[src[i], tgt[pairs[j][i]]] for i in range(len(tgt))]
        print(len(positions))
        lines.append(
            LineSegments2(
                LineSegmentsGeometry(positions=positions),
                LineMaterial(linewidth=2, color="green"),
            )
        )

    v._displayed_non_pickable_objects.add(lines)


# draw lines connecting src and tgt points
# strength is used to determine colour intensity.
# this function is much slower due to need to generate a new line segment for each pair
def add_lines_colour(v, src, tgt, pairs=None, k=1, strength=None):
    lines = []
    if pairs is None: pairs = [i for i in range(len(src))]
    if k==1: pairs = [pairs]
    if strength is None: strength = [1.0 for i in range(len(pairs[0]))]

    print(len(strength))
    colour_intensity = [int(255 * s) for s in strength]
    colour = [rgb_to_hex(100, c, 0) for c in colour_intensity]
    for i in range(len(tgt)):
        positions = [[src[i], tgt[pairs[j][i]]] for j in range(k)]

        lines.append(
            LineSegments2(
                LineSegmentsGeometry(positions=positions),
                LineMaterial(linewidth=1, color=colour[i]),
            )
        )

    v._displayed_non_pickable_objects.add(lines)


# visually show matching points
# pairs is a list of indices of the matching points in tgt cloud to src cloud
def visualise_matching_points(src_cld, tgt_cld, blueprint, pairs=None, strength=None, k=1, same_cloud=False):

    # generate visualiser with blank ifc
    ifc = setup_ifc_file(blueprint)
    v = JupyterIFCRenderer(ifc, size=(700, 550))
    
    add_cloud(v, src_cld.astype(np.float64), colour="#ff7070")
    add_cloud(v, tgt_cld.astype(np.float64), colour="#7070ff")
    
    if same_cloud:
        tgt_cld = src_cld

    # add elements to visualiser
    if strength is None:
        add_lines(v, src_cld, tgt_cld, pairs=pairs)
    else:
        add_lines_colour(v, src_cld, tgt_cld, pairs=pairs, strength=strength, k=k)

    #print(src_cld.shape, type(src_cld[0][0]), tgt_cld.shape, type(tgt_cld))
    return v


# generate visualisation of loss between src and tgt clouds
def visualise_loss(src_cld, tgt_cld, blueprint, loss="chamfer", strength=None, k=1, pairs=None, same_cloud=False):

    # calculate loss
    cuda = torch.device("cuda")
    target_pcd_tensor = torch.tensor([tgt_cld], device=cuda)
    src_pcd_tensor = torch.tensor([src_cld], device=cuda)

    # if loss is not chamfer, then use the pairs provided
    if loss == "chamfer":
        print("SG", src_pcd_tensor.shape, target_pcd_tensor.shape)
        chamferDist = ChamferDistance()
        nn = chamferDist(
            src_pcd_tensor, target_pcd_tensor, bidirectional=False, return_nn=True, k=k
        )

        if k == 1:
            pairs = torch.flatten(nn[0].idx[0].detach().cpu()).numpy()
            print("unique", len(np.unique(pairs)))
            #print("pairs", pairs)
        else:
            print("int", nn[0].idx[0][:,0].detach().cpu().numpy().shape)
            pairs = [nn[0].idx[0][:,i].detach().cpu().numpy() for i in range(k)]

    return visualise_matching_points(src_cld, tgt_cld, blueprint, pairs=pairs, strength=strength, k=k, same_cloud=same_cloud)


# produce a colour map based on the density of a point cloud
# parallelised version
def visualise_density(clouds, colormap_name='plasma'):
    # compute nearest neighbours to calculated density
    clouds = torch.tensor(clouds, device="cuda")
    chamferDist = ChamferDistance()
    nn = chamferDist(clouds, clouds, bidirectional=False, return_nn=True, k=32)
    
    # measure normalised density
    density = torch.mean(nn[0].dists[:,:,1:], dim=2)
    eps = 0.00001
    density = 1 / (density + eps)
    high, low = torch.max(density), torch.min(density)
    diff = high - low
    density = (density - low) / diff
    density = density.detach().cpu().numpy()
    
    # map colour
    colours = np.zeros((density.shape[0], density.shape[1], 4))
    colormap = plt.get_cmap(colormap_name)
    for i, cloud in enumerate(density):
        for j, pt in enumerate(cloud):
            colours[i,j] = colormap(pt)

    return colours


# general function to plot multiple sets of values
def plot_dists(ax, losses, labels, title, xlabel="point cloud index", ylabel="distance (log)", 
               log=True, limit=None, legend=True):
    if limit == None:
        limit = len(losses[0])
    x = np.arange(0, limit*10, 10)

    for i, loss in enumerate(losses):
        if log:
            ax.plot(x, np.log(loss[:limit]), label=labels[i])
        else:
            ax.plot(x, loss[:limit], label=labels[i])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        ax.legend()
        
        
# visualise a list of point clouds as an animation using open3d
# use ctrl+c to copy and ctrl+v to set camera and zoom inside visualiser
def create_point_cloud_animation(cloud_list, loss_func, save_image=False, colours=None):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    cloud = cloud_list[0]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(cloud)
    if colours is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colours[0])
    vis.add_geometry(point_cloud)
    stops = [9,39,99,299,999]

    for i in range(len(cloud_list)):
        time.sleep(0.01 + 0.05/(i/10+1))
        cloud = cloud_list[i]
        point_cloud.points = o3d.utility.Vector3dVector(cloud)
        if colours is not None:
            point_cloud.colors = o3d.utility.Vector3dVector(colours[i])
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()
        if save_image and i in stops:
            vis.capture_screen_image("data/" + loss_func + str(i) + ".jpg", do_render=True)
    vis.destroy_window()

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)


# produce a colour map based on the weights of a point cloud
def visualise_point_loss(weights, high, low, colormap_name='plasma'):
    # normalise weights
    diff = high - low
    weights = (weights - low) / diff
    
    # map colour
    colours = np.zeros((len(weights), 4))
    colormap = plt.get_cmap(colormap_name)
    for j, pt in enumerate(weights):
        colours[j] = colormap(pt)

    return colours[:,:3]