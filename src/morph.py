import numpy as np
import os.path
import torch
import pickle
from tqdm.notebook import tqdm
from chamferdist import ChamferDistance
import open3d as o3d
import gc

from src.chamfer import *


# morph a sphere into the shape of an input point cloud
# by optimising chamfer loss iteratively
# total points = num_points**2
def morph_sphere(src_pcd_tensor, num_points, iterations, learning_rate, stops=[],
                 loss_func= "chamfer", measure_consistency=True, sphere=True, return_assignment=True):
    
    cuda = torch.device("cuda")
    if sphere:
        # gnerate sphere
        # Generate spherical coordinates
        theta = np.linspace(0, 2 * np.pi, num_points)
        phi = np.linspace(0, np.pi, num_points)

        # Create a meshgrid from spherical coordinates
        theta, phi = np.meshgrid(theta, phi)

        # Convert spherical coordinates to Cartesian coordinates
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        # Stack the coordinates to form a 3D point cloud and reshape to (num_points * num_points, 3)
        sphere_points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        sphere_points = np.array([sphere_points for i in range(len(src_pcd_tensor))])
        sphere_points = torch.tensor(sphere_points, device=cuda, 
                                     requires_grad=True)
    else:
        sphere_points = torch.rand(1, num_points**2, 3, device=cuda, 
                                   dtype=torch.double, requires_grad=True)
    
    # optimise
    optimizer = torch.optim.Adam([sphere_points], lr=learning_rate)
    intermediate, losses, assingments = [], [], []
    chamferDist = ChamferDistance()
    assignments = []

    for i in tqdm(range(iterations)):
        optimizer.zero_grad()
        
        if loss_func == "chamfer":
            nn = chamferDist(
                src_pcd_tensor, sphere_points, bidirectional=True, return_nn=True)
            loss = torch.sum(nn[1].dists) + torch.sum(nn[0].dists)
            assignment = [nn[0].idx[:,:,0].detach().cpu().numpy(), nn[1].idx[:,:,0].detach().cpu().numpy()]
        elif loss_func == "emd":
            loss, assignment = calc_emd(sphere_points, src_pcd_tensor, 0.05, 50)
            assignment = assignment.detach().cpu().numpy()
        elif loss_func == "direct":
            loss = torch.sum(torch.square(sphere_points -src_pcd_tensor))
            assignment = None
        elif loss_func == "pair":
            loss, assignment = get_pair_loss_clouds_tensor(src_pcd_tensor, sphere_points, add_pair_loss=True, it=i)
        elif loss_func == "jittery":
            loss = get_jittery_cd_tensor(src_pcd_tensor, sphere_points, k=1, it=i)
        elif loss_func == "self":
            loss = get_self_cd_tensor(src_pcd_tensor, sphere_points)
        elif loss_func == "reverse":
            loss, assignment = calc_reverse_weighted_cd_tensor(src_pcd_tensor, sphere_points, return_assignment=True, k=32)
        elif loss_func == "prob":
            loss, assignment = calc_pairing_probabilty_loss_tensor(src_pcd_tensor, sphere_points, k=64)
        elif loss_func == "uniform":
            loss, assignment = calc_uniform_chamfer_loss_tensor(src_pcd_tensor, sphere_points, return_assignment=True, k=32)
        elif loss_func == "single":
            loss, assignment = calc_uniform_single_chamfer_loss_tensor(src_pcd_tensor, sphere_points, return_assignment=True, k=32)
        elif loss_func == "infocd":
            loss, assignment = calc_cd_like_InfoV2(src_pcd_tensor, sphere_points, return_assignment=True)
        elif loss_func == "density":
            loss = calc_relative_density_loss_tensor(src_pcd_tensor, sphere_points, return_assignment=False)
        else:
            print("unspecified loss")
            
        #print("a", assignment[0].shape)
        loss.backward()
        optimizer.step()
        #print("iteration", i, "loss", loss.item())
        
        if i in stops:
            intermediate.append(sphere_points.clone())
            losses.append(loss.item())
            if measure_consistency:
                assignments.append(assignment)
            
    # calculate final chamfer loss
    dist = chamferDist(
                src_pcd_tensor, sphere_points, bidirectional=True)
    emd_loss, _ = calc_emd(sphere_points, src_pcd_tensor, 0.05, 50)
    print("final chamfer dist", dist.item(), "emd", emd_loss.item())
    
    # save assignments for analysis
#     if measure_consistency:
#         with open("data/assignments_" + loss_func + ".pkl", "wb") as f:
#             pickle.dump(assingments, f)
            
    intermediate = torch.stack(intermediate)
    if return_assignment:
        assignments = assignments
        return intermediate, losses, assignments
    return intermediate, losses


def run_morph(cld1_name, loss_func):
    cuda = torch.device("cuda")
    cld1 = np.array(o3d.io.read_point_cloud(cld1_name).points)
    src_pcd_tensor = torch.tensor([cld1], device=cuda)

    iterations = 1000
    #stops = [0, 10, 50, 100, 150, 500, 999]
    stops = [i for i in range(0,iterations,2)]

#     morphed, losses = morph_sphere(src_pcd_tensor, 64, iterations, 0.01, stops, loss_func=loss_func, 
#                                    return_assignment=False)
    morphed, losses = morph_sphere(src_pcd_tensor, 64, iterations, 0.01, stops, measure_consistency=False, 
                                   loss_func=loss_func, return_assignment=False)
    morphed = torch.flatten(morphed, start_dim=1, end_dim=2)
    morphed = morphed.cpu().detach().numpy()
    #print(morphed.shape)

    # save frames
    with open("data/" + loss_func + ".pkl", "wb") as f:
        pickle.dump(morphed, f)

    with open("data/loss_" + loss_func + ".pkl", "wb") as f:
        pickle.dump(losses, f)

    # Save the PointCloud to a PCD file
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(morphed[-1])
    o3d.io.write_point_cloud("data/sphere_" + loss_func + ".pcd", point_cloud)
    

# batch sphere optimisation (for metrics)

def sphere_morph_metrics(loss_func, shapenet_path, save=True):
    # load shapenet test dataset
    folders = os.listdir(shapenet_path)
    print(loss_func)
    cuda = torch.device("cuda")
    chamferDist = ChamferDistance()

    iterations = 1001
    stops = [i for i in range(0,iterations,10)]
    chamfer_results = np.zeros(len(stops))
    emd_results = np.zeros(len(stops))
    count = 0
    assignments_folders = []
    
    for fl in folders:
        files = os.listdir(shapenet_path + fl)
        clouds = []
        for cl in files:
            clouds.append(np.array(o3d.io.read_point_cloud(shapenet_path + fl + "/" + cl).points))
        clouds = np.array(clouds)
        clouds = torch.tensor(clouds, device=cuda)
        count += len(clouds)
        #print(clouds.shape)

        # optimise spheres and gather intermediate clouds
        
        morphed, losses, assignments = morph_sphere(clouds, 64, iterations, 0.01, stops, loss_func=loss_func, 
                                       return_assignment=True)
        assignments_folders.append(assignments)

        #calculate chamfer and EMD
        print(morphed.shape)

        # loop through stops
        for i, mr in enumerate(tqdm(morphed)):
            nn = chamferDist(clouds, mr, bidirectional=True, return_nn=True)
            cd_loss = torch.sum(nn[1].dists) + torch.sum(nn[0].dists)
            #cd_assignment = [nn[0].idx[0,:,0].detach().cpu().numpy(), nn[1].idx[0,:,0].detach().cpu().numpy()]
            emd_loss, _ = calc_emd(clouds, mr, 0.05, 50)
            #emd_assignment = emd_assignment.detach().cpu().numpy()

            #print(cd_loss.item(), emd_loss.item())
#             cd_assignments_cat.append(cd_assignment)
#             emd_assignments_cat.append(emd_assignment)
            chamfer_results[i] += cd_loss
            emd_results[i] += emd_loss

        torch.cuda.empty_cache()
        gc.collect()
    
    print(count, chamfer_results[0])
    chamfer_results = chamfer_results/count
    emd_results = emd_results/count

    if save:
        # save results
        with open("data/" + loss_func + "_metrics.pkl", "wb") as f:
            pickle.dump([chamfer_results, emd_results, assignments_folders], f)
    else:
        return chamfer_results, emd_results
