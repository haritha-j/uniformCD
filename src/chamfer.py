
import torch
import numpy as np
from chamferdist import ChamferDistance
import torch.nn.functional as F

from utils.EMD import emd_module as emd


# calculate approximate earth mover's distance
# NOTE: gradient is only calculated for output, not gt
def calc_emd(output, gt, eps=0.005, iterations=50):
    emd_loss = emd.emdModule()
    dist, assignment = emd_loss(output, gt, eps, iterations)
    emd_out = torch.sum(dist)/len(output)*2
    return emd_out, assignment


# compute direction vectors of k neighbours
def knn_vectors(src_pcd_tensor, target_pcd_tensor, k):
    cuda = torch.device("cuda")

    # get nn
    chamferDist = ChamferDistance()
    nn = chamferDist(
        src_pcd_tensor, target_pcd_tensor, bidirectional=False, return_nn=True,
    k=k)
    nn = nn[0] # islolate forward direction

    # calculate directions
    N = src_pcd_tensor.shape[0] # batch size
    P1 = src_pcd_tensor.shape[1] # n_points in src cloud
    vectors = torch.zeros((N, P1, k, 3), device=cuda)

    # iterate through neighbours
    for i in range(k):
        diff = src_pcd_tensor - nn.knn[:,:,i,:]

        # normalise
        denom = torch.sqrt(torch.sum(torch.square(diff), 2))
        denom = denom.unsqueeze(2)
        vectors[:,:,i,:] = torch.div(diff, denom)

    #print(vectors.shape, nn.dists.shape)
    return vectors, nn.dists


# check if vectors are coplanar
# hardcoded to k=3 for now. TODO: support arbitrary dimensions
def check_coplanarity(vectors):
    cross = torch.cross(vectors[:,:,1,:], vectors[:,:,2,:], 2)
    #print(vectors[:,:,0,:].shape, cross.shape)
    dot = torch.einsum('ijk,ijk->ij', vectors[:,:,0,:], cross)

    #print(dot.shape, dot[0][:5], torch.max(dot))
    dot = torch.nan_to_num(dot)
    return(torch.absolute(dot))


def directional_chamfer_one_direction(src_pcd_tensor, target_pcd_tensor, k, direction_weight):
    vect, dists = knn_vectors(src_pcd_tensor, target_pcd_tensor, k)
    coplanarity = check_coplanarity(vect)
    dists = dists[:, :, 0] # isolate nearest neighbour

    #print("shapes", coplanarity.shape, dists.shape, (src_pcd_tensor.shape))
    dists = dists * (1-direction_weight) + dists * coplanarity * direction_weight
    dists = torch.sum(torch.sum(dists, dim=1), dim=0)
    dists = dists/coplanarity.shape[0]
    return dists


# compute mahalanobis distance between a set of point clouds and a mixture of gaussians
# specifically, distance is computed against each gaussian in the mixture, and the minimum distance is used
def mahalanobis_distance_gmm(
    target_pcd_tensor, means, covariances, robust=None, delta=0.1, weights=None
):
    # print("inputs", target_pcd_tensor.shape, means.shape, covariances.shape)
    # (b, n, 3), (b, 100, 3)
    dists = torch.zeros(
        (target_pcd_tensor.shape[0], target_pcd_tensor.shape[1], means.shape[1])
    ).cuda()

    # iterate through gaussians in the mixture
    for g in range(means.shape[1]):
        # compute mahalanobis distance between the points in the target cloud and the gaussian
        means_reshaped = (
            means[:, g, :].unsqueeze(1).repeat(1, target_pcd_tensor.shape[1], 1)
        )
        covariances_inv = torch.inverse(covariances)
        # print("inv", covariances_inv.shape)
        covariances_reshaped = (
            covariances_inv[:, g, :, :]
            .unsqueeze(1)
            .repeat(1, target_pcd_tensor.shape[1], 1, 1)
        )
        # print("reshaped", means_reshaped.shape, covariances_reshaped.shape)

        diff = target_pcd_tensor - means_reshaped
        # print("diff", diff.shape, torch.matmul(covariances_reshaped, dedifflta.unsqueeze(3)).shape, diff.view(diff.shape[0], diff.shape[1], 1, diff.shape[2]).shape)
        d = torch.matmul(
            diff.view(diff.shape[0], diff.shape[1], 1, diff.shape[2]),
            torch.matmul(covariances_reshaped, diff.unsqueeze(3)),
        )
        d = d.squeeze()
        # print("d", d.shape)
        dists[:, :, g] = d

    # find minimum distance for each point in the clouds
    min_dists, _ = torch.min(dists, dim=2)
    min_dists = torch.square(min_dists)/1000000
    #print("min dists", dists.shape, min_dists.shape, torch.sum(min_dists, dim=1).shape)

    # reduce
    return torch.sum(min_dists, dim=1)



# this method compares an input point cloud, with a second point cloud
def get_cloud_chamfer_loss_tensor(
    src_pcd_tensor,
    tgt_pcd_tensor,
    alpha=1.0,
    separate_directions=False,
    robust=None,
    delta=0.1,
    bidirectional_robust=True,
    reduction=None
):
    src_pcd_tensor = src_pcd_tensor.transpose(2, 1)
    tgt_pcd_tensor = tgt_pcd_tensor.transpose(2, 1)

    chamferDist = ChamferDistance()
    bidirectional_dist = chamferDist(
        tgt_pcd_tensor,
        src_pcd_tensor,
        bidirectional=True,
        reduction=reduction,
        separate_directions=separate_directions,
    )
    if separate_directions == True:
        bidirectional_dist = torch.cat(
            [
                torch.unsqueeze(bidirectional_dist[0], dim=-1),
                torch.unsqueeze(bidirectional_dist[1], dim=-1),
            ],
            1,
        )

    return bidirectional_dist


def farthest_point_sample_gpu(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    cuda = torch.device("cuda")
    N, D = point.shape
    xyz = point[:, :3]
    centroids = torch.zeros((npoint,), device=cuda, dtype=torch.long)
    distance = torch.ones((N,), device=cuda, dtype=torch.double) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, -1)
    return centroids


def get_pair_loss_clouds_tensor(x, y, k=1, add_pair_loss=True, it=0, return_assignment=True):
    cuda = torch.device("cuda")
    
    chamferDist = ChamferDistance()
    if not add_pair_loss:
        if k==1:
            bidirectional_dist = chamferDist(
                x,
                y,
                bidirectional=True,
                reduction="mean",
                separate_directions=False,
                robust=None
            )
        else:
            nn = chamferDist(x, y, bidirectional=True, return_nn=True, k=k)
            batch_size, point_count, _ = x.shape
            bidirectional_dist = torch.sum(nn[0].dists) + torch.sum(nn[1].dists)
            bidirectional_dist = bidirectional_dist / (batch_size * point_count)
    else:
        # add a loss term for mismatched pairs
        k = 3
        nn = chamferDist(
            x, y, bidirectional=True, return_nn=True, k=k
        )
        #print("d", nn[0].dists.grad_fn, nn[0].idx.grad_fn)
        #bidirectional_dist = torch.sum(nn[1].dists[:,:,0]) + torch.sum(nn[0].dists[:,:,0])
        bidirectional_dist = torch.sum(nn[1].dists[:,:,0]) 
        batch_size, point_count, _ = x.shape
        # print("s", nn[0].idx.shape, nn[1].idx.shape)
        
        idx_fwd = torch.unsqueeze(nn[0].idx[:, :, 0], 2).repeat(1, 1, k)
        idx_bwd = torch.unsqueeze(nn[1].idx[:, :, 0], 2).repeat(1, 1, k)
        # print("f", idx_fwd.shape)
        true_idx_fwd = torch.gather(idx_fwd, 1,  nn[1].idx) # tgt[[src[match]]]
        true_idx_bwd = torch.gather(idx_bwd, 1, nn[0].idx) # tgt[[src[match]]]
        # print("t", true_idx_fwd[0,:2], nn[0].idx[0,649], nn[1].idx[0,:2])
        
        # manual chamfer loss
        paired_points_x_to_y = torch.stack([y[i][nn[0].idx[i]] for i in range(nn[0].idx.shape[0])])
        # print("p", paired_points_x_to_y.shape, torch.unsqueeze(x, 2).repeat(1, 1, k, 1).shape)
        pair_dist_x_to_y = paired_points_x_to_y - torch.unsqueeze(x, 2).repeat(1, 1, k, 1)

        paired_points_y_to_x = torch.stack([x[i][true_idx_bwd[i]] for i in range(true_idx_bwd.shape[0])])
        # print("p2", paired_points_y_to_x.shape, paired_points_x_to_y.shape)
        pair_dist_y_to_x = paired_points_y_to_x - paired_points_x_to_y

        pair_dist = torch.sum(torch.square(pair_dist_x_to_y + pair_dist_y_to_x), 3)
        mdb, min_idx_bwd = torch.min(pair_dist, 2)
        #print("p3", mdb.shape, mdb[0,:5], min(mdb[0]), torch.count_nonzero(mdb[0]))
        
        # select the best neighbour of x in y (nn[0]) such that the x->y->x distance is minimized
        min_dist_bwd = torch.gather(nn[0].dists, 2, min_idx_bwd.unsqueeze(2).repeat(1,1,k))[:, :, 0]
        #print("p4", min_dist_bwd.shape) 
        
        
        # reverse
        rpaired_points_y_to_x = torch.stack([x[i][nn[1].idx[i]] for i in range(nn[1].idx.shape[0])])
        # print("p", paired_points_x_to_y.shape, torch.unsqueeze(x, 2).repeat(1, 1, k, 1).shape)
        rpair_dist_y_to_x = rpaired_points_y_to_x - torch.unsqueeze(y, 2).repeat(1, 1, k, 1)

        rpaired_points_x_to_y = torch.stack([y[i][true_idx_fwd[i]] for i in range(true_idx_fwd.shape[0])])
        # print("p2", paired_points_y_to_x.shape, paired_points_x_to_y.shape)
        rpair_dist_x_to_y = rpaired_points_x_to_y - rpaired_points_y_to_x

        rpair_dist = torch.sum(torch.square(rpair_dist_y_to_x + rpair_dist_x_to_y), 3)
        mdf, min_idx_fwd = torch.min(rpair_dist, 2)
        #print("p5", min_idx_fwd.shape, nn[1].dists.shape)
        
        # select the best neighbour of x in y (nn[0]) such that the x->y->x distance is minimized
        min_dist_fwd = torch.gather(nn[1].dists, 2, min_idx_fwd.unsqueeze(2).repeat(1,1,k))[:, :, 0]
        #print("p6", min_dist_fwd.shape)
        
        pair_distance = torch.sum(min_dist_bwd) #+ torch.sum(min_dist_fwd)
        #pair_dist = torch.sum(mdf) + torch.sum(mdb)
        
        # pair_dist += reverse_pair_dist
        print("manual", (torch.sum(torch.square(rpair_dist_y_to_x[:,:,0,:])) + torch.sum(torch.square(pair_dist_x_to_y[:,:,0,:]))).item())
        print("dist", bidirectional_dist.item(), pair_distance.item())
        bidirectional_dist = bidirectional_dist + pair_distance
        #bidirectional_dist = pair_distance 
        bidirectional_dist = bidirectional_dist / (batch_size)
        
            
    if return_assignment:
        min_ind_1 = torch.gather(nn[1].idx, 2, min_idx_fwd.unsqueeze(2).repeat(1,1,k))[:, :, 0][0]
        min_ind_0 = torch.gather(nn[0].idx, 2, min_idx_bwd.unsqueeze(2).repeat(1,1,k))[:, :, 0][0]
        return bidirectional_dist, [min_ind_0, min_ind_1]
    
    else:
        return bidirectional_dist


# add jitter to the point correspondences in CD
# NOTE: hard coded for batch_size = 1 only
def get_jittery_cd_tensor(x, y, k=1, it=0):
    cuda = torch.device("cuda")
    chamferDist = ChamferDistance()

    nn = chamferDist(
        x, y, bidirectional=True, return_nn=True, k=1
    )
    #print("d", nn[0].dists.grad_fn, nn[0].idx.grad_fn)
    bidirectional_dist = torch.sum(nn[1].dists) + torch.sum(nn[0].dists)
    batch_size, point_count, _ = x.shape
    
    fps= False

    #jitter_size = int(64*(0.001*(1000-it)))+1
    jitter_size = 2
    print("jitter", jitter_size)
    perm1 = torch.randperm(x.size(1), device=cuda)[:jitter_size]
    perm2 = torch.randperm(x.size(1), device=cuda)[:jitter_size].unsqueeze(1)
    nn_copy = nn[0].idx.clone()
    
    if fps:
        # farthest point sample
        centroids  = farthest_point_sample_gpu(x[0], jitter_size)
        nn_copy[0][perm1] = centroids.unsqueeze(1)
    else:   
        # randomly permute
        for cloud in nn_copy:
            cloud[perm1] = perm2
    paired_points_x_to_y = torch.stack([y[i][torch.flatten(nn_copy[i])] for i in range(nn_copy.shape[0])])
    pair_dist_x_to_y = paired_points_x_to_y - x
    
    # reverse
    rperm1 = torch.randperm(y.size(1), device=cuda)[:jitter_size]
    rperm2 = torch.randperm(y.size(1), device=cuda)[:jitter_size].unsqueeze(1)
    rnn_copy = nn[1].idx.clone()
            
    if fps:
        # farthest point sample
        rcentroids  = farthest_point_sample_gpu(y[0], jitter_size)
        rnn_copy[0][rperm1] = rcentroids.unsqueeze(1)
    else:
        #randomly permute
        for cloud in rnn_copy:
            cloud[rperm1] = rperm2
    rpaired_points_x_to_y = torch.stack([x[i][torch.flatten(rnn_copy[i])] for i in range(rnn_copy.shape[0])])
    rpair_dist_x_to_y = rpaired_points_x_to_y - y
    
    
    pair_dist = torch.sum(torch.square(pair_dist_x_to_y)) + torch.sum(torch.square(rpair_dist_x_to_y))
    
    print("dist", bidirectional_dist.item(), pair_dist.item())
    #bidirectional_dist = bidirectional_dist + pair_dist
    bidirectional_dist = pair_dist 
    bidirectional_dist = bidirectional_dist / (batch_size)
        
    return bidirectional_dist


# add self loss to CD
def get_self_cd_tensor(x, y, thresh=0.001):
    cuda = torch.device("cuda")
    
    chamferDist = ChamferDistance()

    # add a loss term for mismatched pairs
    nn = chamferDist(
        x, y, bidirectional=True, return_nn=True, k=1
    )
    #print("d", nn[0].dists.grad_fn, nn[0].idx.grad_fn)
    bidirectional_dist = torch.sum(nn[1].dists) + torch.sum(nn[0].dists)
    batch_size, point_count, _ = x.shape
    
    # compute self loss for gen cloud
    nn2 = chamferDist(y, y, bidirectional=False, return_nn=True, k=2)
    self_loss = torch.sum(torch.square(torch.clamp(thresh - nn2[0].dists[:,:,1], min=0)))
    #self_loss = torch.sum(torch.square((torch.abs(nn[0].dists[:,:,0] - nn2[0].dists[:,:,1]))))

    print("dist", bidirectional_dist.item(), self_loss.item())
    bidirectional_dist = bidirectional_dist + self_loss*1000
    #bidirectional_dist = pair_dist 
    bidirectional_dist = bidirectional_dist / (batch_size)
        
    return bidirectional_dist


# compute reverse weighted chamfer loss
# this is computed by calclating nn at a large k, then scaling each correspondences's 
# distance by the reverse CD of that correspondence. The minimum of these is used to index 
# the coorespondence to be chosen for measuring chamfer distance.
# In other words, whenever a point in cloud B already has a close correspondence in cloud A,
# it becomes less attractive to other points in cloud A, pushing points in cloud A to find 
# other correspondences.
def calc_reverse_weighted_cd_tensor(x, y, k=32, return_assignment=False):   
    chamferDist = ChamferDistance()
    # add a loss term for mismatched pairs
    nn = chamferDist(
        x, y, bidirectional=True, return_nn=True, k=k
    )
    pow = 2
    #print("d", nn[0].dists.shape, nn[0].idx.shape)
    
    # get closest distances in reverse direction
    scaling_factors_1 = nn[0].dists[:,:,0].unsqueeze(2).repeat(1, 1, k)
    denominator_1 = torch.pow(torch.gather(scaling_factors_1, 1, nn[1].idx), pow)
    #denominator_1 = torch.gather(scaling_factors_1, 1, nn[1].idx)
    # divide by closest distance in reverse direction, selectfind minimum
    scaled_dist_1 = torch.div(nn[1].dists, denominator_1)
    scaled_dist_1x, i1 = torch.min(scaled_dist_1, 2)
    #scaled_dist_1x = scaled_dist_1x - torch.ones_like(scaled_dist_1x)
    #print("d", torch.min(scaled_dist_1x[0]))
    
    
    #min_dist_1 = torch.stack([nn[1].dists[0][i][i1[0][i]] for i in range(nn[1].dists[0].shape[0])]).unsqueeze(0)
    # select distance that corresponds to above minimum index
    min_dist_1 = torch.gather(nn[1].dists, 2, i1.unsqueeze(2).repeat(1,1,k))[:, :, 0]
    #print("s", min_dist_1[0][100], nn[1].dists[0][100])

    # reverse direction
    scaling_factors_0 = nn[1].dists[:,:,0].unsqueeze(2).repeat(1, 1, k)
    #denominator_0 = torch.gather(scaling_factors_0, 1, nn[0].idx)
    denominator_0 = torch.pow(torch.gather(scaling_factors_0, 1, nn[1].idx), pow)
    scaled_dist_0 = torch.div(nn[0].dists, denominator_0)
    scaled_dist_0x, i0 = torch.min(scaled_dist_0, 2)
    #scaled_dist_0x = scaled_dist_0x - torch.ones_like(scaled_dist_0x)
    #print(i2.shape)
    #min_dist_0 = torch.stack([nn[0].dists[0][i][i2[0][i]] for i in range(nn[0].dists[0].shape[0])]).unsqueeze(0)
    min_dist_0 = torch.gather(nn[0].dists, 2, i0.unsqueeze(2).repeat(1,1,k))[:, :, 0]
 
    #bidirectional_dist = torch.sum(nn[1].dists[:,:,0]) + torch.sum(nn[0].dists[:, :, 0])
    #self_loss = torch.sum(scaled_dist_1x) + torch.sum(scaled_dist_0x)
    self_loss = torch.sum(min_dist_1) + torch.sum(min_dist_0)
    batch_size, point_count, _ = x.shape

    #print("dist", bidirectional_dist.item(), self_loss.item())
    #bidirectional_dist = bidirectional_dist #+ self_loss
    bidirectional_dist = self_loss
    bidirectional_dist = bidirectional_dist / (batch_size)
    
    
    if return_assignment:
        min_ind_1 = torch.gather(nn[1].idx, 2, i1.unsqueeze(2).repeat(1,1,k))[:, :, 0][0]
        min_ind_0 = torch.gather(nn[0].idx, 2, i0.unsqueeze(2).repeat(1,1,k))[:, :, 0][0]

        return bidirectional_dist, [min_ind_0, min_ind_1]
    else:
        return bidirectional_dist


# weight the distance of each correspondence by the distances to all its correspondences
def calc_neighbour_weighted_cd_tensor(x, y, k=32, return_assignment=True):   
    cuda = torch.device("cuda")
    chamferDist = ChamferDistance()
    # add a loss term for mismatched pairs
    nn = chamferDist(
        x, y, bidirectional=True, return_nn=True, k=k
    )
    #print("d", nn[0].dists.shape, nn[0].idx.shape)
    
    # compile a list of points in y that correspond to x
    # NOTE: only for batch size = 1
    # sum 1/ all distances from y that correspond to x for each point in x
    # nn[0] is from x to y, nn[1] is from y to x 
    # Create a mask where nn[1].idx[0, :, 0] is equal to the range values (0, 1, 2, ..., nn[0].idx.shape[1] - 1)
    mask = (nn[1].idx[0, :, 0].unsqueeze(1) == torch.arange(nn[0].idx.shape[1], device=cuda).unsqueeze(0))

    # Calculate values for each index using the mask
    values = torch.where(mask, 1/nn[1].dists[0, :, 0], torch.tensor(0., device=cuda))

    # Sum along the appropriate dimension to get the final dists_x
    dists_x = torch.sum(values, dim=0)
    # print(dists_x.shape, dists_x[:5])

    dists_x = (dists_x + 1.).unsqueeze(0)

    # scale all distances by the scaling factor
    scaling_factors_1 = dists_x.unsqueeze(2).repeat(1, 1, k)
    denominator_1 = torch.gather(scaling_factors_1, 1, nn[1].idx)
    # divide by closest distance in reverse direction, select minimum
    scaled_dist_1 = torch.mul(nn[1].dists, denominator_1)
    scaled_dist_1x, i1 = torch.min(scaled_dist_1, 2)

    #print("den", dists_x[0][:5])
    # select distance that corresponds to above minimum index
    min_dist_1 = torch.gather(nn[1].dists, 2, i1.unsqueeze(2).repeat(1,1,k))[:, :, 0]
    #print("s", min_dist_1[0][100], nn[1].dists[0][100])


    # reverse direction
    # Create a mask where nn[1].idx[0, :, 0] is equal to the range values (0, 1, 2, ..., nn[0].idx.shape[1] - 1)
    mask = (nn[0].idx[0, :, 0].unsqueeze(1) == torch.arange(nn[1].idx.shape[1], device=cuda).unsqueeze(0))

    # Calculate values for each index using the mask
    values = torch.where(mask, 1/nn[0].dists[0, :, 0], torch.tensor(0., device=cuda))

    # Sum along the appropriate dimension to get the final dists_x
    dists_y = torch.sum(values, dim=0)
    dists_y = (dists_y + 1.).unsqueeze(0)
    
    scaling_factors_0 = dists_y.unsqueeze(2).repeat(1, 1, k)
    denominator_0 = torch.gather(scaling_factors_0, 1, nn[0].idx)
    scaled_dist_0 = torch.mul(nn[0].dists, denominator_0)
    #])
    scaled_dist_0x, i0 = torch.min(scaled_dist_0, 2)
    #scaled_dist_0x = scaled_dist_0x - torch.ones_like(scaled_dist_0x)
    #print(i2.shape)
    #min_dist_0 = torch.stack([nn[0].dists[0][i][i2[0][i]] for i in range(nn[0].dists[0].shape[0])]).unsqueeze(0)
    min_dist_0 = torch.gather(nn[0].dists, 2, i0.unsqueeze(2).repeat(1,1,k))[:, :, 0]
 
    # print("d", nn[0].dists[:,:5,0], scaled_dist_0x[0, :5])
    # print("d2", nn[1].dists[:,:5,0], scaled_dist_1x[0, :5], scaled_dist_1[0, :5], denominator_1[0, :5], nn[1].dists[0:,:5])
    bidirectional_dist = torch.sum(nn[1].dists[:,:,0]) + torch.sum(nn[0].dists[:, :, 0])
    
    self_loss = torch.sum(scaled_dist_1x) + torch.sum(scaled_dist_0x)
    #self_loss = torch.sum(min_dist_1) + torch.sum(min_dist_0)
    batch_size, point_count, _ = x.shape

    #print("dist", bidirectional_dist.item(), self_loss.item())
    #bidirectional_dist = bidirectional_dist #+ self_loss
    bidirectional_dist = self_loss
    bidirectional_dist = bidirectional_dist / (batch_size)
    
    if return_assignment:
        min_ind_1 = torch.gather(nn[1].idx, 2, i1.unsqueeze(2).repeat(1,1,k))[:, :, 0][0]
        min_ind_0 = torch.gather(nn[0].idx, 2, i0.unsqueeze(2).repeat(1,1,k))[:, :, 0][0]

        return bidirectional_dist, [min_ind_0, min_ind_1]
    else:
        return bidirectional_dist


# for each point Ai, measure the probability of being pared with each point in the other cloud B
# as a continuous function going to zero at k.
# P is thresholded by max and min values of the value matrix
# do the same for cloud B
# loss = SUM( P(Ai->Bj) * (1 - P(Bj->Ai))) and vice versa 
def calc_pairing_probabilty_loss_tensor(x, y, k=32, return_assignment=True):
    cuda = torch.device("cuda")
    chamferDist = ChamferDistance()
    # add a loss term for mismatched pairs
    nn = chamferDist(
        x, y, bidirectional=True, return_nn=True, k=k
    )
    #print("d", nn[0].dists.shape, nn[0].idx.shape)
    
    # compile a list of points in y that correspond to x
    # NOTE: only for batch size = 1
    # sum 1/ all distances from y that correspond to x for each point in x
    # nn[0] is from x to y, nn[1] is from y to x 
    # Create a mask where nn[1].idx[0, :, 0] is equal to the range values (0, 1, 2, ..., nn[0].idx.shape[1] - 1)
    # print("a", nn[1].idx[0, :, 0].unsqueeze(1).shape, torch.arange(nn[0].idx.shape[1]).unsqueeze(0).shape)
    # print("a2", nn[1].idx[0].shape, torch.arange(nn[0].idx.shape[1]).unsqueeze(0).repeat(k,1).shape)
    # print("a3", nn[0].idx[0, :, 0].unsqueeze(1).shape, torch.arange(nn[1].idx.shape[1], device=cuda).unsqueeze(0).shape)
    # print("b", nn[1].dists[0,:,0].shape)
    
    
    # probs_x = torch.zeros((nn[0].idx.shape[1], nn[1].idx.shape[1]), device=cuda)
    # for i in range(nn[0].idx.shape[1]):
    #     values = torch.where(nn[1].idx[0] == i, 1/nn[1].dists[0], 0.01)
    #     #print("v", values.shape, torch.sum(values, 1).shape)
    #     #print(dists_x.shape, torch.sum(values, 1).shape)
    #     probs_x[i] = torch.sum(values, 1)
    
    idx = torch.arange(nn[0].idx.shape[1], device=cuda).unsqueeze(1).unsqueeze(2)
    probs_x = torch.where(nn[1].idx[0] == idx, 1/nn[1].dists[0], 0.01)
    probs_x = torch.sum(probs_x, 2)
    #print("p", probs_x.shape)
    #probs_x = probs_x / torch.max(probs_x)
    probs_x = F.normalize(probs_x, p=2, dim=0)
    #print(probs_x.shape, torch.count_nonzero(probs_x, 1))
 
    idx = torch.arange(nn[1].idx.shape[1], device=cuda).unsqueeze(1).unsqueeze(2)
    probs_y = torch.where(nn[0].idx[0] == idx, 1/nn[0].dists[0], 0.01)
    probs_y = torch.sum(probs_y, 2)
    #probs_y = probs_y / torch.max(probs_y)
    probs_y = F.normalize(probs_y, p=2, dim=0)
    #print(torch.max(probs_y, dim=0))


    # mask = (nn[1].idx[0] == torch.arange(nn[0].idx.shape[1], device=cuda).unsqueeze(0).repeat(k,1))
    # #mask = (nn[1].idx[0, :, 0].unsqueeze(1) == torch.arange(nn[0].idx.shape[1], device=cuda).unsqueeze(0))
    # print("m", mask.shape, torch.count_nonzero(mask))

    # # Calculate values for each index using the mask
    # probs_x = torch.where(mask, 1/nn[1].dists[0], torch.tensor(0., device=cuda))
    # print("nz", probs_x[:2], torch.count_nonzero(probs_x, 0))
    # # normalise. farthest distance (out of knn range) = 0, closest = 1
    # probs_x = probs_x / torch.max(probs_x)
    # # print("values", probs_x.shape, torch.min(probs_x), torch.max(probs_x))

    # # reverse direction
    # # Create a mask where nn[1].idx[0, :, 0] is equal to the range values (0, 1, 2, ..., nn[0].idx.shape[1] - 1)
    # mask = (nn[0].idx[0, :, 0].unsqueeze(1) == torch.arange(nn[1].idx.shape[1], device=cuda).unsqueeze(0))

    # # Calculate values for each index using the mask
    # probs_y = torch.where(mask, 1/nn[0].dists[0, :, 0], torch.tensor(0., device=cuda))
    # # normalise. farthest distance (out of knn range) = 0, closest = 1
    # probs_y = probs_y / torch.max(probs_y)
    
    
    #print("values", probs_y.shape, torch.min(probs_y), torch.max(probs_y))
    #probability_loss = torch.sum(torch.mul(probs_x, (1 - torch.transpose(probs_y, 0, 1)))) + torch.sum(torch.mul(probs_y, (1 - torch.transpose(probs_x, 0, 1))))
    probability_loss = torch.sum(1. - torch.mul(probs_x, torch.transpose(probs_y, 0, 1))) #+ torch.sum(1 - torch.mul(probs_y, torch.transpose(probs_x, 0, 1)))
    #probability_loss = -1*torch.sum(torch.mul(probs_x, torch.transpose(probs_y, 0, 1))) #+ torch.sum(1 - torch.mul(probs_y, torch.transpose(probs_x, 0, 1)))
    probability_loss = probability_loss*0.1
    
    bidirectional_dist = torch.sum(nn[1].dists[:,:,0]) + torch.sum(nn[0].dists[:, :, 0])

    batch_size, point_count, _ = x.shape

    # print("dist", bidirectional_dist.item(), probability_loss.item())
    bidirectional_dist = bidirectional_dist + probability_loss
    #bidirectional_dist = probability_loss
    bidirectional_dist = bidirectional_dist / (batch_size)
    
    if return_assignment:
        return bidirectional_dist, [nn[0].idx[0,:,0], nn[1].idx[0,:,0]]
    return bidirectional_dist


# aside from matching by shortest distance, also match by density around each point
# density for each point is measured by the sum of its distances to its k neighbours in the same cloud
def calc_uniform_chamfer_loss_tensor(x, y, k=32, return_assignment=False, return_dists=False):
    chamferDist = ChamferDistance()
    eps = 0.00001
    k=32
    k2 = 32 # reduce k to check density in smaller patches
    power = 2
    # add a loss term for mismatched pairs
    nn = chamferDist(
        x, y, bidirectional=True, return_nn=True, k=k
    )
    
 
    # measure density with itself
    nn_x = chamferDist(x, x, bidirectional=False, return_nn=True, k=k2)
    density_x = torch.mean(nn_x[0].dists[:,:,1:], dim=2)
    density_x = 1 / (density_x + eps)
    high, low = torch.max(density_x), torch.min(density_x)
    diff = high - low
    density_x = (density_x - low) / diff
    
    # measure density with other cloud
    density_xy = torch.mean(nn[0].dists[:,:,:k2-1], dim=2)
    density_xy = 1 / (density_xy + eps)
    high, low = torch.max(density_xy), torch.min(density_xy)
    diff = high - low
    density_xy = (density_xy - low) / diff
    w_x = torch.div(density_xy, density_x)
    #print("w", w_x.shape, w_x[0])
    w_x = torch.pow(w_x, power)
    scaling_factors_1 = w_x.unsqueeze(2).repeat(1, 1, k)
    multiplier = torch.gather(scaling_factors_1, 1, nn[1].idx)
    
    scaled_dist_1 = torch.mul(nn[1].dists, multiplier)
    scaled_dist_1x, i1 = torch.min(scaled_dist_1, 2)
        
    # measure density with itself
    nn_y = chamferDist(y, y, bidirectional=False, return_nn=True, k=k2)
    density_y = torch.mean(nn_y[0].dists[:,:,1:], dim=2)
    density_y = 1 / (density_y + eps)
    high, low = torch.max(density_y), torch.min(density_y)
    diff = high - low
    density_y = (density_y - low) / diff
    
    # measure density with other cloud
    density_yx = torch.mean(nn[1].dists[:,:,:k2-1], dim=2)
    density_yx = 1 / (density_yx + eps)
    high, low = torch.max(density_yx), torch.min(density_yx)
    diff = high - low
    density_yx = (density_yx - low) / diff
    w_y = torch.div(density_yx, density_y)
    #print("w", w_x.shape, w_x[0])
    w_y = torch.pow(w_y, power)
    scaling_factors_0 = w_y.unsqueeze(2).repeat(1, 1, k)
    multiplier = torch.gather(scaling_factors_0, 1, nn[0].idx)
    
    scaled_dist_0 = torch.mul(nn[0].dists, multiplier)
    scaled_dist_0x, i0 = torch.min(scaled_dist_0, 2)
    
    #print("d", w_x.shape, i1.shape)
    # reverse

    min_dist_1 = torch.gather(nn[1].dists, 2, i1.unsqueeze(2).repeat(1,1,k))[:, :, 0]
    min_dist_0 = torch.gather(nn[0].dists, 2, i0.unsqueeze(2).repeat(1,1,k))[:, :, 0]
    
    uniform_cd = torch.sum(torch.sqrt(min_dist_1)) + torch.sum(torch.sqrt(min_dist_0))
    #uniform_cd = torch.sum(min_dist_1) + torch.sum(min_dist_0)
    #uniform_cd = torch.sum(min_dist_1) + torch.sum(nn[0].dists[:, :, 0])
    batch_size, point_count, _ = x.shape

    #print("dist", bidirectional_dist.item(), self_loss.item())
    bidirectional_dist = torch.sum(nn[1].dists[:,:,0]) + torch.sum(nn[0].dists[:, :, 0])
    # print("cd", torch.sum(nn[1].dists[:,:,0]).item(), torch.sum(nn[0].dists[:, :, 0]).item())
    # print("uniform", torch.sum(min_dist_1).item(), torch.sum(min_dist_0).item())
    bidirectional_dist = uniform_cd
    bidirectional_dist = bidirectional_dist / (batch_size)
    
    if return_dists:
        return min_dist_0, min_dist_1
    
    if return_assignment:
        min_ind_1 = torch.gather(nn[1].idx, 2, i1.unsqueeze(2).repeat(1,1,k))[:, :, 0]
        min_ind_0 = torch.gather(nn[0].idx, 2, i0.unsqueeze(2).repeat(1,1,k))[:, :, 0]
        
        return bidirectional_dist, [min_ind_0.detach().cpu().numpy(), min_ind_1.detach().cpu().numpy()]
    else:
        return bidirectional_dist
    
    

# aside from matching by shortest distance, also match by density around each point
# density for each point is measured by the sum of its distances to its k neighbours in the same cloud
def calc_relative_density_loss_tensor(x, y, k=32, return_assignment=False):
    chamferDist = ChamferDistance()
    eps = 0.00001

    # add a loss term for mismatched pairs
    nn = chamferDist(
        x, y, bidirectional=True, return_nn=True, k=k
    )
    
    k2 = 32 # reduce k to check density in smaller patches
    
    # measure density with itself
    nn_x = chamferDist(x, x, bidirectional=False, return_nn=True, k=k2)
    density_x = torch.mean(nn_x[0].dists[:,:,1:], dim=2)
    density_x = 1 / (density_x + eps)
    # high, low = torch.max(density_x), torch.min(density_x)
    # diff = high - low
    # density_x = (density_x - low) / diff
    
    # measure density with other cloud
    density_xy = torch.mean(nn[0].dists[:,:,:k2-1], dim=2)
    density_xy = 1 / (density_xy + eps)
    # high, low = torch.max(density_xy), torch.min(density_xy)
    # diff = high - low
    # density_xy = (density_xy - low) / diff
    
    w_x = torch.sqrt(torch.div(density_xy, density_x))
    #print("w", w_x.shape, w_x[0])

    # measure density with itself
    nn_y = chamferDist(y, y, bidirectional=False, return_nn=True, k=k2)
    density_y = torch.mean(nn_y[0].dists[:,:,1:], dim=2)
    density_y = 1 / (density_y + eps)
    # high, low = torch.max(density_y), torch.min(density_y)
    # diff = high - low
    # density_y = (density_y - low) / diff
    
    # measure density with other cloud
    density_yx = torch.mean(nn[1].dists[:,:,:k2-1], dim=2)
    density_yx = 1 / (density_yx + eps)
    # high, low = torch.max(density_yx), torch.min(density_yx)
    # diff = high - low
    # density_yx = (density_yx - low) / diff
    
    w_y = torch.sqrt(torch.div(density_yx, density_y))
    #print("w", w_x.shape, w_x[0])

    d_loss_x = torch.sum(torch.abs(w_x - 1))
    d_loss_y = torch.sum(torch.abs(w_y - 1))

    density_loss = d_loss_x + d_loss_y
    
    batch_size, point_count, _ = x.shape
    bidirectional_dist = torch.sum(nn[1].dists[:,:,0]) + torch.sum(nn[0].dists[:, :, 0])
    print("d", d_loss_x.item(), d_loss_y.item(), bidirectional_dist.item())

    bidirectional_dist = bidirectional_dist + density_loss/10
    bidirectional_dist = bidirectional_dist / (batch_size)

    return bidirectional_dist


# aside from matching by shortest distance, also match by density around each point
# density for each point is measured by the sum of its distances to its k neighbours in the same cloud
def calc_uniform_single_chamfer_loss_tensor(x, y, k=32, return_assignment=False):
    chamferDist = ChamferDistance()
    eps = 0.00001

    # add a loss term for mismatched pairs
    nn = chamferDist(
        x, y, bidirectional=True, return_nn=True, k=k
    )
    
    k2 = 32 # reduce k to check density in smaller patches
    power = 8
    # measure density with itself
    # nn_x = chamferDist(x, x, bidirectional=False, return_nn=True, k=k2)
    # density_x = torch.mean(nn_x[0].dists[:,:,1:], dim=2)
    # density_x = 1 / (density_x + eps)
    # high, low = torch.max(density_x), torch.min(density_x)
    # diff = high - low
    # density_x = (density_x - low) / diff
    
    # measure density with other cloud
    density_xy = torch.mean(nn[0].dists[:,:,:k2], dim=2)
    density_xy = 1 / (density_xy + eps)
    high, low = torch.max(density_xy), torch.min(density_xy)
    diff = high - low
    density_xy = (density_xy - low) / diff
    #w_x = torch.div(density_xy, density_x)
    w_x = density_xy
    #print("w", w_x.shape, w_x[0])
    w_x = torch.pow(w_x, power)
    scaling_factors_1 = w_x.unsqueeze(2).repeat(1, 1, k)
    multiplier = torch.gather(scaling_factors_1, 1, nn[1].idx)
    
    scaled_dist_1 = torch.mul(nn[1].dists, multiplier)
    scaled_dist_1x, i1 = torch.min(scaled_dist_1, 2)
        
    # measure density with itself
    # nn_y = chamferDist(y, y, bidirectional=False, return_nn=True, k=k2)
    # density_y = torch.mean(nn_y[0].dists[:,:,1:], dim=2)
    # density_y = 1 / (density_y + eps)
    # high, low = torch.max(density_y), torch.min(density_y)
    # diff = high - low
    # density_y = (density_y - low) / diff
    
    # measure density with other cloud
    density_yx = torch.mean(nn[1].dists[:,:,:k2], dim=2)
    density_yx = 1 / (density_yx + eps)
    high, low = torch.max(density_yx), torch.min(density_yx)
    diff = high - low
    density_yx = (density_yx - low) / diff
    #w_x = torch.div(density_yx, density_y)
    w_x = density_yx
    #print("w", w_x.shape, w_x[0])
    w_x = torch.pow(w_x, power)
    scaling_factors_0 = w_x.unsqueeze(2).repeat(1, 1, k)
    multiplier = torch.gather(scaling_factors_0, 1, nn[0].idx)
    
    scaled_dist_0 = torch.mul(nn[0].dists, multiplier)
    scaled_dist_0x, i0 = torch.min(scaled_dist_0, 2)
    
    #print("d", w_x.shape, i1.shape)
    # reverse

    min_dist_1 = torch.gather(nn[1].dists, 2, i1.unsqueeze(2).repeat(1,1,k))[:, :, 0]
    min_dist_0 = torch.gather(nn[0].dists, 2, i0.unsqueeze(2).repeat(1,1,k))[:, :, 0]
    
    uniform_cd = torch.sum(min_dist_1) + torch.sum(min_dist_0)
    #uniform_cd = torch.sum(min_dist_1) + torch.sum(nn[0].dists[:, :, 0])
    batch_size, point_count, _ = x.shape

    #print("dist", bidirectional_dist.item(), self_loss.item())
    bidirectional_dist = torch.sum(nn[1].dists[:,:,0]) + torch.sum(nn[0].dists[:, :, 0])
    #print("dist, uniform", bidirectional_dist.item(), uniform_cd.item())
    bidirectional_dist = uniform_cd
    bidirectional_dist = bidirectional_dist / (batch_size)
    
    if return_assignment:
        min_ind_1 = torch.gather(nn[1].idx, 2, i1.unsqueeze(2).repeat(1,1,k))[:, :, 0][0]
        min_ind_0 = nn[0].idx[0,:,0]

        return bidirectional_dist, [min_ind_0, min_ind_1]
    else:
        return bidirectional_dist


# infoCD loss
def calc_cd_like_InfoV2(x, y, return_assignment=False):
    chamferDist = ChamferDistance()
    nn = chamferDist(
        x, y, bidirectional=True, return_nn=True)
        
    dist1, dist2, idx1, idx2 = nn[0].dists, nn[1].dists, nn[0].idx, nn[1].idx
    dist1 = torch.clamp(dist1, min=1e-9)
    dist2 = torch.clamp(dist2, min=1e-9)
    d1 = torch.sqrt(dist1)
    d2 = torch.sqrt(dist2)

    distances1 = - torch.log(torch.exp(-0.5 * d1)/(torch.sum(torch.exp(-0.5 * d1) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)
    distances2 = - torch.log(torch.exp(-0.5 * d2)/(torch.sum(torch.exp(-0.5 * d2) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)

    if return_assignment:
        return (torch.sum(distances1) + torch.sum(distances2)) / 2, [idx1.detach().cpu().numpy(), 
                                                                     idx2.detach().cpu().numpy()]
    return (torch.sum(distances1) + torch.sum(distances2)) / 2


# return any of chamfer, EMD, reverse or jittered chamfer loss
# NOTE: for EMD, gradient is only calculated for y, not x
def calculate_3d_loss(x, y, loss_funcs, it=0, batch_size=None):

    losses = {}
    for loss_func in loss_funcs:
            if loss_func == "chamfer":
                chamferDist = ChamferDistance()
                losses[loss_func] = chamferDist(x, y, bidirectional=True).item()
            elif loss_func == "emd":
                losses[loss_func] = calc_emd(y, x)[0].item()
            elif loss_func == "reverse":
                losses[loss_func] = calc_reverse_weighted_cd_tensor(x, y, return_assignment=False).item()
            elif loss_func == "jittery":
                losses[loss_func] = get_jittery_cd_tensor(x, y, it=it).item()
    
    return losses



# measure point-wise distance between two clouds
def get_point_distance(src, tgt, loss="chamfer"):
    cuda = torch.device("cuda")
    src_tensor = torch.tensor([src], device=cuda)
    tgt_tensor = torch.tensor([tgt], device=cuda)
    chamferDist = ChamferDistance()

    if loss == "chamfer":
        nn = chamferDist(
            src_tensor, tgt_tensor, bidirectional=True, return_nn=True)
        weights = nn[1].dists[0,:,0]
        #print( torch.sum(nn[1].dists))
        loss = torch.sum(nn[1].dists)
        
    elif loss == "uniform":
        dist_0, dist_1 = calc_uniform_chamfer_loss_tensor(src_tensor, tgt_tensor, return_dists=True, k=32)
        loss = torch.sum(dist_1[0])
        #print( torch.sum(dist_1[0]))
        weights = dist_1[0]
    
    return weights.detach().cpu().numpy(), loss.item()


def calc_direct(x, y):
    return torch.sum(torch.square(x-y))