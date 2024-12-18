import torch
from shapely import LineString, frechet_distance

def get_features(x):
    last_frame = torch.roll(x, 1, 2) 
    delta_xy = (x - last_frame)[:,:,1:,:]
    speed = torch.norm(delta_xy, dim=3)
    avg_speed_single_period = torch.nanmean(speed, dim=2)
    avg_speed = torch.nanmean(avg_speed_single_period, dim=1)


    acceleration = speed - torch.roll(speed, 1, 2)

    avg_acc = torch.nanmean(torch.nanmean(acceleration, dim=2), dim=1)
    threshold = 0.5
    active_frames = speed > threshold
    active_time = torch.sum(active_frames, dim=2)
    # divide active_time by number of non-nan frames
    active_time = active_time / (active_frames.shape[2] - torch.sum(torch.isnan(speed), dim=2))
    avg_active_time = torch.nanmean(active_time, dim=1,dtype=torch.float32)
    sharp_turn = torch.sum(torch.sum(delta_xy*torch.roll(delta_xy, 1, 2),dim=3)[:,:,2:]<0,dim=2)

    # divide sharp_turn by number of non-nan frames
    sharp_turn = sharp_turn / (delta_xy.shape[2] - torch.sum(torch.any(torch.isnan(delta_xy),dim=3), dim=2))
    avg_sharp_turn = torch.nanmean(sharp_turn, dim=1,dtype=torch.float32)



    x_shifted = x-x[:,:,0,:].unsqueeze(2)
    rotation_angle = -torch.arctan(x_shifted[:,:,-1,1]/x_shifted[:,:,-1,0]).unsqueeze(2)
    # angles = torch.arctan(x_shifted[:,:,:,1]/x_shifted[:,:,:,0])
    # rotation_angles = end_angle - angles
    cos_rot = torch.cos(rotation_angle).unsqueeze(3)
    sin_rot = torch.sin(rotation_angle).unsqueeze(3)
    rotated_x = torch.sum(x_shifted.unsqueeze(4)*torch.cat((cos_rot, -sin_rot,sin_rot,cos_rot), dim=3).reshape(x_shifted.shape[0],x_shifted.shape[1],1,2,2),axis=4)

    final_x=torch.where(torch.isnan(rotated_x), x_shifted, rotated_x)

    #fill nan on final x by using the next and last frame along dimension 2
    final_x_filled = torch.where(torch.isnan(final_x), torch.roll(final_x, 1, 2), final_x)
    final_x_filled = torch.where(torch.isnan(final_x_filled), torch.roll(final_x_filled, -1, 2), final_x_filled)
    

    frechet_dist = torch.zeros(x.shape[0], x.shape[1]-1)
    for i in range(x.shape[0]):
        for period in range(x.shape[1]-1):
            frechet_dist[i][period] = frechet_distance(LineString(final_x_filled[i][period]), LineString(final_x_filled[i][period+1]))

    mean_frechet_dist = torch.nanmean(frechet_dist, dim=1)
    mean_frechet_dist = torch.where(torch.isnan(mean_frechet_dist), torch.nanmean(mean_frechet_dist), mean_frechet_dist)

    features = torch.stack((avg_speed, avg_acc, avg_active_time, avg_sharp_turn, mean_frechet_dist), dim=1)

    features = (features - torch.mean(features, dim=0))/torch.std(features, dim=0)
    return features
