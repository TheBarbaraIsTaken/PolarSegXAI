'Helper functions for loading and visualizing SemanticKITTI PCD data'
import numpy as np
import yaml
import torch

def read_velodyne_bin(bin_path):
    'Reading Velodyne .bin file'
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # Points are represented by (x, y, z, intensity)

    return points

def load_labels(file_path):
    'Reading .label file'
    labels = np.fromfile(file_path, dtype=np.uint32)

    return labels & 0xFFFF  # SemanticKITTI uses the lower 16 bits for labels

def label2color(labels):
    'Encode the lables to colors using the color map of SemanticKITTI'
    colors = np.zeros((labels.shape[0], 3))
    unique_labels = np.unique(labels)
    with open('semantic-kitti.yaml', 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    colormap = semkittiyaml['color_map']

    for label in unique_labels:
        colors[labels == label] = colormap.get(label, [255, 255, 255])  # Default to black if label not in colormap

    return colors[:, ::-1] / 255.0

def get_grid_ind(xyz_pol):
    'Get the grid index for each point in pcd'
    # fixed volume space constans
    max_bound = np.asarray([50,np.pi,1.5])
    min_bound = np.asarray([3,-np.pi,-3])
    cur_grid_size = np.asarray([480,360,32])

    crop_range = max_bound - min_bound
    intervals = crop_range/(cur_grid_size-1)

    grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int64)

    return grid_ind

def cart2polar(input_xyz):
    'Cartezian coordinates to polar'
    rho = np.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
    phi = np.arctan2(input_xyz[:,1],input_xyz[:,0])

    return np.stack((rho,phi,input_xyz[:,2]),axis=1)

def shift_labels(labels):
    'Shift label numbers to SemanticKITTI standard'
    def train2SemKITTI(input_label):
        # delete 0 label (uses uint8 trick : 0 - 1 = 255 )
        return input_label + 1
    
    labels = train2SemKITTI(labels)
    labels = np.expand_dims(labels,axis=1)
    labels = labels.astype(np.uint32)

    DATA = yaml.safe_load(open("semantic-kitti.yaml", 'r'))
    remapdict = DATA["learning_map_inv"]

    # make lookup table for mapping
    maxkey = max(remapdict.keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())

    upper_half = labels >> 16      # get upper half for instances
    lower_half = labels & 0xFFFF   # get lower half for semantics
    lower_half = remap_lut[lower_half]  # do the remapping of semantics
    labels = (upper_half << 16) + lower_half   # reconstruct full label
    labels = labels.astype(np.uint32)

    return labels.reshape(-1)

def grid_label2color(xyz, predicted_grid_labels):
    'Convert grid labels to point colors'
    xyz_pol = cart2polar(xyz)

    grid_ind = get_grid_ind(xyz_pol)
    predicted_labels = predicted_grid_labels[grid_ind[:,0], grid_ind[:,1], grid_ind[:,2]]

    predicted_colors = label2color(shift_labels(predicted_labels))

    return predicted_colors

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()