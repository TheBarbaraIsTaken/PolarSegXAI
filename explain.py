import yaml
import numpy as np
import torch
from dataloader.dataset import SemKITTI_label_name
from network.BEV_Unet import BEV_Unet
from network.ptBEV import ptBEVnet
import cv2
from pytorch_grad_cam import GradCAM
import os


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()


class GradCamPCDPolarNet:
    "Grad-Cam for semantic segmentation on pointcloud data with PolarNet and SemanticKITTI dataset"
    def __init__(self, scene, file, yaml_path="semantic-kitti.yaml",
                 device="cpu", grid_size=[480, 360, 32], circular_padding=True, fea_dim=9, 
                 model_save_path = 'pretrained_weight/SemKITTI_PolarSeg.pt'):
        self.scene = scene
        self.file = file
        self.yaml_path = yaml_path

        # yaml file
        with open(self.yaml_path, 'r') as stream:
            self.semkittiyaml = yaml.safe_load(stream)

        # Model parameters
        self.pytorch_device = torch.device(device)
        self.unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
        self.grid_size = grid_size
        self.compression_model = self.grid_size[2]
        self.circular_padding = circular_padding
        self.fea_dim = fea_dim
        self.model_save_path = model_save_path

        # File paths
        self.bin_file_path = f'data/sequences/{self.scene}/velodyne/{self.file}.bin'
        self.label_file = f"data/sequences/{self.scene}/labels/{self.file}.label"
        self.feature_path = f"out/predicted_labels/{scene}/{file}_features.npy"

        # Load point cloud data
        self.points = self.read_velodyne_bin(self.bin_file_path)
        self.xyz = self.points[:, :3]

        self.intensity = self.points[:, 3]
        self.intensity_colors = np.zeros((self.points.shape[0], 3))
        self.intensity_colors[:, 0] = self.intensity

        # Load labels
        self.true_labels = self.load_labels(self.label_file)
        self.true_colors = self.label2color(self.true_labels, self.semkittiyaml['color_map'])

        # Build model
        self.my_BEV_model=BEV_Unet(n_class=len(self.unique_label), n_height=self.compression_model, 
                                   input_batch_norm=True, dropout=0.5, circular_padding=self.circular_padding)
        self.my_model = ptBEVnet(self.my_BEV_model, pt_model='pointnet', grid_size=self.grid_size, fea_dim=self.fea_dim, max_pt_per_encode=256,
                                out_pt_fea_dim=512, kernal_size=1, pt_selection='random', fea_compre=self.compression_model, explain=True)
        if os.path.exists(model_save_path):
            self.my_model.load_state_dict(torch.load(self.model_save_path, map_location=torch.device('cpu')))  # Run on CPU

        self.my_model.to(self.pytorch_device)
        self.my_BEV_model.to(self.pytorch_device)
        self.my_model.eval()
        self.my_BEV_model.eval()
        
        # Grad-Cam parameters
        self.target_layers = [self.my_BEV_model.network.dropout]

    def calculate_grad_cam(self, target="road", colormap=cv2.COLORMAP_JET):
        # Get input values for feature extracting model
        self.pt_fea = np.load(self.feature_path)  # centered data on each voxel (3), polar coord (3), xyz (2), reflection value (1)
        self.val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(self.pytorch_device) for i in self.pt_fea]

        self.val_grid = [self.get_grid_ind(self.cart2polar(self.xyz))]
        self.val_grid_ten = [torch.from_numpy(i[:,:2]).to(self.pytorch_device) for i in self.val_grid]

        # Run a forward pass
        self.input_tensor = self.my_model(self.val_pt_fea_ten, self.val_grid_ten)
        self.output = self.my_BEV_model(self.input_tensor)

        print(self.output[0,:10,0,0,0])

        # Inputs for GradCam
        normalized_masks = torch.nn.functional.softmax(self.output, dim=1).cpu()

        sem_class_to_idx = {self.semkittiyaml['labels'][v]: k-1 for k,v in self.semkittiyaml['learning_map_inv'].items()}

        target_category = sem_class_to_idx[target]
        target_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
        target_mask_float = np.float32(target_mask == target_category)
        
        targets = [SemanticSegmentationTarget(target_category, target_mask_float)]

        with GradCAM(model=self.my_BEV_model, target_layers=self.target_layers) as cam:
            self.grayscale_cam = cam(input_tensor=self.input_tensor, targets=targets)[0]

        heatmap = cv2.applyColorMap(np.uint8(255 * self.grayscale_cam), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255

        cam = heatmap / np.max(heatmap)
        self.cam_image = np.uint8(255 * cam)
        self.heatmap_colors = self.cam_image[self.val_grid[0][:,0], self.val_grid[0][:,1],:]/255

        # [0,1]: low-high importance
        self.importances = self.grayscale_cam[self.val_grid[0][:,0], self.val_grid[0][:,1]]

        self.predict_labels = torch.argmax(self.output, dim=1)
        self.predict_labels = self.predict_labels.cpu().detach().numpy()[0]

        return self.predict_labels, self.importances, self.heatmap_colors

    @staticmethod
    def cart2polar(input_xyz):
        'Cartezian coordinates to polar'
        rho = np.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
        phi = np.arctan2(input_xyz[:,1],input_xyz[:,0])

        return np.stack((rho,phi,input_xyz[:,2]),axis=1)
    
    @staticmethod
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


    @staticmethod
    def read_velodyne_bin(bin_path):
        'Reading Velodyne .bin file'
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # Points are represented by (x, y, z, intensity)

        return points
    
    @staticmethod
    def load_labels(file_path):
        'Reading .label file'
        labels = np.fromfile(file_path, dtype=np.uint32)

        return labels & 0xFFFF  # SemanticKITTI uses the lower 16 bits for labels
    
    @staticmethod
    def label2color(labels, colormap, default=[255, 255, 255], color_model="bgr"):
        'Encode the lables to colors using the provided color map default is bgr'
        colors = np.zeros((labels.shape[0], 3))
        unique_labels = np.unique(labels)

        for label in unique_labels:
            colors[labels == label] = colormap.get(label, default)  # Default color if label not in colormap

        if color_model == "rgb":
            colors / 255.0
        elif color_model == "bgr":
            return colors[:, ::-1] / 255.0
        else: raise NotImplementedError




if __name__ == "__main__":
    grad_cam = GradCamPCDPolarNet(scene="08", file="000000")
    pred_labels, importance, heatmap = grad_cam.calculate_grad_cam()
    #print(grad_cam.true_colors.shape)
