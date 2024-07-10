import yaml
import numpy as np
import torch
from dataloader.dataset import SemKITTI_label_name
from network.BEV_Unet import BEV_Unet
from network.ptBEV import ptBEVnet
import cv2
from pytorch_grad_cam import GradCAM
import os
#import open3d as o3d
from sklearn.cluster import DBSCAN
from time import time

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
    def __init__(self, scene=None, file=None, yaml_path="semantic-kitti.yaml", load_scene=True,
                 device="cpu", grid_size=[480, 360, 32], circular_padding=True, fea_dim=9, 
                 model_save_path='pretrained_weight/SemKITTI_PolarSeg.pt'):
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

        if load_scene:
            if self.scene is None or self.file is None:
                raise ValueError("There isn't any file or scene to load")
            
            self.load_scene()

    def load_new_scene(self, scene, file):
        self.scene = scene
        self.file = file

        # File paths
        self.bin_file_path = f'data/sequences/{self.scene}/velodyne/{self.file}.bin'
        self.label_file = f"data/sequences/{self.scene}/labels/{self.file}.label"
        self.feature_path = f"out/predicted_labels/{self.scene}/{file}_features.npy"

        # Load point cloud data
        self.points = self.read_velodyne_bin(self.bin_file_path)
        self.xyz = self.points[:, :3]

        self.intensity = self.points[:, 3]
        self.intensity_colors = np.zeros((self.points.shape[0], 3))
        self.intensity_colors[:, 0] = self.intensity

        # Load labels
        self.true_labels = self.load_labels(self.label_file)
        self.true_colors = self.label2color(self.true_labels, self.semkittiyaml['color_map'])

    def load_scene(self):
        self.load_new_scene(self.scene, self.file)

    def calculate_grad_cam(self, target="road", colormap=cv2.COLORMAP_JET):
        # Get input values for feature extracting model
        self.pt_fea = np.load(self.feature_path)  # centered data on each voxel (3), polar coord (3), xyz (2), reflection value (1)
        self.val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(self.pytorch_device) for i in self.pt_fea]

        self.val_grid = [self.get_grid_ind(self.cart2polar(self.xyz))]
        self.val_grid_ten = [torch.from_numpy(i[:,:2]).to(self.pytorch_device) for i in self.val_grid]

        # Run a forward pass
        self.input_tensor = self.my_model(self.val_pt_fea_ten, self.val_grid_ten)
        self.output = self.my_BEV_model(self.input_tensor)

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

    @staticmethod
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

class ExplainTarget:
    def __init__(self, target="car", scene="08"):
        self.grad_cam = GradCamPCDPolarNet(load_scene=False)
        self.target = target
        self.scene = scene

    def pcd_stat(self, n_file=5):
        s = 0
        for f in range(0, n_file):
            start = time()
            f = str(f)
            file = str(f.rjust(6, "0"))

            label_file = f"data/sequences/{self.scene}/labels/{file}.label"
            true_labels = GradCamPCDPolarNet.load_labels(label_file)

            s += len(true_labels)
        return s



    def stats(self, n_file=5):
        sem_class_to_idx = {v: k for k,v in self.grad_cam.semkittiyaml['labels'].items()}
        target_id = sem_class_to_idx[self.target]
        print(target_id)

        # 2) importance predicted label = target: (avg, std) label != target: (avg, std)
        self.predicted_stats = []
        self.max_stats = []

        # 4) importances for the 4 region
        self.confusion_stats = []

        # 3) What predicted instad of the correct label
        self.fp_instead_stats = {k: 0 for k,_ in self.grad_cam.semkittiyaml['labels'].items()}
        self.fn_instead_stats = {k: 0 for k,_ in self.grad_cam.semkittiyaml['labels'].items()}

        # 5) Total predicted label, total true label
        self.total_pred = 0
        self.total_true = 0

        # 6) accuracy
        self.total_tp = 0


        for f in range(0, n_file):
            start = time()
            f = str(f)
            file = str(f.rjust(6, "0"))

            label_file = f"data/sequences/{self.scene}/labels/{file}.label"
            label_output_path = f"out/predicted_labels/{self.scene}/{file}.npy"
            bin_file_path = f'data/sequences/{self.scene}/velodyne/{file}.bin'
            importance_path = f"xai/importance_{self.target}_{file}.npy"

            points = GradCamPCDPolarNet.read_velodyne_bin(bin_file_path)
            xyz = points[:, :3]
            xyz_pol = GradCamPCDPolarNet.cart2polar(xyz)

            grid_ind = GradCamPCDPolarNet.get_grid_ind(xyz_pol)
            predicted_grid_labels = np.load(label_output_path)

            predicted_labels = GradCamPCDPolarNet.shift_labels(predicted_grid_labels[grid_ind[:,0], grid_ind[:,1], grid_ind[:,2]])
            true_labels = GradCamPCDPolarNet.load_labels(label_file)
            importances = np.load(importance_path)

            # 2)
            pred_target_mask = predicted_labels == target_id
            self.predicted_stats.append((np.mean(importances[pred_target_mask]), np.std(importances[pred_target_mask]),
                                        np.mean(importances[~pred_target_mask]), np.std(importances[~pred_target_mask])))
            
            self.max_stats.append((np.max(importances[pred_target_mask]), np.max(importances[~pred_target_mask])))

            # 4)
            tp_mask = (true_labels == target_id) & (predicted_labels == target_id)
            tn_mask = (true_labels != target_id) & (predicted_labels != target_id)
            fp_mask = (true_labels != target_id) & (predicted_labels == target_id)
            fn_mask = (true_labels == target_id) & (predicted_labels != target_id)

            tp_importance = importances[tp_mask]
            tn_importance = importances[tn_mask]
            fp_importance = importances[fp_mask]
            fn_importance = importances[fn_mask]


            self.confusion_stats.append((tp_importance.sum(), len(tp_importance),
                                         tn_importance.sum(), len(tn_importance),
                                         fp_importance.sum(), len(fp_importance),
                                         fn_importance.sum(), len(fn_importance)))
            
            # 3)
            unique_fp, counts_fp = np.unique(true_labels[fp_mask], return_counts=True)
            unique_fn, counts_fn = np.unique(predicted_labels[fn_mask], return_counts=True)

            for i, label in enumerate(unique_fp):
                self.fp_instead_stats[label] += counts_fp[i]

            for i, label in enumerate(unique_fn):
                self.fn_instead_stats[label] += counts_fn[i]

            # 5)
            self.total_pred += (predicted_labels == target_id).sum()
            self.total_true += (true_labels == target_id).sum()

            # 6)
            self.total_tp += tp_mask.sum()


            end = time()
            print(f"{f:<3}=> {end-start}")

        self.predicted_stats, self.confusion_stats = np.array(self.predicted_stats), np.array(self.confusion_stats)

        return self.max_stats, self.predicted_stats, self.confusion_stats, self.fp_instead_stats, self.fn_instead_stats, (self.total_pred, self.total_true, self.total_tp)

    def find_objects(self, n_file=5, eps=2, min_samples=15):
        sem_class_to_idx = {v: k for k,v in self.grad_cam.semkittiyaml['labels'].items()}
        target_id = sem_class_to_idx[self.target]

        self.samples = dict()
        for f in range(0, n_file):
            print("START")
            start = time()
            f = str(f)
            file = str(f.rjust(6, "0"))
            self.samples[file] = {"fn": dict(), "tp": dict()}

            label_file = f"data/sequences/{self.scene}/labels/{file}.label"
            label_output_path = f"out/predicted_labels/{self.scene}/{file}.npy"
            bin_file_path = f'data/sequences/{self.scene}/velodyne/{file}.bin'

            points = GradCamPCDPolarNet.read_velodyne_bin(bin_file_path)
            xyz = points[:, :3]
            xyz_pol = GradCamPCDPolarNet.cart2polar(xyz)

            grid_ind = GradCamPCDPolarNet.get_grid_ind(xyz_pol)

            predicted_grid_labels = np.load(label_output_path)
            predicted_labels = GradCamPCDPolarNet.shift_labels(predicted_grid_labels[grid_ind[:,0], grid_ind[:,1], grid_ind[:,2]])
            true_labels = GradCamPCDPolarNet.load_labels(label_file)

            mask = true_labels == target_id
            xyz_tmp = xyz[mask]
            predicted_labels_tmp = predicted_labels[mask]

            # Find objects
            #min_samples = 40  # for road
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz_tmp)

            cluster_labels = clustering.labels_
            n_clusters = np.max(cluster_labels)

            # Grad cam
            # self.grad_cam.load_new_scene(self.scene, file)
            # pred_labels, importance, heatmap = self.grad_cam.calculate_grad_cam()
            # self.samples[file]["importances"] = importance

            end = time()
            print("END")
            print(end-start)


            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                predicted_labels_cluster = predicted_labels_tmp[cluster_mask]

                tp_mask = predicted_labels_cluster == target_id
                tp_sum = tp_mask.sum()
                fn_mask = ~tp_mask
                fn_sum = fn_mask.sum()

                is_tp = tp_sum > fn_sum

                if not is_tp:
                    id = len(self.samples[file]["fn"])
                    self.samples[file]["fn"][id] = {
                        "cluster_mask": cluster_mask,
                        "true": tp_sum,
                        "false": fn_sum
                    }

                    print(file, tp_sum > fn_sum, tp_sum, fn_sum)
                else:
                    id = len(self.samples[file]["tp"])
                    self.samples[file]["tp"][id] = {
                        "cluster_mask": cluster_mask,
                        "true": tp_sum,
                        "false": fn_sum
                    }

                # print((true_labels_cluster != target_id).sum())

            #print(n_clusters)
        return self.samples

        # self.best_file_tp = "00"
        # self.best_mask_tp = np.array([])
        # best_sum_tp = 0
        # self.best_file_tn = "00"
        # self.best_mask_tn = np.array([])
        # best_sum_tn = 0
        # self.best_file_fp = "00"
        # self.best_mask_fp = np.array([])
        # best_sum_fp = 0
        # self.best_file_fn = "00"
        # self.best_mask_fn = np.array([])
        # best_sum_fn = 0

        # tp_pcd = o3d.geometry.PointCloud()
        # tn_pcd = o3d.geometry.PointCloud()
        # fp_pcd = o3d.geometry.PointCloud()
        # fn_pcd = o3d.geometry.PointCloud()
            

        #     tp_mask = (true_labels == target_id) & (predicted_labels == target_id)
        #     tp_sum = tp_mask.sum()
        #     tn_mask = (true_labels != target_id) & (predicted_labels != target_id)
        #     tn_sum = tn_mask.sum()
        #     fp_mask = (true_labels != target_id) & (predicted_labels == target_id)
        #     fp_sum = fp_mask.sum()
        #     fn_mask = (true_labels == target_id) & (predicted_labels != target_id)
        #     fn_sum = fn_mask.sum()

        #     pcd.points = o3d.utility.Vector3dVector(xyz)
        #     cluster_labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))
        #     max_label = cluster_labels.max()
        #     print(f"point cloud has {max_label + 1} clusters")


        #     # tp_pcd.points = o3d.utility.Vector3dVector(xyz[tp_mask,:])
        #     # tn_pcd.points = o3d.utility.Vector3dVector(xyz[tn_mask,:])
        #     # fp_pcd.points = o3d.utility.Vector3dVector(xyz[fp_mask,:])
        #     # fn_pcd.points = o3d.utility.Vector3dVector(xyz[fn_mask,:])


        #     # if (tp_sum > best_sum_tp):
        #     #     best_sum_tp = tp_sum
        #     #     self.best_mask_tp = tp_mask.copy()
        #     #     self.best_file_tp = file
            
        #     # if (tn_sum > best_sum_tn):
        #     #     best_sum_tn = tn_sum
        #     #     self.best_mask_tn = tn_mask.copy()

        #     #     self.best_file_tn = file
            
        #     # if (fp_sum > best_sum_fp):
        #     #     best_sum_fp = fp_sum
        #     #     self.best_mask_fp = fp_mask.copy()
        #     #     self.best_file_fp = file
            
        #     # if (fn_sum > best_sum_fn):
        #     #     best_sum_fn = fn_sum
        #     #     self.best_mask_fn = fn_mask.copy()
        #     #     self.best_file_fn = file

        # return ((self.best_file_tp, self.best_mask_tp), (self.best_file_tn, self.best_mask_tn), 
        #         (self.best_file_fp, self.best_mask_fp),  (self.best_file_fn, self.best_mask_fn))


if __name__ == "__main__":
    # Test
    # grad_cam = GradCamPCDPolarNet(scene="08", file="000000")
    # pred_labels, importance, heatmap = grad_cam.calculate_grad_cam()
    # print(importance.shape)

    xai = ExplainTarget()
    xai.stats()
    # out = xai.find_objects()
    # tp, tn, fp, fn = out
    # print(tp[1].sum(), tn[1].sum(), fp[1].sum(), fn[1].sum())
    # print(tp[0], tn[0], fp[0], fn[0])


    # Save all gradcam importances
    # grad_cam = GradCamPCDPolarNet(load_scene=False)
    # target = "car"

    # for i in range(0,501):
    #     i = str(i)
    #     file = str(i.rjust(6, "0"))
        
    #     start = time()
    #     grad_cam.load_new_scene("08", file)
    #     _, importances, _ = grad_cam.calculate_grad_cam(target=target)
    #     end = time()

    #     print(f"{i:<3}=> {end-start}")

    #     np.save(f"xai/importance_{target}_{file}.npy", importances)
    pass
