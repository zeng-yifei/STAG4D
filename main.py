import os
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from einops import rearrange, repeat
import imageio
import rembg

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer_4d import Renderer, MiniCam
from dataset_4d import SparseDataset

def save_image_to_local(image_tensor, file_path):
    # Ensure the image tensor is in the range [0, 1]
    image_tensor = image_tensor.clamp(0, 1) 

    # Save the image tensor to the specified file path
    vutils.save_image(image_tensor, file_path)

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5
        
        #self.use_depth = opt.use_depth

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.t = 0
        self.time = 0
        self.train_steps = 1  # steps per rendering loop
        self.init = True
        self.stage = 'coarse'
        self.path = self.opt.path
        self.save_step = self.opt.save_step
        
        if self.opt.size is not None:
            self.size = self.opt.size
        else:
            self.size = len(os.listdir(os.path.join(self.path,'ref')))
        self.frames=self.size
        self.dataset = SparseDataset(self.opt, self.size, H=self.H, W=self.W, device=self.device)
        self.dataloader =self.dataset.dataloader()
        self.iter = iter(self.dataloader)
        self.ref_view_batch, self.input_mask_batch,self.zero123_view_batch,self.zero123_masks_batch = next(self.iter)
        self.input_img_torch_batch,self.input_mask_torch_batch,self.zero123plus_imgs_torch_batch,self.zero123plus_masks_torch_batch=[],[],[],[]
        

        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt

        # override if provide a checkpoint
        
        self.renderer.initialize(num_pts=self.opt.num_pts)            

        self.point_nums = []
        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed
        
    def prepare_image(self,idx):
        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)
        
        self.zero123plus_imgs_torch=[]
        self.zero123plus_masks_torch=[]
        # input image
        if self.input_imgs is not None:
            for i in np.arange(6):
                #print(idx,i)
                self.input_img2_torch=(torch.from_numpy(self.input_imgs[i]).permute(2, 0, 1).unsqueeze(0).to(self.device))
                self.zero123plus_imgs_torch.append(F.interpolate(self.input_img2_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False))

                self.input_mask2_torch=torch.from_numpy(self.input_masks[i]).permute(2, 0, 1).unsqueeze(0).to(self.device)
                self.zero123plus_masks_torch.append(F.interpolate(self.input_mask2_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False))
                
        self.input_img_torch_batch.append(self.input_img_torch)
        self.input_mask_torch_batch.append(self.input_mask_torch)
        self.zero123plus_imgs_torch_batch.append(self.zero123plus_imgs_torch)
        self.zero123plus_masks_torch_batch.append(self.zero123plus_masks_torch)
        
        # prepare embeddings
        with torch.no_grad():
            self.guidance_zero123.get_img_embeds(self.input_img_torch, self.zero123plus_imgs_torch)


    def prepare_train(self):

        self.step = 0
        self.end_step = self.save_step+1
        
        ## given a load_path, load corresponding model
        if self.opt.load_path is not None:
           if self.opt.load_step is not None:
               self.step = self.opt.load_step
           else:
               #default loading save_step ply
               self.step = self.save_step 
           auto_path = os.path.join(self.opt.outdir,self.opt.load_path + str(self.step))

           ply_path = os.path.join(auto_path,'model.ply')
           self.renderer.gaussians.load_model(auto_path)
           self.renderer.gaussians.load_ply(ply_path)
           self.end_step =self.step+self.end_step
        
        ## setup training
        self.renderer.gaussians.training_setup(self.opt)
        
        ## do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer
        
        # default camera
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )
        self.set_fix_cam()
        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None


        print(f"[INFO] loading zero123...")
        from guidance.zero123_4d_utils import Zero123
        self.guidance_zero123 = Zero123(self.device)
        print(f"[INFO] loaded zero123!")

        ## load multiview reference images
        for i in np.arange(len(self.ref_view_batch)):
                self.input_img =   self.ref_view_batch[i]
                self.input_mask =  self.input_mask_batch[i]
                self.input_imgs =  self.zero123_view_batch[i]
                self.input_masks = self.zero123_masks_batch[i]
                self.prepare_image(i)


    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()


        
        torch.autograd.set_detect_anomaly(True)
        for _ in range(self.train_steps):

            if self.step<self.opt.init_steps:
                self.init = True
                self.stage = 'coarse'
            else:
                self.init = False
                self.stage = 'fine'
            
            if self.step == self.end_step:
                exit()
                
            ## save model
            if self.step == self.save_step:
                auto_path = os.path.join(self.opt.outdir,self.opt.save_path + str(self.step))
                os.makedirs(auto_path,exist_ok=True)
                ply_path = os.path.join(auto_path,'model.ply')
                self.renderer.gaussians.save_ply(ply_path)
                self.renderer.gaussians.save_deformation(auto_path)
                
            
            if self.step>self.opt.position_lr_max_steps:
                self.opt.position_lr_max_steps = self.opt.position_lr_max_steps2

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)
            viewspace_point_tensor_list = []
            radii_list = []
            visibility_filter_list = []
            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)
            self.guidance_zero123.update_step(0,self.step)

            loss = 0
            
            if self.step%self.opt.valid_interval == 0:
                self.save_renderings( 0, 0, 2 ,'front')
                self.save_renderings( 180, 0, 2 ,'back')
                

            render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)

            # avoid too large elevation (> 80 or < -80), and make sure it always cover [-30, 30]
            min_ver = max(min(-30, -30 - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(30, 30 - self.opt.elevation), 80 - self.opt.elevation)


            for _ in np.arange(self.opt.batch_size):
                
                #sample time
                if self.init:
                        self.t = self.frames//2
                        self.time = self.t/self.frames
                else:   
                        self.t = np.random.randint(0,self.frames)
                        self.time = self.t/self.frames

                self.input_img_torch =   self.input_img_torch_batch[self.t]
                self.input_mask_torch =  self.input_mask_torch_batch[self.t]
                self.zero123plus_imgs_torch =  self.zero123plus_imgs_torch_batch[self.t]
                self.zero123plus_masks_torch = self.zero123plus_masks_torch_batch[self.t]
                
                ## need to do rgb loss in the batch
                cur_cam = self.fixed_cam
                cur_cam.time=self.time
                
                out = self.renderer.render(cur_cam,stage=self.stage)
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]  
                radii_list.append(radii.unsqueeze(0))
                visibility_filter_list.append(visibility_filter.unsqueeze(0))
                viewspace_point_tensor_list.append(viewspace_point_tensor)
                # rgb loss
                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                image_loss =step_ratio* 20000*  F.mse_loss(image, self.input_img_torch)
                loss = loss + image_loss
                
                alpha = out["alpha"].unsqueeze(0)
                alpha_loss = step_ratio* 5000*  F.mse_loss(alpha, self.input_mask_torch)
                loss = loss + alpha_loss
                
                images = []
                poses = []

                vers_plus, hors_plus, radii_plus = [], [], []
                self.guidance_zero123.update_step(1,self.step)
                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = 0
                

                


                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                
                poses.append(pose)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                cur_cam.time=self.time
                if hor<30 and hor>-30 or np.random.rand()>0.4:
                    idx=None
                    vers_plus.append(torch.tensor(ver,device=self.device).unsqueeze(dim=0))
                    hors_plus.append(torch.tensor(hor,device=self.device).unsqueeze(dim=0))
                    radii_plus.append(torch.tensor(radius,device=self.device).unsqueeze(dim=0))
                elif hor>0:
                    idx=hor//60
                    vers_plus.append(torch.tensor(ver-self.fixed_elevation[idx],device=self.device).unsqueeze(dim=0))
                    hors_plus.append(torch.tensor(hor-self.fixed_azimuth[idx],device=self.device).unsqueeze(dim=0))
                    radii_plus.append(torch.tensor(radius,device=self.device).unsqueeze(dim=0))
                elif hor<0:
                    idx = (360+hor)//60
                    vers_plus.append(torch.tensor(ver-self.fixed_elevation[idx],device=self.device).unsqueeze(dim=0))
                    hors_plus.append(torch.tensor(hor-self.fixed_azimuth[idx],device=self.device).unsqueeze(dim=0))
                    radii_plus.append(torch.tensor(radius,device=self.device).unsqueeze(dim=0))

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                out = self.renderer.render(cur_cam, bg_color=bg_color,stage=self.stage)
                viewspace_point_tensor, visibility_filter, radii_rendering = out["viewspace_points"], out["visibility_filter"], out["radii"]  
                radii_list.append(radii_rendering.unsqueeze(0))
                visibility_filter_list.append(visibility_filter.unsqueeze(0))
                viewspace_point_tensor_list.append(viewspace_point_tensor)
                image = out["image"].unsqueeze(0)# [1, 3, H, W] in [0, 1]
                images.append(image)
                
                
            
                images_render = torch.cat(images, dim=0)
                #poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)
                vers_batch = torch.cat(vers_plus, dim=0).cpu().numpy()
                hors_batch = torch.cat(hors_plus, dim=0).cpu().numpy()
                radii_batch = torch.cat(radii_plus, dim=0).cpu().numpy()

                # guidance loss
                # as we have different reference views, so each time we only pass 1 image into zero123 for guidance
                zero123_loss = self.opt.lambda_zero123 * self.guidance_zero123.train_step(images_render, vers_batch, hors_batch, radii_batch, step_ratio,idx=idx,t = self.t)
                loss = loss + zero123_loss
            
            # tv loss
            scales = out['scales']
            tv_loss = self.renderer.gaussians.compute_regulation(self.opt.time_smoothness_weight, self.opt.plane_tv_weight, self.opt.l1_time_planes)
            loss += self.opt.lambda_tv * tv_loss
            
            # scale loss from physgaussian
            r = self.opt.scale_loss_ratio
            scale_loss = (torch.mean(torch.maximum(torch.max(scales,dim=1).values/ \
                                                  (torch.min(scales,dim=1).values+1e-8),\
                                                    torch.ones_like(torch.max(scales,dim=1).values)*r))-r) * scales.shape[0]
            loss += scale_loss

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
            for idx in range(0, len(viewspace_point_tensor_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            radii = torch.cat(radii_list,0).max(dim=0).values
            visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
                if self.step % self.opt.densification_interval == 1 :

                    self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold_percent, min_opacity=0.01, extent=1, max_screen_size=2)



        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                 f"step = {self.step: 5d} (+{self.train_steps: 2d})\n loss = {loss.item():.4f}\nzero123_loss = {zero123_loss.item():.4f}image_loss ={image_loss.item():.4f}\nloss_alpha = {alpha_loss.item():.4f} scale_loss:{scale_loss.item():.4f} ",
            )

    def set_fix_cam(self):
        self.fixed_cam_plus=[]
        self.fixed_elevation = []
        self.fixed_azimuth = []
        
        pose = orbit_camera(self.opt.elevation-30,30 , self.opt.radius)
        self.fixed_elevation.append(-30)
        self.fixed_azimuth.append(30)
        self.fixed_cam_plus.append(MiniCam(
                pose,
                self.opt.ref_size,
                self.opt.ref_size,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            ))
        
        pose = orbit_camera(self.opt.elevation+20, 90, self.opt.radius)
        self.fixed_elevation.append(20)
        self.fixed_azimuth.append(90)
        self.fixed_cam_plus.append(MiniCam(
                pose,
                self.opt.ref_size,
                self.opt.ref_size,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            ))
        pose = orbit_camera(self.opt.elevation-30, 150, self.opt.radius)
        self.fixed_elevation.append(-30)
        self.fixed_azimuth.append(150)
        self.fixed_cam_plus.append(MiniCam(
                pose,
                self.opt.ref_size,
                self.opt.ref_size,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            ))
        
        pose = orbit_camera(self.opt.elevation+20, 210, self.opt.radius)
        self.fixed_elevation.append(+20)
        self.fixed_azimuth.append(210)
        self.fixed_cam_plus.append(MiniCam(
                pose,
                self.opt.ref_size,
                self.opt.ref_size,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            ))
        
        pose = orbit_camera(self.opt.elevation-30, 270, self.opt.radius)
        self.fixed_elevation.append(-30)
        self.fixed_azimuth.append(270)
        self.fixed_cam_plus.append(MiniCam(
                pose,
                self.opt.ref_size,
                self.opt.ref_size,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            ))
        
        pose = orbit_camera(self.opt.elevation+20, 330, self.opt.radius)
        self.fixed_elevation.append(20)
        self.fixed_azimuth.append(330)
        self.fixed_cam_plus.append(MiniCam(
                pose,
                self.opt.ref_size,
                self.opt.ref_size,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            ))
        
    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                time=self.time
            )
            #print(cur_cam.time)
            out = self.renderer.render(cur_cam, self.gaussain_scale_factor,stage=self.stage)

            buffer_image = out[self.mode]  # [3, H, W]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(3, 1, 1)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            # display input_image
            if self.overlay_input_img and self.input_img is not None:
                self.buffer_image = (
                    self.buffer_image * (1 - self.overlay_input_img_ratio)
                    + self.input_img * self.overlay_input_img_ratio
                )

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!

    
    def load_input(self, file):
        # load image
        pass
        # load image

    @torch.no_grad()
    def save_renderings(self, elev=0, azim=0, radius=2, name='front'):
        images=[]
        for i in np.arange(self.frames):
            
            pose = orbit_camera(elev, azim, radius)
            cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
            )   
            cam.time=float(i/self.frames)
            out = self.renderer.render(cam,stage=self.stage)
            image = out["image"].unsqueeze(0)
            images.append(image)
            os.makedirs(f'./valid/{self.opt.save_path}/{self.step}_{name}',exist_ok=True)
            save_image_to_local(image[0].detach(),f'./valid/{self.opt.save_path}/{self.step}_{name}/{str(i).zfill(2)}.jpg')
        samples=torch.cat(images,dim=0)
        
        vid = (
            (rearrange(samples, "t c h w -> t h w c") * 255).clamp(0,255).detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        video_path = f'./valid/{self.opt.save_path}/{self.step}_{name}/video.mp4'
        imageio.mimwrite(video_path, vid)


    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            self.renderer.gaussians.save_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            self.renderer.gaussians.save_ply(path)

        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")
                
                # overlay stuff
                with dpg.group(horizontal=True):

                    def callback_toggle_overlay_input_img(sender, app_data):
                        self.overlay_input_img = not self.overlay_input_img
                        self.need_update = True

                    dpg.add_checkbox(
                        label="overlay image",
                        default_value=self.overlay_input_img,
                        callback=callback_toggle_overlay_input_img,
                    )

                    def callback_set_overlay_input_img_ratio(sender, app_data):
                        self.overlay_input_img_ratio = app_data
                        self.need_update = True

                    dpg.add_slider_float(
                        label="ratio",
                        min_value=0,
                        max_value=1,
                        format="%.1f",
                        default_value=self.overlay_input_img_ratio,
                        callback=callback_set_overlay_input_img_ratio,
                    )

                # prompt stuff
            
                dpg.add_input_text(
                    label="prompt",
                    default_value=self.prompt,
                    callback=callback_setattr,
                    user_data="prompt",
                )

                dpg.add_input_text(
                    label="negative",
                    default_value=self.negative_prompt,
                    callback=callback_setattr,
                    user_data="negative_prompt",
                )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        self.save_model(mode=user_data)

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_button(
                        label="geo",
                        tag="_button_save_mesh",
                        callback=callback_save,
                        user_data='geo',
                    )
                    dpg.bind_item_theme("_button_save_mesh", theme_button)

                    dpg.add_button(
                        label="geo+tex",
                        tag="_button_save_mesh_with_tex",
                        callback=callback_save,
                        user_data='geo+tex',
                    )
                    dpg.bind_item_theme("_button_save_mesh_with_tex", theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    # dpg.add_button(
                    #     label="init", tag="_button_init", callback=self.prepare_train
                    # )
                    # dpg.bind_item_theme("_button_init", theme_button)

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

                def callback_set_gaussain_scale(sender, app_data):
                    self.gaussain_scale_factor = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label="gaussain scale",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                    default_value=self.gaussain_scale_factor,
                    callback=callback_set_gaussain_scale,
                )

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="Gaussian3D",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def train(self, iters=500):
        
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
            # do a last prune
            #self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
            

            
            
        # save
        self.save_model(mode='model')
        self.save_model(mode='geo+tex')
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)

    if opt.gui:
        gui.render()
    else:
        gui.train(opt.save_step+1)