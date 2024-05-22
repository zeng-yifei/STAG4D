import os
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F

import rembg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cam_utils import orbit_camera, OrbitCamera
from gs_renderer_4d import Renderer, MiniCam
from dataset_4d import SparseDataset
from einops import rearrange, repeat
import torchvision.utils as vutils
import imageio

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


        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.t = 0
        self.time =0
        self.train_steps = 1  # steps per rendering loop
        
        self.path =self.opt.path
        if self.opt.size is not None:
            self.size = self.opt.size
        else:
            self.size = len(os.listdir(os.path.join(self.path,'ref')))
        self.frames=self.size


        # override if provide a checkpoint
        
        self.renderer.initialize(num_pts=5000)            


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

        # prepare embeddings

            #save_image_to_local(self.input_img_torch[0],'./valild2/ref_{}.jpg'.format(idx))
            #save_image_to_local(self.input_img_torch_batch[idx][0],'./valild2/batch_{}.jpg'.format(idx))
            #save_image_to_local(self.input_imgs_torch[0][0].detach(),'./valild2/ref0_{}.jpg'.format(idx))

    def prepare_train(self):

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
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
        self.set_fix_cam2()
        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")





        #self.renderer.gaussians.initialize_post_first_timestep()

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        
        self.stage='fine'
        
        if self.opt.load_step==None: 
            self.step=8000
        else:
            self.step = self.opt.load_step
        auto_path = os.path.join(self.opt.outdir,self.opt.save_path + str(self.step))
        #os.makedirs(auto_path,exist_ok=True)
        ply_path = os.path.join(auto_path,'model.ply')
        self.renderer.gaussians.load_model(auto_path)
        self.renderer.gaussians.load_ply(ply_path)


        self.renderer.gaussians.update_learning_rate(self.step)
        

        self.save_renderings(name='front')
        self.save_renderings(azim=180,name='back')
        self.save_renderings(azim=-30,name='front_moving',interval=2)
        self.save_renderings(azim=150,name='back_moving',interval=2)
        self.save_renderings(azim=0,name='round',interval=360//self.size)
        

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                 f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {tv_loss.item():.4f}tv_loss = {loss.item():.4f}\nzero123_loss = {zero123_loss.item():.4f}image_loss ={image_loss.item():.4f} ",
            )

    @torch.no_grad()
    def save_renderings(self, elev=0, azim=0, radius=2, name='front', interval=0):
        if interval==0:
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
                #os.makedirs(f'./valid/{self.opt.save_path}/final_{name}',exist_ok=True)
                #save_image_to_local(image[0].detach(),f'./valid/{self.opt.save_path}/final_{name}/{str(i).zfill(2)}.jpg')
            samples=torch.cat(images,dim=0)
            
            vid = (
                (rearrange(samples, "t c h w -> t h w c") * 255).clamp(0,255).detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            video_path = f'./valid/{self.opt.save_path}/video_{name}.mp4'
            imageio.mimwrite(video_path, vid)
        else:
            images=[]
            for i in np.arange(self.frames):
                
                pose = orbit_camera(elev, (azim+interval*i)%360, radius)
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
                #os.makedirs(f'./valid/{self.opt.save_path}/final_{name}',exist_ok=True)
                #save_image_to_local(image[0].detach(),f'./valid/{self.opt.save_path}/final_{name}/{str(i).zfill(2)}.jpg')
            samples=torch.cat(images,dim=0)
            
            vid = (
                (rearrange(samples, "t c h w -> t h w c") * 255).clamp(0,255).detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            video_path = f'./valid/{self.opt.save_path}/video_{name}.mp4'
            imageio.mimwrite(video_path, vid)
            
        
        
    def set_fix_cam2(self):
        self.fixed_cam2=[]
        self.fixed_elevation = []
        self.fixed_azimuth = []
        
        pose = orbit_camera(self.opt.elevation-30,30 , self.opt.radius)
        self.fixed_elevation.append(-30)
        self.fixed_azimuth.append(30)
        self.fixed_cam2.append(MiniCam(
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
        self.fixed_cam2.append(MiniCam(
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
        self.fixed_cam2.append(MiniCam(
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
        self.fixed_cam2.append(MiniCam(
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
        self.fixed_cam2.append(MiniCam(
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
        self.fixed_cam2.append(MiniCam(
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
            out = self.renderer.render(cur_cam, self.gaussain_scale_factor)

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

        # load image
        import glob
        self.input_imgs=[]
        self.input_masks=[]
        file_list = glob.glob(self.pattern)
        print(self.pattern,file_list)
        for files in sorted(file_list):
                    print(f"Reading file: {self.pattern}")
                   
                    print(f'[INFO] load image from {files}...')
                    img = cv2.imread(files, cv2.IMREAD_UNCHANGED)
                    if img.shape[-1] == 3:
                        if self.bg_remover is None:
                            self.bg_remover = rembg.new_session()
                        img = rembg.remove(img, session=self.bg_remover)

                    img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    img = img.astype(np.float32) / 255.0

                    self.input_mask = img[..., 3:]
                    # white bg
                    self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
                    # bgr to rgb
                    self.input_img = self.input_img[..., ::-1].copy()
                    
                    self.input_imgs.append(self.input_img)
                    self.input_masks.append(self.input_mask)
        
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

        # load prompt
        file_prompt = file.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            print(f'[INFO] load prompt from {file_prompt}...')
            with open(file_prompt, "r") as f:
                self.prompt = f.read().strip()
                

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
        self.prepare_train()

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
        gui.train(opt.iters)