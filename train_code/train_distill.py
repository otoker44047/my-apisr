# -*- coding: utf-8 -*-
import sys
import os
import torch


# Import important files
root_path = os.path.abspath('.')
sys.path.append(root_path)
from opt import opt
from architecture.rrdb import RRDBNet
from architecture.dat import DAT               
from train_code.train_master import train_master



# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

#from google's paper, distillation loss should be only one KL divergence.
#https://openaccess.thecvf.com/content/CVPR2022/papers/Beyer_Knowledge_Distillation_A_Good_Teacher_Is_Patient_and_Consistent_CVPR_2022_paper.pdf
class train_distill(train_master):
    def __init__(self, options, args) -> None:
        super().__init__(options, args, "distill")   # Pass a model name unique code


    def loss_init(self):
        # Prepare pixel loss
        self.pixel_loss_load()
        #raise NotImplementedError("no loss for distill yet")

    def call_model(self):
        # Generator Prepare (Don't formet torch.compile if needed)
        self.student_generator = RRDBNet(3, 3, scale=self.options['scale'], num_block=self.options['ESR_blocks_num']).cuda()
        # Generator: DAT light
        if opt['model_size'] == "light":
            # DAT light model 762K param
            self.teacher_generator = DAT(
                upscale=4,
                in_chans=3,
                img_size=64,
                img_range=1.,
                depth=[18],
                embed_dim=60,
                num_heads=[6],
                expansion_factor=2,
                resi_connection='3conv',
                split_size=[8,32],
                upsampler='pixelshuffledirect',
            ).cuda()
        
        elif opt['model_size'] == "small":
            # DAT small model 11.21M param
            self.teacher_generator = DAT(
                upscale=4,
                in_chans=3,
                img_size=64,
                img_range=1.,
                depth=[6,6,6,6,6,6],
                embed_dim=180,
                num_heads=[6,6,6,6,6,6],
                expansion_factor=2,
                resi_connection='1conv',
                split_size=[8,16],
                upsampler='pixelshuffledirect',
            ).cuda()

        else:
            raise NotImplementedError("We don't support such model size in DAT model")
        # self.generator = torch.compile(self.generator).cuda()
        self.student_generator.train()
        self.teacher_generator.train()

    
    def run(self):
        self.master_run()

    
    def calculate_loss(self, student_gen_hr, teacher_gen_hr, imgs_hr):

        # Generator pixel loss (l1 loss):  generated vs. GT
        teacher_l_g_pix = self.teacher_cri_pix(teacher_gen_hr, imgs_hr, self.batch_idx)
        self.weight_store["teacher_loss"] = teacher_l_g_pix 
        self.teacher_generator_loss += teacher_l_g_pix

        student_l_g_pix = self.student_cri_pix(student_gen_hr, imgs_hr, self.batch_idx)
        student_diff_pix = self.student_cri_distill(student_gen_hr, teacher_gen_hr, self.batch_idx)
        self.weight_store["student_loss"] = student_l_g_pix + student_diff_pix
        self.student_generator_loss += student_l_g_pix + student_diff_pix

    def tensorboard_report(self, iteration):
        # self.writer.add_scalar('Loss/train-Generator_Loss-Iteration', self.generator_loss, iteration)
        self.writer.add_scalar('Loss/Teacher_Loss-Iteration', self.weight_store["teacher_loss"], iteration)
        self.writer.add_scalar('Loss/Student_Loss-Iteration', self.weight_store["student_loss"], iteration)

