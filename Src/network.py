# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    network.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <winshare@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/06/22 16:44:50 by tanwenxuan        #+#    #+#              #
#    Updated: 2020/06/23 20:28:58 by winshare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# Copyright 2020 tanwenxuan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



from Src.config import cfg

class network(cfg):
    def __init__(self):
        cfg.__init__(self)
        print("# ---------------------------------------------------------------------------- #")
        print("#                            network init start                                #")
        print("# ---------------------------------------------------------------------------- #")
        
        

        # ------------------------- network parameter decode ------------------------- #


        self.model=None
        if self.DefaultNetwork:
            print("# ===== \033[35m Use the default network :\033[32m ✓ \033[0m")
            self.model=self.DefaultNetworkDict[self.MissionType]
            self.model=self.init_model()
        else:
            print("# ===== \033[35m Use the default network :\033[31m x \033[0m")
            self.modeldict=self.network_library[self.MissionType]
            self.model=self.modeldict[self.NetType]
        




        print("# ===== \033[35m MissionType : \033[36m%s\033[0m "%self.MissionType)
        print("# ===== \033[35m Network Name: \033[36m%s\033[0m "%self.NetType)


        if self.debug:
            print("# ===== \033[35m Network function \033[36m%s\033[0m"%self.model)



        # -------------------------- model init & give parameter ---------------------- #

        # ---------------------------------- resume ---------------------------------- #
        if self.resume:
            checkpoint = torch.load(self.checkpoint, map_location='cpu')
            self.model_without_ddp.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.start_epoch = checkpoint['epoch'] + 1
        # ------------------------ init without pretrain model ----------------------- #
        if not self.download_pretrain_model:
            print("# ==== initialize_weights ")
            self._initialize_weights(self.model)






        # ---------------------------- optimizer instance ---------------------------- #
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer=self.optimizer(
            params,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        # --------------------------- lr scheduler instance -------------------------- #
        self.lr_scheduler=self.lr_scheduler(
            self.optimizer,
            milestones=self.lr_steps,
            gamma=self.lr_gamma
        )




        print("# ---------------------------------------------------------------------------- #")
        print("#                            network init done                                 #")
        print("# ---------------------------------------------------------------------------- #")

    def init_model(self):
        return self.model(
            pretrained=self.download_pretrain_model,
            num_classes=self.class_num,
            progress=True
        )
        

    def _initialize_weights(self,model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()