





class TRAINER():
    def __init__(self):
            
        """
        Instance Segmentation Output
            Train:
                The model returns a Dict[Tensor] during training, 
                containing :
                the classification regression losses for both the RPN and the R-CNN, 
                the mask loss.
            Validation:
                returns the post-processed predictions as a List[Dict[Tensor]] containing:
                * boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values of x between 0 and W and values of y between 0 and H
                * labels (Int64Tensor[N]): the predicted labels for each image
                * scores (Tensor[N]): the scores or each prediction
                * keypoints (FloatTensor[N, K, 3]): the locations of the predicted keypoints, in [x, y, v] format.

        Segmentation Output:
            Train：
                output segmentation target map ,we need use custom loss function to compute loss value,
                for 'backward()'function
            Validation：
                output segmentation target map
        Detection Output：
            Train：
                The model returns a Dict[Tensor] during training, 
                bbox loss
                classifier loss
            Validation：
                * boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values of x between 0 and W and values of y between 0 and H
                * labels (Int64Tensor[N]): the predicted labels for each image
                * scores (Tensor[N]): the scores or each prediction
        """
        print("# ---------------------------------------------------------------------------- #")
        print("#                                    TRAINER                                   #")
        print("# ---------------------------------------------------------------------------- #")

            
    def one_epoch(self,index):
        # ------------------------------ Train one epoch ----------------------------- #
        


        print("# ============================= train epoch {index} ========================== #".format(index=index))

        # --------------------------- General Epoch Module --------------------------- #

        self.model.train()
        bar=tqdm(self.trainloader,dynamic_ncols=True)
        for image,target in bar:
            if self.devices=="GPU":
                image,target=self.copy_to_gpu(image,target)
            
            # ------------------------------- Loss function ------------------------------ #
                
            if not self.MissionType=="Segmentation" or not self.DefaultNetwork:
                
                print(image)
                print(target)
                
                lossdict=self.model(image,target)
                print(lossdict)
                exit(0)

                losses = sum(loss for loss in lossdict.values())
                lossstr={k:v.item() for k,v in lossdict.items()}
        
            else:
                output=self.model(image)
                loss=self.Loss_Function(output,target)
                lossstr=loss.item()
        
            self.writer.add_scalars(self.NetType+'_Loss Function',lossstr,global_step=self.global_step)
        
            # ------------------------------ Inference Once ------------------------------ #
            
            
            # -------------------------------- Output Loss ------------------------------- #

            information="# Step : {step} |loss : {loss} |\n".format(step=self.global_step,loss=str(lossstr))
            bar.set_description(information)

            # --------------------------------- Backward --------------------------------- #

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            
            # -------------------------- Add log to Tensorboard -------------------------- #

            self.global_step+=1
            
            
