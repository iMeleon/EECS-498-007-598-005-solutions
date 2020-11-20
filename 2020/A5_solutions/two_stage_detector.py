"""
If this solution was useful to you and you want to thank me, please give me a star:
https://github.com/iMeleon/EECS-498-007-598-005-solutions
"""
import time
import math
import torch 
import torch.nn as nn
from torch import optim
import torchvision
from a5_helper import *
import matplotlib.pyplot as plt
from single_stage_detector import GenerateAnchor, GenerateProposal, IoU


def hello_two_stage_detector():
    print("Hello from two_stage_detector.py!")
def GenerateProposal(anchors, offsets, method='YOLO'):
  """
  Proposal generator.

  Inputs:
  - anchors: Anchor boxes, of shape (B, A, H', W', 4). Anchors are represented
    by the coordinates of their top-left and bottom-right corners.
  - offsets: Transformations of shape (B, A, H', W', 4) that will be used to
    convert anchor boxes into region proposals. The transformation
    offsets[b, a, h, w] = (tx, ty, tw, th) will be applied to the anchor
    anchors[b, a, h, w]. For YOLO, assume that tx and ty are in the range
    (-0.5, 0.5).
  - method: Which transformation formula to use, either 'YOLO' or 'FasterRCNN'
  
  Outputs:
  - proposals: Region proposals of shape (B, A, H', W', 4), represented by the
    coordinates of their top-left and bottom-right corners. Applying the
    transform offsets[b, a, h, w] to the anchor [b, a, h, w] should give the
    proposal proposals[b, a, h, w].
  
  """
  assert(method in ['YOLO', 'FasterRCNN'])
  proposals = None
  ##############################################################################
  # TODO: Given anchor coordinates and the proposed offset for each anchor,    #
  # compute the proposal coordinates using the transformation formulas above.  #
  ##############################################################################
  # Replace "pass" statement with your code
  B, A, H, W, _ = anchors.shape
  proposals = torch.zeros(B,A,H,W,5,device='cuda')
  final_proposals = []
  for b in range(B):
    for a in range(A):
      for h in range(H):
        for w in range(W):
          x_tl,y_tl,x_br,y_br = anchors[b,a,h,w]
          tx, ty, tw, th = offsets[b,a,h,w]

          width = x_br - x_tl
          height = y_br - y_tl
          x_center = x_br - width/2
          y_center = y_br - height/2
          if method == 'FasterRCNN':
            x_center = x_center + tx*width
            y_center= y_center + ty*height
          else:
            x_center = x_center + tx
            y_center= y_center + ty
          width = width*torch.exp(tw)
          height = height*torch.exp(th)

          x_tl = x_center - width/2
          y_tl = y_center - height/2
          x_br = x_center + width/2
          y_br = y_center + height/2
          proposals[b,a,h,w] = torch.tensor([b,x_tl,y_tl,x_br,y_br])

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return proposals

class ProposalModule(nn.Module):
  def __init__(self, in_dim, hidden_dim=256, num_anchors=9, drop_ratio=0.3):
    super().__init__()

    assert(num_anchors != 0)
    self.num_anchors = num_anchors
    ##############################################################################
    # TODO: Define the region proposal layer - a sequential module with a 3x3    #
    # conv layer, followed by a Dropout (p=drop_ratio), a Leaky ReLU and         #
    # a 1x1 conv.                                                                #
    # HINT: The output should be of shape Bx(Ax6)x7x7, where A=self.num_anchors. #
    #       Determine the padding of the 3x3 conv layer given the output dim.    #
    ##############################################################################
    # Make sure that your region proposal module is called pred_layer
    # 6 = (4+2P) .
    self.model = torch.nn.Sequential(
    torch.nn.Conv2d(in_dim, hidden_dim, 3,padding=1),
    torch.nn.Dropout(drop_ratio),
    torch.nn.LeakyReLU(),
    torch.nn.Conv2d(hidden_dim, int(num_anchors*6), 1)
    )      
    # Replace "pass" statement with your code
      ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

  def _extract_anchor_data(self, anchor_data, anchor_idx):
    """
    Inputs:
    - anchor_data: Tensor of shape (B, A, D, H, W) giving a vector of length
      D for each of A anchors at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving anchor indices to extract

    Returns:
    - extracted_anchors: Tensor of shape (M, D) giving anchor data for each
      of the anchors specified by anchor_idx.
    """
    B, A, D, H, W = anchor_data.shape
    # print(B, A, D, H, W)
    anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
    # print(anchor_data.shape)
    # print(anchor_idx)
    extracted_anchors = anchor_data[anchor_idx]
    return extracted_anchors

  def forward(self, features, pos_anchor_coord=None, \
              pos_anchor_idx=None, neg_anchor_idx=None):
    """
    Run the forward pass of the proposal module.

    Inputs:
    - features: Tensor of shape (B, in_dim, H', W') giving features from the
      backbone network.
    - pos_anchor_coord: Tensor of shape (M, 4) giving the coordinates of
      positive anchors. Anchors are specified as (x_tl, y_tl, x_br, y_br) with
      the coordinates of the top-left corner (x_tl, y_tl) and bottom-right
      corner (x_br, y_br). During inference this is None.
    - pos_anchor_idx: int64 Tensor of shape (M,) giving the indices of positive
      anchors. During inference this is None.
    - neg_anchor_idx: int64 Tensor of shape (M,) giving the indicdes of negative
      anchors. During inference this is None.

    The outputs from this module are different during training and inference.
    
    During training, pos_anchor_coord, pos_anchor_idx, and neg_anchor_idx are
    all provided, and we only output predictions for the positive and negative
    anchors. During inference, these are all None and we must output predictions
    for all anchors.

    Outputs (during training):
    - conf_scores: Tensor of shape (2M, 2) giving the classification scores
      (object vs background) for each of the M positive and M negative anchors.
    - offsets: Tensor of shape (M, 4) giving predicted transforms for the
      M positive anchors.
    - proposals: Tensor of shape (M, 4) giving predicted region proposals for
      the M positive anchors.

    Outputs (during inference):
    - conf_scores: Tensor of shape (B, A, 2, H', W') giving the predicted
      classification scores (object vs background) for all anchors
    - offsets: Tensor of shape (B, A, 4, H', W') giving the predicted transforms
      for all anchors
    """
    if pos_anchor_coord is None or pos_anchor_idx is None or neg_anchor_idx is None:
      mode = 'eval'
    else:
      mode = 'train'
    conf_scores, offsets, proposals = None, None, None
    ############################################################################
    # TODO: Predict classification scores (object vs background) and transforms#
    # for all anchors. During inference, simply output predictions for all     #
    # anchors. During training, extract the predictions for only the positive  #
    # and negative anchors as described above, and also apply the transforms to#
    # the positive anchors to compute the coordinates of the region proposals. #
    #                                                                          #
    # HINT: You can extract information about specific proposals using the     #
    # provided helper function self._extract_anchor_data.                      #
    # HINT: You can compute proposal coordinates using the GenerateProposal    #
    # function from the previous notebook.                                     #
    ############################################################################
    # Replace "pass" statement with your code
    if mode == 'eval':
      out  = self.model(features)
      B, A, H, W = out.shape
      out = out.reshape(B,self.num_anchors,-1,H,W)
      conf_scores = out[:,:,:2]
      offsets =  out[:,:,2:]
    else:
      out  = self.model(features)
      B, A, H, W = out.shape
      pos_anchors = self._extract_anchor_data(out.reshape(B,self.num_anchors,-1,H,W),pos_anchor_idx)
      neg_anchors = self._extract_anchor_data(out.reshape(B,self.num_anchors,-1,H,W),neg_anchor_idx)
      conf_scores = conf_scores = torch.cat((pos_anchors[:,:2], neg_anchors[:,:2]), 0)
      offsets = pos_anchors[:,2:]
      proposals = GenerateProposal(pos_anchors[:,2:].reshape(pos_anchor_coord.shape[0],1,1,1,4),offsets.reshape(pos_anchor_coord.shape[0],1,1,1,4))
      proposals = proposals.reshape(-1,5)
      
      # print(proposals.shape)
    ###########################1###################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    if mode == 'train':
      return conf_scores, offsets, proposals
    elif mode == 'eval':
      return conf_scores, offsets


def ConfScoreRegression(conf_scores, batch_size):
  """
  Binary cross-entropy loss

  Inputs:
  - conf_scores: Predicted confidence scores, of shape (2M, 2). Assume that the
    first M are positive samples, and the last M are negative samples.

  Outputs:
  - conf_score_loss: Torch scalar
  """
  # the target conf_scores for positive samples are ones and negative are zeros
  M = conf_scores.shape[0] // 2
  GT_conf_scores = torch.zeros_like(conf_scores)
  GT_conf_scores[:M, 0] = 1.
  GT_conf_scores[M:, 1] = 1.

  conf_score_loss = nn.functional.binary_cross_entropy_with_logits(conf_scores, GT_conf_scores, \
                                     reduction='sum') * 1. / batch_size
  return conf_score_loss


def BboxRegression(offsets, GT_offsets, batch_size):
  """"
  Use SmoothL1 loss as in Faster R-CNN

  Inputs:
  - offsets: Predicted box offsets, of shape (M, 4)
  - GT_offsets: GT box offsets, of shape (M, 4)
  
  Outputs:
  - bbox_reg_loss: Torch scalar
  """
  bbox_reg_loss = nn.functional.smooth_l1_loss(offsets, GT_offsets, reduction='sum') * 1. / batch_size
  return bbox_reg_loss


class RPN(nn.Module):
  def __init__(self):
    super().__init__()

    # READ ONLY
    self.anchor_list = torch.tensor([[1., 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]])
    self.feat_extractor = FeatureExtractor()
    self.prop_module = ProposalModule(1280, num_anchors=self.anchor_list.shape[0])

  def forward(self, images, bboxes, output_mode='loss'):
    """
    Training-time forward pass for the Region Proposal Network.

    Inputs:
    - images: Tensor of shape (B, 3, 224, 224) giving input images
    - bboxes: Tensor of ground-truth bounding boxes, returned from the DataLoader
    - output_mode: One of 'loss' or 'all' that determines what is returned:
      If output_mode is 'loss' then the output is:
      - total_loss: Torch scalar giving the total RPN loss for the minibatch
      If output_mode is 'all' then the output is:
      - total_loss: Torch scalar giving the total RPN loss for the minibatch
      - pos_conf_scores: Tensor of shape (M, 1) giving the object classification
        scores (object vs background) for the positive anchors
      - proposals: Tensor of shape (M, 4) giving the coordiantes of the region
        proposals for the positive anchors
      - features: Tensor of features computed from the backbone network
      - GT_class: Tensor of shape (M,) giving the ground-truth category label
        for the positive anchors.
      - pos_anchor_idx: Tensor of shape (M,) giving indices of positive anchors
      - neg_anchor_idx: Tensor of shape (M,) giving indices of negative anchors
      - anc_per_image: Torch scalar giving the number of anchors per image.
    
    Outputs: See output_mode

    HINT: The function ReferenceOnActivatedAnchors from the previous notebook
    can compute many of these outputs -- you should study it in detail:
    - pos_anchor_idx (also called activated_anc_ind)
    - neg_anchor_idx (also called negative_anc_ind)
    - GT_class
    """
    # weights to multiply to each loss term
    w_conf = 1 # for conf_scores
    w_reg = 5 # for offsets

    assert output_mode in ('loss', 'all'), 'invalid output mode!'
    total_loss = None
    conf_scores, proposals, features, GT_class, pos_anchor_idx, anc_per_img = \
      None, None, None, None, None, None
    ##############################################################################
    # TODO: Implement the forward pass of RPN.                                   #
    # A few key steps are outlined as follows:                                   #
    # i) Image feature extraction,                                               #
    # ii) Grid and anchor generation,                                            #
    # iii) Compute IoU between anchors and GT boxes and then determine activated/#
    #      negative anchors, and GT_conf_scores, GT_offsets, GT_class,           #
    # iv) Compute conf_scores, offsets, proposals through the region proposal    #
    #     module                                                                 #
    # v) Compute the total_loss for RPN which is formulated as:                  #
    #    total_loss = w_conf * conf_loss + w_reg * reg_loss,                     #
    #    where conf_loss is determined by ConfScoreRegression, w_reg by          #
    #    BboxRegression. Note that RPN does not predict any class info.          #
    #    We have written this part for you which you've already practiced earlier#
    # HINT: Do not apply thresholding nor NMS on the proposals during training   #
    #       as positive/negative anchors have been explicitly targeted.          #
    ##############################################################################
    # Replace "pass" statement with your code
    out = self.feat_extractor(images)
    batch_size = out.shape[0]
    grid = GenerateGrid(images.shape[0])
    anchors = GenerateAnchor(self.anchor_list, grid)
    iou_mat =  IoU(anchors, bboxes)
    activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, activated_anc_coord, negative_anc_coord = ReferenceOnActivatedAnchors(anchors, bboxes, grid, iou_mat,neg_thresh=0.2)
    conf_scores, offsets, proposals =  self.prop_module(out,activated_anc_coord,activated_anc_ind, negative_anc_ind)
    conf_loss = ConfScoreRegression(conf_scores, batch_size)
    reg_loss = BboxRegression(offsets, GT_offsets, batch_size)
    anc_per_img = anchors.shape[1]*anchors.shape[2]*anchors.shape[3]
    total_loss = w_conf * conf_loss + w_reg * reg_loss 
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    if output_mode == 'loss':
      return total_loss
    else:
      return total_loss, conf_scores, proposals, out, GT_class, pos_anchor_idx, anc_per_img


  def inference(self, images, thresh=0.5, nms_thresh=0.7, mode='RPN',return_withclass= False):
    """
    Inference-time forward pass for the Region Proposal Network.

    Inputs:
    - images: Tensor of shape (B, 3, H, W) giving input images
    - thresh: Threshold value on confidence scores. Proposals with a predicted
      object probability above thresh should be kept. HINT: You can convert the
      object score to an object probability using a sigmoid nonlinearity.
    - nms_thresh: IoU threshold for non-maximum suppression
    - mode: One of 'RPN' or 'FasterRCNN' to determine the outputs.

    The region proposal network can output a variable number of region proposals
    per input image. We assume that the input image images[i] gives rise to
    P_i final propsals after thresholding and NMS.

    NOTE: NMS is performed independently per-image!

    Outputs:
    - final_proposals: List of length B, where final_proposals[i] is a Tensor
      of shape (P_i, 4) giving the coordinates of the predicted region proposals
      for the input image images[i].
    - final_conf_probs: List of length B, where final_conf_probs[i] is a
      Tensor of shape (P_i,) giving the predicted object probabilities for each
      predicted region proposal for images[i]. Note that these are
      *probabilities*, not scores, so they should be between 0 and 1.
    - features: Tensor of shape (B, D, H', W') giving the image features
      predicted by the backbone network for each element of images.
      If mode is "RPN" then this is a dummy list of zeros instead.
    """
    assert mode in ('RPN', 'FasterRCNN'), 'invalid inference mode!'

    features, final_conf_probs, final_proposals = None, None, None
    ##############################################################################
    # TODO: Predicting the RPN proposal coordinates `final_proposals` and        #
    # confidence scores `final_conf_probs`.                                     #
    # The overall steps are similar to the forward pass but now you do not need  #
    # to decide the activated nor negative anchors.                              #
    # HINT: Threshold the conf_scores based on the threshold value `thresh`.     #
    # Then, apply NMS to the filtered proposals given the threshold `nms_thresh`.#
    # HINT: Use `torch.no_grad` as context to speed up the computation.          #
    ##############################################################################
    # Replace "pass" statement with your code
    #features = torch.zeros()
    with torch.no_grad():
      out = self.feat_extractor(images)
      N = images.shape[0]
      batch_size = out.shape[0]
      if mode == "FasterRCNN":
        features = out
      grid = GenerateGrid(images.shape[0])
      anchors = GenerateAnchor(self.anchor_list, grid)
      final_proposals = []
      final_conf_probs = []
      conf_scores, offsets =  self.prop_module(out)
      offsets = offsets.permute(0,1,3,4,2)
      offsets = GenerateProposal(anchors, offsets, method='YOLO')
      offsets = offsets.permute(0,1,4,2,3)
      for n in range(N):
        local_conf_scores = conf_scores[n]
        local_offsets =  offsets[n]
        cat = torch.cat((local_conf_scores, local_offsets), 1)
        cat[:,0] =  torch.sigmoid(cat[:,0])
        cat = cat.permute(1,0,2,3).reshape(7,-1).permute(1,0)
        cat = cat[cat[:,0]>thresh]
        # print(cat[:,0].reshape(-1).shape)
        # print((cat[:,2:]).shape)
        idx = torchvision.ops.nms(cat[:,3:], cat[:,1].reshape(-1), nms_thresh)
        cat = cat[idx]
        if return_withclass:
          final_proposals.append(cat[:,2:].detach())
        else:
          final_proposals.append(cat[:,3:].detach())
        final_conf_probs.append(cat[:,0].reshape(-1,1).detach())
      # print(cat.shape)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    if mode == 'RPN':
      features = [torch.zeros_like(i) for i in final_conf_probs] # dummy class
    return final_proposals, final_conf_probs, features


class TwoStageDetector(nn.Module):
  def __init__(self, in_dim=1280, hidden_dim=256, num_classes=20, \
               roi_output_w=2, roi_output_h=2, drop_ratio=0.3):
    super().__init__()

    assert(num_classes != 0)
    self.num_classes = num_classes
    self.roi_output_w, self.roi_output_h = roi_output_w, roi_output_h
    self.output_size = (roi_output_w, roi_output_h)
    ##############################################################################
    # TODO: Declare your RPN and the region cxlassification layer (in Fast R-CNN).#
    # The region classification layer is a sequential module with a Linear layer,#
    # followed by a Dropout (p=drop_ratio), a ReLU nonlinearity and another      #
    # Linear layer that predicts classification scores for each proposal.        #
    # HINT: The dimension of the two Linear layers are in_dim -> hidden_dim and  #
    # hidden_dim -> num_classes.                                                 #
    ##############################################################################
    # Your RPN and classification layers should be named as follows
    self.rpn = RPN()
    self.cls_layer =  torch.nn.Sequential(
    torch.nn.Linear(in_dim, hidden_dim),
    torch.nn.Dropout(drop_ratio),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim,num_classes)
    )
    self.MeanPool = nn.AvgPool2d((7,7))
    # Replace "pass" statement with your code
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

  def forward(self, images, bboxes):
    """
    Training-time forward pass for our two-stage Faster R-CNN detector.

    Inputs:
    - images: Tensor of shape (B, 3, H, W) giving input images
    - bboxes: Tensor of shape (B, N, 5) giving ground-truth bounding boxes
      and category labels, from the dataloader.

    Outputs:
    - total_loss: Torch scalar giving the overall training loss.
    """
    total_loss = None
    ##############################################################################
    # TODO: Implement the forward pass of TwoStageDetector.                      #
    # A few key steps are outlined as follows:                                   #
    # i) RPN, including image feature extraction, grid/anchor/proposal           #
    #       generation, activated and negative anchors determination.            #
    # ii) Perform RoI Align on proposals and meanpool the feature in the spatial #
    #     dimension.                                                             #
    # iii) Pass the RoI feature through the region classification layer which    #
    #      gives the class probilities.                                          #
    # iv) Compute class_prob through the prediction network and compute the      #
    #     cross entropy loss (cls_loss) between the prediction class_prob and    #
    #      the reference GT_class. Hint: Use F.cross_entropy loss.               #
    # v) Compute the total_loss which is formulated as:                          #
    #    total_loss = rpn_loss + cls_loss.                                       #
    ##############################################################################
    # Replace "pass" statement with your code
    torch.cuda.synchronize()
    total_loss, conf_scores, proposals, out, GT_class, pos_anchor_idx, anc_per_img  = self.rpn(images,bboxes,output_mode = 'all')
    out = torchvision.ops.roi_pool(out,proposals,self.output_size)
    out =torch.mean(out,(2,3),False)
    out = self.cls_layer(out)
    # print(GT_class.shape)
    # print(out.shape)
    cls_loss = torch.nn.functional.cross_entropy(out,GT_class)
    total_loss = total_loss + cls_loss
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return total_loss

  def inference(self, images, thresh=0.5, nms_thresh=0.7):
    """"
    Inference-time forward pass for our two-stage Faster R-CNN detector

    Inputs:
    - images: Tensor of shape (B, 3, H, W) giving input images
    - thresh: Threshold value on NMS object probabilities
    - nms_thresh: IoU threshold for NMS in the RPN

    We can output a variable number of predicted boxes per input image.
    In particular we assume that the input images[i] gives rise to P_i final
    predicted boxes.

    Outputs:
    - final_proposals: List of length (B,) where final_proposals[i] is a Tensor
      of shape (P_i, 4) giving the coordinates of the final predicted boxes for
      the input images[i]
    - final_conf_probs: List of length (B,) where final_conf_probs[i] is a
      Tensor of shape (P_i,) giving the predicted probabilites that the boxes
      in final_proposals[i] are objects (vs background)
    - final_class: List of length (B,), where final_class[i] is an int64 Tensor
      of shape (P_i,) giving the predicted category labels for each box in
      final_proposals[i].
    """
    final_proposals, final_conf_probs, final_class = None, None, None
    ##############################################################################
    # TODO: Predicting the final proposal coordinates `final_proposals`,        #
    # confidence scores `final_conf_probs`, and the class index `final_class`.  #
    # The overall steps are similar to the forward pass but now you do not need #
    # to decide the activated nor negative anchors.                             #
    # HINT: Use the RPN inference function to perform thresholding and NMS, and #
    # to compute final_proposals and final_conf_probs. Use the predicted class  #
    # probabilities from the second-stage network to compute final_class.       #
    ##############################################################################
    #  torch.cuda.synchronize()s
    final_proposals, final_conf_probs, out = self.rpn.inference( images, thresh=thresh, nms_thresh=nms_thresh,mode = 'FasterRCNN',return_withclass = True) 
    final_class = []
    # print(final_proposals[0].shape)
    B = len(final_conf_probs)
    hh,W,H = out[0].shape
    # print(hh,W,H)
    # print('loh')
    for b in range(B):
      local_out = out[b].reshape(1,hh,W,H)
      # print(local_out.shape)
      local_out = torchvision.ops.roi_align(local_out,final_proposals[b],self.output_size)
      local_out =torch.mean(local_out,(2,3),False)
      local_out = self.cls_layer(local_out)
      if local_out.shape[0] == 0:
        final_class.append(final_conf_probs[b].clone())
      else:
        local_class,idx = local_out.max(dim= 1)
        final_class.append(idx.reshape(-1,1))
        final_proposals[b]= final_proposals[b][:,1:]
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return final_proposals, final_conf_probs, final_class
"""
If this solution was useful to you and you want to thank me, please give me a star:
https://github.com/iMeleon/EECS-498-007-598-005-solutions
"""