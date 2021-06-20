import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class WaveletTransform(nn.Module): 
    def __init__(self, scale=1, dec=True, params_path='wavelet_weights_c2.pkl', transpose=True, groups=1):
        super(WaveletTransform, self).__init__()
        
        self.scale = scale
        self.dec = dec
        self.transpose = transpose
        
        # groups = 1
        ks = int(math.pow(2, self.scale)  )
        nc = groups * ks * ks
        
        if dec:
            self.conv = nn.Conv2d(in_channels=groups, out_channels=nc,
                                  kernel_size=ks, stride=ks, padding=0, groups=groups, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=groups,
                                           kernel_size=ks, stride=ks, padding=0, groups=groups, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                with open(params_path,'rb') as f :
                    dct = pickle.load(f)
                m.weight.data = torch.from_numpy(dct['rec%d' % ks])
                m.weight.requires_grad = False  
    
    
    def forward(self, x):
#         if len(x.shape) == 3 :
#             x = x.unsqueeze(1)
            
        if self.dec:
            output = self.conv(x)          
            if self.transpose:
                osz = output.size()
                #print(osz)
                output = output.view(osz[0], 1, -1, osz[2], osz[3]).transpose(1,2).contiguous().view(osz)
        else:
            if self.transpose:
                xx = x
                xsz = xx.size()
                xx = xx.view(xsz[0], -1, 1, xsz[2], xsz[3]).transpose(1,2).contiguous().view(xsz)             
            output = self.conv(xx)        
        return output 



class WaveletParams(nn.Module): 
    def __init__(self, scale=1, params_path='wavelet_weights_c2.pkl', transpose=True, groups=1, thold=0.1):
        super(WaveletParams, self).__init__()
        
        self.scale = scale
        self.transpose = transpose
        self.groups = groups
        self.thold = thold
        self.wt = WaveletTransform(scale=1, dec=True, params_path=params_path, transpose=transpose, groups=groups)
    
    
    def forward(self, x):
        if len(x.shape) == 3 :
            x = x.unsqueeze(1)
        
        data = x
        output_list = []
        for i in range(self.scale) :
            cur_wavelet = self.wt(data)
            if self.groups == 1 :
                data, cur_output = cur_wavelet.split([1,3], dim=1)
                cur_output = torch.abs(cur_output)
    #             cur_output = cur_output.mean(dim=1).unsqueeze(1).cpu().data.numpy()
                cur_output, _ = torch.max(cur_output,dim=1)
                cur_output = cur_output.unsqueeze(1).cpu().data.numpy()
                
                n,c,h,w = cur_output.shape
                output_min = cur_output.min(axis=-1).min(axis=-1)
                output_min = np.repeat(output_min, h, axis=0)
                output_min = np.repeat(output_min, w, axis=1)
                output_min = output_min.reshape((n,c,h,w))
                output_max = cur_output.max(axis=-1).max(axis=-1)
                output_max = np.repeat(output_max, h, axis=0)
                output_max = np.repeat(output_max, w, axis=1)
                output_max = output_max.reshape((n,c,h,w))
                cur_output = (cur_output-output_min) / (output_max-output_min)
                
                processed_list = []
                for image in cur_output :
                    for interval in np.arange(0,1,0.1) :
                        curr_per = (image<=interval+0.1).sum() / image.size
                        if curr_per>=0.85 :
                            result = np.zeros(image.shape, dtype=np.float32)
                            result[ image>=interval+0.1 ] = 1
                            processed_list.append(result)
                            break
                    if interval==0.9 :
                        np.save("error_disp.npy", data.detach().cpu().numpy())
                        raise Exception("interval is over limit!")
                
                gt = np.stack(processed_list)
                # gt[cur_output>=self.thold] = 1
                cur_output = torch.Tensor(gt)
            elif self.groups == 3 :
                r,g,b = cur_wavelet.chunk(3, dim=1)
                # print("r:{}, g:{}, b:{}, cur_wavelet:{}".format(r.size(), g.size(), b.size(), cur_wavelet.size()))
                r_data, r_cur_output = r.split([1,3], dim=1)
                g_data, g_cur_output = g.split([1,3], dim=1)
                b_data, b_cur_output = b.split([1,3], dim=1)
                data = torch.cat([r_data,g_data,b_data],dim=1)
                cur_output = torch.cat([r_cur_output,g_cur_output,b_cur_output],dim=1)
                cur_output = torch.abs(cur_output)
            else :
                raise Exception("groups should be either 1 or 3!")
            cur_wavelet = torch.abs(cur_wavelet)
            cur_output = cur_output.cuda()
            output_list.append(cur_output)
        output_list.append(data)
        return output_list


