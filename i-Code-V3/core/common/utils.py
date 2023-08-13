import random
import torch
from collections import OrderedDict

import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision import transforms as tvtrans

from decord import VideoReader, cpu, gpu

###############
# text helper #
###############

def remove_duplicate_word(tx):
    def combine_words(input, length):
        combined_inputs = []
        if len(splitted_input)>1:
            for i in range(len(input)-1):
                combined_inputs.append(input[i]+" "+last_word_of(splitted_input[i+1],length)) #add the last word of the right-neighbour (overlapping) sequence (before it has expanded), which is the next word in the original sentence
        return combined_inputs, length+1

    def remove_duplicates(input, length):
        bool_broke=False #this means we didn't find any duplicates here
        for i in range(len(input) - length):
            if input[i]==input[i + length]: #found a duplicate piece of sentence!
                for j in range(0, length): #remove the overlapping sequences in reverse order
                    del input[i + length - j]
                bool_broke = True
                break #break the for loop as the loop length does not matches the length of splitted_input anymore as we removed elements
        if bool_broke:
            return remove_duplicates(input, length) #if we found a duplicate, look for another duplicate of the same length
        return input

    def last_word_of(input, length):
        splitted = input.split(" ")
        if len(splitted)==0:
            return input
        else:
            return splitted[length-1]

    def split_and_puncsplit(text):
        tx = text.split(" ")
        txnew = []
        for txi in tx:
            txqueue=[]
            while True:
                if txi[0] in '([{':
                    txqueue.extend([txi[:1], '<puncnext>'])
                    txi = txi[1:]
                    if len(txi) == 0:
                        break
                else:
                    break
            txnew += txqueue
            txstack=[]
            if len(txi) == 0:
                continue
            while True:
                if txi[-1] in '?!.,:;}])':
                    txstack = ['<puncnext>', txi[-1:]] + txstack
                    txi = txi[:-1]
                    if len(txi) == 0:
                        break
                else:
                    break
            if len(txi) != 0:
                txnew += [txi]
            txnew += txstack
        return txnew

    if tx == '':
        return tx

    splitted_input = split_and_puncsplit(tx)
    word_length = 1
    intermediate_output = False
    while len(splitted_input)>1:
        splitted_input = remove_duplicates(splitted_input, word_length)
        if len(splitted_input)>1:
            splitted_input, word_length = combine_words(splitted_input, word_length)
        if intermediate_output:
            print(splitted_input)
            print(word_length)
    output = splitted_input[0]
    output = output.replace(' <puncnext> ', '')
    return output

#################
# vision helper #
#################

def regularize_image(x, image_size=512):
    BICUBIC = T.InterpolationMode.BICUBIC
    if isinstance(x, str):
        x = Image.open(x)
    elif isinstance(x, Image.Image):
        x = x.convert('RGB')
    elif isinstance(x, np.ndarray):
        x = Image.fromarray(x).convert('RGB')
    elif isinstance(x, torch.Tensor):
        pass
    else:
        assert False, 'Unknown image type'
    
    transforms = T.Compose([
                T.RandomCrop(min(x.size)),
                T.Resize(
                    (image_size, image_size),
                    interpolation=BICUBIC,
                ),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
    x = transforms(x)
    assert (x.shape[1]==image_size) & (x.shape[2]==image_size), \
        'Wrong image size'

    x = x * 2 - 1
    return x


def center_crop(img, new_width=None, new_height=None):        

    width = img.shape[2]
    height = img.shape[1]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))
    if len(img.shape) == 3:
        center_cropped_img = img[:, top:bottom, left:right]
    else:
        center_cropped_img = img[:, top:bottom, left:right, ...]

    return center_cropped_img

def _transform(n_px):
    return Compose([
        Resize([n_px, n_px], interpolation=T.InterpolationMode.BICUBIC),])
    
def regularize_video(video, image_size=256):
    min_shape = min(video.shape[1:3])
    video = center_crop(video, min_shape, min_shape)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)
    video = _transform(image_size)(video)       
    video = video/255.0 * 2.0 - 1.0
    return video.permute(1, 0, 2, 3)

def time_to_indices(video_reader, time):
    times = video_reader.get_frame_timestamp(range(len(video_reader))).mean(-1)
    indices = np.searchsorted(times, time)
    # Use `np.bitwise_or` so it works both with scalars and numpy arrays.
    return np.where(np.bitwise_or(indices == 0, times[indices] - time <= time - times[indices - 1]), indices,
                    indices - 1)

def load_video(video_path, sample_duration=8.0, num_frames=8):
    sample_duration = 4.0
    num_frames = 4

    vr = VideoReader(video_path, ctx=cpu(0))
    framerate = vr.get_avg_fps()
    video_frame_len = len(vr)
    video_len = video_frame_len/framerate
    sample_duration = min(sample_duration, video_len)

    if video_len > sample_duration:
        s = random.random() * (video_len - sample_duration)
        t = s + sample_duration
        start, end = time_to_indices(vr, [s, t])
        end = min(video_frame_len-1, end)
        start = min(start, end-1)
        downsamlp_indices = np.linspace(start, end, num_frames, endpoint=True).astype(int).tolist()
    else:            
        downsamlp_indices = np.linspace(0, video_frame_len-1, num_frames, endpoint=True).astype(int).tolist()

    video = vr.get_batch(downsamlp_indices).asnumpy()
    return video


###############
# some helper #
###############

def atomic_save(cfg, net, opt, step, path):
    if isinstance(net, (torch.nn.DataParallel,
                        torch.nn.parallel.DistributedDataParallel)):
        netm = net.module
    else:
        netm = net
    sd = netm.state_dict()
    slimmed_sd = [(ki, vi) for ki, vi in sd.items()
        if ki.find('first_stage_model')!=0 and ki.find('cond_stage_model')!=0]

    checkpoint = {
        "config" : cfg,
        "state_dict" : OrderedDict(slimmed_sd),
        "step" : step}
    if opt is not None:
        checkpoint['optimizer_states'] = opt.state_dict()
    import io
    import fsspec
    bytesbuffer = io.BytesIO()
    torch.save(checkpoint, bytesbuffer)
    with fsspec.open(path, "wb") as f:
        f.write(bytesbuffer.getvalue())

def load_state_dict(net, cfg):
    pretrained_pth_full  = cfg.get('pretrained_pth_full' , None)
    pretrained_ckpt_full = cfg.get('pretrained_ckpt_full', None)
    pretrained_pth       = cfg.get('pretrained_pth'      , None)
    pretrained_ckpt      = cfg.get('pretrained_ckpt'     , None)
    pretrained_pth_dm    = cfg.get('pretrained_pth_dm'   , None)
    pretrained_pth_ema   = cfg.get('pretrained_pth_ema'  , None)
    strict_sd = cfg.get('strict_sd', False)
    errmsg = "Overlapped model state_dict! This is undesired behavior!"

    if pretrained_pth_full is not None or pretrained_ckpt_full is not None:
        assert (pretrained_pth is None) and \
               (pretrained_ckpt is None) and \
               (pretrained_pth_dm is None) and \
               (pretrained_pth_ema is None), errmsg            
        if pretrained_pth_full is not None:
            target_file = pretrained_pth_full
            sd = torch.load(target_file, map_location='cpu')
            assert pretrained_ckpt is None, errmsg
        else:
            target_file = pretrained_ckpt_full
            sd = torch.load(target_file, map_location='cpu')['state_dict']
        print('Load full model from [{}] strict [{}].'.format(
            target_file, strict_sd))
        net.load_state_dict(sd, strict=strict_sd)

    if pretrained_pth is not None or pretrained_ckpt is not None:
        assert (pretrained_ckpt_full is None) and \
               (pretrained_pth_full is None) and \
               (pretrained_pth_dm is None) and \
               (pretrained_pth_ema is None), errmsg
        if pretrained_pth is not None:
            target_file = pretrained_pth
            sd = torch.load(target_file, map_location='cpu')
            assert pretrained_ckpt is None, errmsg
        else:
            target_file = pretrained_ckpt
            sd = torch.load(target_file, map_location='cpu')['state_dict']
        print('Load model from [{}] strict [{}].'.format(
            target_file, strict_sd))
        sd_extra = [(ki, vi) for ki, vi in net.state_dict().items() \
            if ki.find('first_stage_model')==0 or ki.find('cond_stage_model')==0]
        sd.update(OrderedDict(sd_extra))
        net.load_state_dict(sd, strict=strict_sd)

    if pretrained_pth_dm is not None:
        assert (pretrained_ckpt_full is None) and \
               (pretrained_pth_full is None) and \
               (pretrained_pth is None) and \
               (pretrained_ckpt is None), errmsg
        print('Load diffusion model from [{}] strict [{}].'.format(
            pretrained_pth_dm, strict_sd))
        sd = torch.load(pretrained_pth_dm, map_location='cpu')
        net.model.diffusion_model.load_state_dict(sd, strict=strict_sd)

    if pretrained_pth_ema is not None:
        assert (pretrained_ckpt_full is None) and \
               (pretrained_pth_full is None) and \
               (pretrained_pth is None) and \
               (pretrained_ckpt is None), errmsg
        print('Load unet ema model from [{}] strict [{}].'.format(
            pretrained_pth_ema, strict_sd))
        sd = torch.load(pretrained_pth_ema, map_location='cpu')
        net.model_ema.load_state_dict(sd, strict=strict_sd)

def auto_merge_imlist(imlist, max=64):
    imlist = imlist[0:max]
    h, w = imlist[0].shape[0:2]
    num_images = len(imlist)
    num_row = int(np.sqrt(num_images))
    num_col = num_images//num_row + 1 if num_images%num_row!=0 else num_images//num_row
    canvas = np.zeros([num_row*h, num_col*w, 3], dtype=np.uint8)
    for idx, im in enumerate(imlist):
        hi = (idx // num_col) * h
        wi = (idx % num_col) * w
        canvas[hi:hi+h, wi:wi+w, :] = im
    return canvas

def latent2im(net, latent):
    single_input = len(latent.shape) == 3
    if single_input:
        latent = latent[None]
    im = net.decode_image(latent.to(net.device))
    im = torch.clamp((im+1.0)/2.0, min=0.0, max=1.0)
    im = [tvtrans.ToPILImage()(i) for i in im]
    if single_input:
        im = im[0]
    return im

def im2latent(net, im):
    single_input = not isinstance(im, list)
    if single_input:
        im = [im]
    im = torch.stack([tvtrans.ToTensor()(i) for i in im], dim=0)
    im = (im*2-1).to(net.device)
    z = net.encode_image(im)
    if single_input:
        z = z[0]
    return z

class color_adjust(object):
    def __init__(self, ref_from, ref_to):
        x0, m0, std0 = self.get_data_and_stat(ref_from)
        x1, m1, std1 = self.get_data_and_stat(ref_to)
        self.ref_from_stat = (m0, std0)
        self.ref_to_stat   = (m1, std1)
        self.ref_from = self.preprocess(x0).reshape(-1, 3)
        self.ref_to = x1.reshape(-1, 3)

    def get_data_and_stat(self, x):
        if isinstance(x, str):
            x = np.array(PIL.Image.open(x))
        elif isinstance(x, PIL.Image.Image):
            x = np.array(x)
        elif isinstance(x, torch.Tensor):
            x = torch.clamp(x, min=0.0, max=1.0)
            x = np.array(tvtrans.ToPILImage()(x))
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise ValueError
        x = x.astype(float)
        m = np.reshape(x, (-1, 3)).mean(0)
        s = np.reshape(x, (-1, 3)).std(0)
        return x, m, s

    def preprocess(self, x):
        m0, s0 = self.ref_from_stat
        m1, s1 = self.ref_to_stat
        y = ((x-m0)/s0)*s1 + m1
        return y

    def __call__(self, xin, keep=0, simple=False):
        xin, _, _ = self.get_data_and_stat(xin)
        x = self.preprocess(xin)
        if simple: 
            y = (x*(1-keep) + xin*keep)
            y = np.clip(y, 0, 255).astype(np.uint8)
            return y

        h, w = x.shape[:2]
        x = x.reshape(-1, 3)
        y = []
        for chi in range(3):
            yi = self.pdf_transfer_1d(self.ref_from[:, chi], self.ref_to[:, chi], x[:, chi])
            y.append(yi)

        y = np.stack(y, axis=1)
        y = y.reshape(h, w, 3)
        y = (y.astype(float)*(1-keep) + xin.astype(float)*keep)
        y = np.clip(y, 0, 255).astype(np.uint8)
        return y

    def pdf_transfer_1d(self, arr_fo, arr_to, arr_in, n=600):
        arr = np.concatenate((arr_fo, arr_to))
        min_v = arr.min() - 1e-6
        max_v = arr.max() + 1e-6
        min_vto = arr_to.min() - 1e-6
        max_vto = arr_to.max() + 1e-6
        xs = np.array(
            [min_v + (max_v - min_v) * i / n for i in range(n + 1)])
        hist_fo, _ = np.histogram(arr_fo, xs)
        hist_to, _ = np.histogram(arr_to, xs)
        xs = xs[:-1]
        # compute probability distribution
        cum_fo = np.cumsum(hist_fo)
        cum_to = np.cumsum(hist_to)
        d_fo = cum_fo / cum_fo[-1]
        d_to = cum_to / cum_to[-1]
        # transfer
        t_d = np.interp(d_fo, d_to, xs)
        t_d[d_fo <= d_to[ 0]] = min_vto
        t_d[d_fo >= d_to[-1]] = max_vto
        arr_out = np.interp(arr_in, xs, t_d)
        return arr_out
