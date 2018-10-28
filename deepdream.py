
import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import urllib.request
import os
import zipfile  


#deep dream is first built by Google Research on August 2015 and related to inception and deep dream lib 
#neural style: a Torch variation of deep dreams, used to transfer artistic styles 
 

def main(): 
    #import the pre-train neural network model from google 
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip' 
    data_dir = '../data/' 
    model_name = os.path.split(url)[-1] 
    local_zip_file = os.path.join(data_dir, model_name)  
    
    if not os.path.exists(local_zip_file): 
        model_url = urllib.request.urlopen(url)
        with open(local_zip_file, 'wb') as output:
            output.write(model_url.read())
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref: 
            zip_ref.extractall(data_dir) 
  
    img_noise = np.random.uniform(size=(224,224,3)) + 40.0 #add some noise to the image  
    model_fn = 'tensorflow_inception_graph.pb'   
    
    graph = tf.Graph() #create Tensorflow session & load the model  
    sess = tf.InteractiveSession(graph=graph) 
    
    with tf.gfile.FastGFile(os.path.join(data_dir, model_fn), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input') 
    
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input':t_preprocessed})
    
    layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name] #list comprehsion func
    feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers] 
    print('Num of layers', len(layers))
    print('Total num of feature channels:', sum(feature_nums))  
    
    def strip_consts(graph_def, max_const_size=32):
        strip_def = tf.GrahDef()
    for k in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(k) 
        if n.op == 'Const': 
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size: 
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

    def rename_nodes(graph_def, rename_func):
        res_def = tf.GraphDef()
    for k in graph_def.node:
        n = res_def.node.add() 
        n.MergeFrom(k)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def
      
    def showarray(a):
        a = np.uint8(np.clip(a, 0, 1)*255)
        plt.imshow(a)
        plt.show()
        
    def visstd(a, s=0.1):
        return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5 
    
    def T(layer):
        return graph.get_tensor_by_name("import/%s:0"%layer) 
    
    def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
        t_score = tf.reduce_mean(t_obj) #score function for optimization
        t_grad = tf.gradients(t_score, t_input)[0] #grad descent 
        
        img = img0.copy()
        for _ in range(iter_n):
            g, _ = sess.run([t_grad, t_score], {t_input:img})
            g /= g.std()+1e-8  
            img += g*step
        showarray(visstd(img)) 
        
    def tffunc(*argtypes): 
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f): 
            output = f(*placeholders)
            def wrapper(*args, **kw):
                return output.eval(dict(zip(placeholders, args)), 
                session=kw.get('session')) 
            return wrapper
        return wrap  
    
    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0,:,:,:]
        resize = tffunc(np.float32, np.int32)(resize)  

    def calc_grad_tiled(img, t_grad, tile_size=512):   
        sz = tile_size 
        h, w = img.shape[:2] 
        sx, sy = np.random.randint(sz, size=2) 
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0) 
        grad = np.zeros_like(img) 
        
        for y in range(0, max(h-sz//2, sz),sz): 
            for x in range(0, max(w-sz//2, sz),sz): 
                sub = img_shift[y:y+sz,x:x+sz] 
                g = sess.run(t_grad, {t_input:sub}) 
                grad[y:y+sz,x:x+sz] = g 
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)     

    #def render_deepdreamvideo():
    def render_deepdream(t_obj, 
                         img0=img_noise,   
                         iter_n=10, 
                         step=1.5, 
                         octave_n=4, 
                         octave_scale=1.4): 
        t_score = tf.reduce_mean(t_obj) 
        t_grad = tf.gradients(t_score, t_input)[0]  
        img = img0
        octaves = [] 
        for _ in range(octave_n-1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw)/octave_scale))
            hi = img-resize(lo, hw)
            img = lo
            octaves.append(hi)
        
        for octave in range(octave_n):
            if octave>0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2])+hi
            for _ in range(iter_n):
                g = calc_grad_tiled(img, t_grad)
                img += g*(step / (np.abs(g).mean()+1e-7)) 
            showarray(img/255.0) 
            
         
    layer = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139 #feature channel to visualize 
    
    img0 = PIL.Image.open('desktop/2018-07-28_1130.png')
    img0 = np.float32(img0)  
    render_deepdream(tf.square(T('mixed4c')), img0)  
      
if __name__ == '__main__': 
    main() 
