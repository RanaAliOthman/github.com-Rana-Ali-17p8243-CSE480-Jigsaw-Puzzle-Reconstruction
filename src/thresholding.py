import numpy as np
from .enhancement import mean_filter, gaussian_filter, histogram
def global_threshold(image,threshold=127,invert=False):
    m=np.asarray(image)>threshold
    return ((~m if invert else m)*255).astype(np.uint8)
def otsu_threshold(image,invert=False):
    h=histogram(image).astype(float); p=h/max(h.sum(),1); w=np.cumsum(p); mu=np.cumsum(p*np.arange(256)); mt=mu[-1]
    score=(mt*w-mu)**2/(w*(1-w)+1e-12); t=int(np.argmax(score)); return global_threshold(image,t,invert),t
def adaptive_threshold(image,size=15,c=5,method="mean",invert=False):
    a=np.asarray(image,dtype=float); local=mean_filter(a,size) if method=="mean" else gaussian_filter(a,size,max(1,size/6))
    m=a>local-c; return ((~m if invert else m)*255).astype(np.uint8)
