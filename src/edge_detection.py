import numpy as np
from collections import deque
from .enhancement import convolution, gaussian_filter
def _gradient(a,kx,ky):
    gx=convolution(a,kx); gy=convolution(a,ky); return np.hypot(gx,gy),np.arctan2(gy,gx),gx,gy
def sobel(image): return _gradient(image,np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),np.array([[-1,-2,-1],[0,0,0],[1,2,1]]))[:2]
def prewitt(image): return _gradient(image,np.array([[-1,0,1]]*3),np.array([[-1,-1,-1],[0,0,0],[1,1,1]]))[:2]
def non_maximum_suppression(mag,ori):
    m=np.asarray(mag); ang=(np.degrees(ori)+180)%180; out=np.zeros_like(m)
    for y in range(1,m.shape[0]-1):
      for x in range(1,m.shape[1]-1):
       a=ang[y,x]; q=r=0
       if a<22.5 or a>=157.5:q,r=m[y,x+1],m[y,x-1]
       elif a<67.5:q,r=m[y+1,x-1],m[y-1,x+1]
       elif a<112.5:q,r=m[y+1,x],m[y-1,x]
       else:q,r=m[y-1,x-1],m[y+1,x+1]
       if m[y,x]>=q and m[y,x]>=r:out[y,x]=m[y,x]
    return out
def double_threshold(nms,low,high):
    a=np.asarray(nms); out=np.zeros(a.shape,np.uint8); out[a>=high]=255; out[(a>=low)&(a<high)]=75; return out
def hysteresis(dt):
    out=np.asarray(dt).copy(); q=deque(map(tuple,np.argwhere(out==255)))
    while q:
      y,x=q.popleft()
      for yy in range(max(0,y-1),min(out.shape[0],y+2)):
       for xx in range(max(0,x-1),min(out.shape[1],x+2)):
        if out[yy,xx]==75:out[yy,xx]=255;q.append((yy,xx))
    out[out!=255]=0; return out
def canny(image,sigma=1.2,low=20,high=50):
    s=gaussian_filter(np.asarray(image,dtype=float),5,sigma); m,o,_,_=_gradient(s,np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),np.array([[-1,-2,-1],[0,0,0],[1,2,1]])); n=non_maximum_suppression(m,o); d=double_threshold(n,low,high); return hysteresis(d),{"smoothed":s,"magnitude":m,"orientation":o,"nms":n,"double_threshold":d}
