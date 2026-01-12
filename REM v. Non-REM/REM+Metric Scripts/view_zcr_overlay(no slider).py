import pandas as pd, matplotlib.pyplot as plt, datetime, re, h5py, numpy as np
from matplotlib.widgets import Slider

filename  = r'.../highrisk02_20240820-141937.h5'
targetID  = 'XI-016162'
sleep_file= r'.../Sleep profile - HighRisk.txt'

# [Sleep parsing identical again]

with h5py.File(filename,'r') as f:
    base=f"Sensors/{targetID}"
    acc=np.array(f[f'{base}/Accelerometer'][:],dtype=np.float64)
    time_raw=np.array(f[f'{base}/Time'][:],dtype=np.float64)
    time_dt=np.array([datetime.datetime.fromtimestamp(t*1e-6) for t in time_raw])
df=pd.DataFrame(acc,columns=['ax','ay','az'],index=pd.to_datetime(time_dt))
sig=df['ax']

def zcr(x):
    x=np.sign(x);x[x==0]=1;x_next=np.roll(x,-1)
    return np.mean(x[:-1]!=x_next[:-1])

zcr_30s=sig.resample('30s').apply(zcr)
zcr_30s=zcr_30s.reindex(epoch_df.index,method='nearest')

fig,ax=plt.subplots(figsize=(15,7));plt.subplots_adjust(bottom=0.25)
ax_slider=plt.axes([0.15,0.1,0.7,0.03]);slider=Slider(ax_slider,'Epoch',0,len(epoch_df)-1,valinit=0,valstep=1)
wake_df=sleep_df[sleep_df['state'].str.lower()=='wake']
def update(i):
    i=int(i);start,end=epoch_df.index[i],epoch_df.index[i]+pd.Timedelta('30s')
    ax.clear();ax.set_xlim(start,end)
    ax.set_title(f"Wake + Zero-Crossing Rate ({start.strftime('%H:%M:%S')})")
    ax.set_xlabel("Time");ax.set_ylabel("ZCR (crossings/sample)");ax.grid(True)
    for s,e in zip(wake_df.index,wake_df.index.to_series().shift(-1)):
        if pd.isna(e):continue
        if s<end and e>start:
            ax.axvspan(max(s,start),min(e,end),color='orange',alpha=0.3)
            ax.axvline(s,color='k',ls='--',lw=1)
    ax.plot(zcr_30s.index,zcr_30s.values,color='green',lw=1.5)
    fig.canvas.draw_idle()
slider.on_changed(update);update(0);plt.show()
