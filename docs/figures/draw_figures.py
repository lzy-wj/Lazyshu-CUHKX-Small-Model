"""Reproducible publication-figure previews for the supplied Lazyshu report.

All values and relationships are transcribed from the supplied six-page PDF.
The modality illustrations are synthetic schematics, not competition samples.
Run: python3 draw_figures.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, FancyArrowPatch
from matplotlib import patheffects
from PIL import Image, ImageOps, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
C = dict(ink='#172B42', muted='#617185', rule='#D6DEE7', pale='#F5F8FB',
         blue='#2367A1', bluebg='#EDF4FA', orange='#C97830', orangebg='#FBF1E7',
         purple='#7860A5', purplebg='#F2EEF8', green='#13866E', greenbg='#EAF6F1',
         gray='#8795A4', light='#E6EBF0', teal='#3B8793')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':16,
                     'text.color':C['ink'], 'axes.labelcolor':C['ink'],
                     'svg.fonttype':'none', 'pdf.fonttype':42, 'ps.fonttype':42})

def canvas(w,h):
    fig=plt.figure(figsize=(w/100,h/100),facecolor='white')
    ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,w),ylim=(h,0)); ax.axis('off')
    return fig,ax

def text(ax,x,y,s,size=16,weight='normal',color=None,ha='left',va='center',**kw):
    return ax.text(x,y,s,fontsize=size,fontweight=weight,color=color or C['ink'],
                   ha=ha,va=va,linespacing=1.35,**kw)

def box(ax,x,y,w,h,fill='white',edge=None,r=8,lw=1):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                    facecolor=fill,edgecolor=edge or C['rule'],linewidth=lw)
    ax.add_patch(p); return p

def line(ax,xy,color=None,lw=1.2,ls='-'):
    ax.plot(*zip(*xy),color=color or C['rule'],lw=lw,ls=ls,solid_capstyle='round')

def arrow(ax,start,end,color=None,lw=1.4,ls='-',rad=0):
    p=FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,
      linewidth=lw,color=color or C['muted'],linestyle=ls,
      connectionstyle=f'arc3,rad={rad}',shrinkA=0,shrinkB=0)
    ax.add_patch(p)

def label(ax,x,y,n,s):
    text(ax,x,y,f'({n})',15,'bold');text(ax,x+45,y,s,17,'bold')

def save(fig,name):
    for ext in ('png','svg','pdf'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=220,facecolor='white')
    plt.close(fig)

# Shared pose, deliberately synthetic. Coordinates are normalized.
POSE=np.array([[.51,.16],[.48,.145],[.54,.145],[.445,.16],[.57,.16],
 [.39,.32],[.60,.32],[.30,.46],[.71,.40],[.22,.38],[.80,.29],
 [.43,.57],[.57,.57],[.37,.75],[.66,.73],[.30,.93],[.70,.92]])
EDGES=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
       (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

def human_field(n=260):
    y,x=np.mgrid[0:1:complex(n),0:1:complex(n)]
    field=np.zeros_like(x)
    for i,j,r in [(5,7,.053),(7,9,.045),(6,8,.053),(8,10,.045),
                  (11,13,.068),(13,15,.052),(12,14,.068),(14,16,.052),
                  (5,11,.085),(6,12,.085),(5,6,.078),(11,12,.076)]:
        a,b=POSE[i],POSE[j]; d=b-a
        t=np.clip(((x-a[0])*d[0]+(y-a[1])*d[1])/(d@d),0,1)
        distance=np.sqrt((x-a[0]-t*d[0])**2+(y-a[1]-t*d[1])**2)
        field=np.maximum(field,np.exp(-(distance/r)**4))
    field=np.maximum(field,np.exp(-(((x-.51)/.084)**2+((y-.175)/.095)**2)**2))
    torso=((x-.495)/.108)**2+((y-.44)/.18)**2
    field=np.maximum(field,np.exp(-torso**2))
    return x,y,field

def fig1():
    fig,ax=canvas(1440,435)
    groups=[(18,705,'VISUAL STREAMS',C['blue']),(740,1170,'STRUCTURED SENSORS',C['blue']),
            (1205,1422,'STRUCTURE ONLY',C['muted'])]
    for x0,x1,title,col in groups:
        text(ax,(x0+x1)/2,23,title,14,'bold',col,'center')
        line(ax,[(x0,43),(x1,43)],col,1.5)
    names=['Depth','Thermal','Infrared','Skeleton','IMU','Radar']
    details=['32 frames','32 frames','32 frames','17 joints','5 devices · 85-D','Temporal grouping']
    x,y,field=human_field()
    for k in range(6):
        xx=18+k*240
        label(ax,xx,72,chr(97+k),names[k])
        ia=fig.add_axes([(xx+2)/1440,(435-294)/435,210/1440,195/435])
        ia.set(xlim=(0,1),ylim=(1,0));ia.set_aspect('auto');ia.axis('off')
        if k<3:
            if k==0:
                rgba=plt.get_cmap('viridis')(.24+.63*(1-y)+.07*np.sin(12*x))
                rgba[field<.33]=matplotlib.colors.to_rgba('#F0F5F8')
            elif k==1:
                heat=.08+.88*field*(.73+.27*np.exp(-((y-.44)/.30)**2))
                rgba=plt.get_cmap('inferno')(heat)
            else:
                intensity=.15+.72*field*(.8+.2*np.cos(x*9+y*4))
                rgba=plt.get_cmap('gray')(intensity)
            ia.imshow(rgba,extent=(0,1,1,0),aspect='auto',interpolation='bilinear')
        elif k==3:
            ia.add_patch(Rectangle((0,0),1,1,color=C['bluebg']))
            for i,j in EDGES: ia.plot(POSE[[i,j],0],POSE[[i,j],1],color=C['blue'],lw=2.2)
            ia.scatter(POSE[:,0],POSE[:,1],s=24,c=C['blue'],edgecolors='white',linewidths=.6,zorder=3)
        elif k==4:
            t=np.linspace(0,1,240)
            ia.add_patch(Rectangle((0,0),1,1,color=C['pale']))
            for j in range(5):
                v=.13+j*.18+.045*np.sin(2*np.pi*(3.3*t+j*.12))+.02*np.sin(2*np.pi*10*t+j)
                ia.plot(t*.92+.04,v,color=[C['blue'],C['teal'],C['purple'],C['blue'],C['gray']][j],lw=1.6)
            ia.plot([.05,.95],[.97,.97],color=C['rule'],lw=1)
        else:
            ia.add_patch(Rectangle((0,0),1,1,color=C['pale']))
            rng=np.random.default_rng(19)
            points=np.clip(rng.normal([.53,.42],[.19,.18],(50,2)),.08,.9)
            ia.scatter(points[:,0],points[:,1],s=rng.uniform(5,20,len(points)),c=C['gray'],alpha=.75)
            ia.plot([.10,.91],[.84,.84],color=C['muted'],lw=1)
            for tt in [.13,.22,.25,.39,.53,.56,.74,.86]:
                ia.plot([tt,tt],[.81,.88],color=C['muted'],lw=1.4)
        text(ax,xx+107,329,details[k],14 if k==5 else 16,'bold',ha='center')
        text(ax,xx+107,357,'Neural input' if k<5 else 'Optional V164 only',14,
             color=C['muted'],ha='center')
    line(ax,[(18,388),(1422,388)])
    text(ax,18,412,'Synthetic modality illustrations; no competition sample is reproduced.',14,color=C['muted'])
    text(ax,1422,412,'Raw radar points are excluded from the neural backbone.',14,color=C['muted'],ha='right')
    save(fig,'figure_1_modalities')

def fig2():
    fig,ax=canvas(700,470)
    rows=[(20,'01','SELECT','Fold 0','Users 3, 8, 18, 23','652 clips'),
          (156,'02','CHECK TRANSFER','Fold 1','Users 5, 7, 17, 22','705 clips')]
    for yy,num,stage,name,users,n in rows:
        box(ax,20,yy,660,106,C['bluebg'],edge='#CCDDEA')
        text(ax,43,yy+31,num,15,'bold',C['blue'])
        text(ax,90,yy+30,name,20,'bold')
        text(ax,660,yy+30,stage,13,'bold',C['blue'],'right')
        text(ax,90,yy+71,users,16)
        text(ax,657,yy+71,n,17,'bold',C['blue'],'right')
    arrow(ax,(350,127),(350,154),C['blue'])
    arrow(ax,(350,262),(350,315),C['muted'])
    box(ax,267,278,166,30,'white','white',r=0)
    text(ax,350,293,'Freeze recipe',15,'bold',ha='center')
    box(ax,20,318,660,82,C['greenbg'],edge='#C8E1D8')
    text(ax,43,346,'03',15,'bold',C['green'])
    text(ax,90,346,'Full-data refit',20,'bold')
    text(ax,657,346,'3,036 clips',17,'bold',C['green'],'right')
    text(ax,90,378,'Deployment checkpoint; no held-out score',15,color=C['muted'])
    text(ax,350,427,'Both folds are development validation.',16,'bold',ha='center')
    text(ax,350,453,'Fold 1 was reused; it is not an independent test set.',14,color=C['muted'],ha='center')
    save(fig,'figure_2_validation')

def fig3():
    fig,ax=canvas(1440,665)
    for xx,ss in [(18,'INPUTS'),(224,'MODALITY ENCODERS'),(611,'MASKED FUSION'),(919,'EXPERTS'),(1260,'DECISION')]:
        text(ax,xx,28,ss,14,'bold',C['muted'])
    # The outline describes the stored base, not a true INT8 inference claim.
    box(ax,207,61,674,396,'#FBFDFF',C['blue'],r=10,lw=1.2)
    text(ax,225,85,'BASE NETWORK + INSERTED ADAPTERS',13,'bold',C['blue'])
    inputs=[(117,'Visual ×3','Depth · Thermal · IR'),(267,'Skeleton','17 joints'),(373,'IMU ×5','World-frame · 85-D')]
    for yy,main,sub in inputs:
        text(ax,18,yy+16,main,19,'bold');text(ax,18,yy+45,sub,13,color=C['muted'])
        arrow(ax,(183,yy+26),(227,yy+26),C['blue'])
    box(ax,228,111,304,135,C['bluebg'],edge='#C9DAE8')
    text(ax,248,138,'Shared R(2+1)D-34',17,'bold')
    text(ax,248,170,'32 frames · shared weights',13,color=C['muted'])
    box(ax,246,192,267,36,C['orangebg'],C['orange'],r=5)
    text(ax,379.5,210,'FP16 adapters · layer 4',13,'bold',C['orange'],'center')
    box(ax,228,266,304,80,C['bluebg'],edge='#C9DAE8')
    text(ax,248,291,'Temporal CNN',19,'bold')
    text(ax,248,322,'Coordinates · bones · velocities',13,color=C['muted'])
    box(ax,228,366,304,71,C['bluebg'],edge='#C9DAE8')
    text(ax,248,390,'Conv1D + BiLSTM',19,'bold')
    text(ax,248,418,'Two temporal convolutions',13,color=C['muted'])
    box(ax,611,203,247,175,C['bluebg'],edge='#C9DAE8')
    text(ax,631,233,'Transformer',18,'bold')
    text(ax,631,268,'2 layers · 6 heads',16)
    text(ax,631,297,'192-D modality tokens',14,color=C['muted'])
    text(ax,631,329,'[CLS]    V    S    I',13,'bold',C['blue'])
    text(ax,631,354,'Missing-stream mask',13,color=C['muted'])
    # Visual stream logits provide the additive skip described in Eq. (1).
    line(ax,[(532,139),(585,139),(585,134),(847,134)],C['blue'],1.5)
    text(ax,694,116,'Visual-logit skip',13,color=C['blue'],ha='center')
    line(ax,[(858,280),(869,280),(869,134)],C['blue'],1.5)
    ax.add_patch(Circle((858,134),11,facecolor='white',edgecolor=C['blue'],lw=1.3))
    text(ax,858,133,'+',17,'bold',C['blue'],'center')
    arrow(ax,(532,177),(611,238),C['blue'])
    arrow(ax,(532,306),(611,291),C['blue'])
    arrow(ax,(532,400),(611,347),C['blue'])
    # Expert stack is a conceptual summary of alternate passes / compact heads.
    line(ax,[(881,134),(905,134),(905,394)],C['blue'],1.4)
    arrow(ax,(869,134),(924,134),C['blue'])
    arrow(ax,(905,276),(924,276),C['purple'])
    arrow(ax,(905,396),(924,396),C['orange'])
    experts=[(100,91,'Base posterior','V24 · soup + TTA',C['blue'],C['bluebg']),
             (230,97,'Sequence route','V57 anchor · V71 retained',C['purple'],C['purplebg']),
             (360,92,'Paired adapter','V114 · anchored fusion',C['orange'],C['orangebg'])]
    for yy,hh,main,sub,col,bg in experts:
        box(ax,924,yy,276,hh,bg,edge=col)
        text(ax,943,yy+29,main,17,'bold')
        text(ax,943,yy+62,sub,14,color=col)
    box(ax,1260,198,162,171,C['greenbg'],C['green'])
    text(ax,1341,228,'Per-route',17,'bold',C['green'],'center')
    text(ax,1341,254,'decoding',17,'bold',C['green'],'center')
    text(ax,1341,292,'V127 select',15,'bold',ha='center')
    text(ax,1341,318,'+ coverage',14,ha='center')
    text(ax,1341,341,'protection',14,ha='center')
    text(ax,1341,359,'See Figure 4',12,color=C['muted'],ha='center')
    arrow(ax,(1200,145),(1260,233),C['blue'])
    arrow(ax,(1200,277),(1260,280),C['purple'])
    arrow(ax,(1200,406),(1260,337),C['orange'])
    # Candidate-specific decoding precedes route selection in the frozen code.
    arrow(ax,(1341,369),(1341,487),C['green'])
    box(ax,924,489,498,68,C['greenbg'],'#C8E1D8')
    text(ax,943,511,'Frozen date rollback',17,'bold')
    text(ax,943,538,'Restores decoded V24 only under the disclosed rule',13,color=C['muted'])
    arrow(ax,(1173,557),(1173,586),C['muted'],ls='--')
    box(ax,924,589,498,51,C['pale'],C['gray'])
    text(ax,943,614,'Optional V164 · evidence-gated radar hedge',13,'bold')
    text(ax,224,492,'Five trained fits → one weighted parameter soup',15,'bold',C['blue'])
    text(ax,224,520,'The soup is built before inference; it adds no five-model ensemble.',13,color=C['muted'])
    line(ax,[(224,545),(880,545)])
    text(ax,224,574,'Base: INT8 storage → BF16 forward. Adapters: separate FP16 files.',13,color=C['muted'])
    text(ax,224,600,'Selected paths never activate V71 FiLM; unmatched output equals V57.',12,color=C['muted'])
    text(ax,224,626,'The frozen package retains and charges the 4.08 MB development asset.',12,'bold',C['muted'])
    save(fig,'figure_3_architecture')

def fig4():
    fig,ax=canvas(700,560)
    text(ax,350,25,'V127 · supervision-aware routing',20,'bold',ha='center')
    box(ax,180,56,340,58,C['pale'])
    text(ax,350,85,'Clip posterior + metadata',15,'bold',ha='center')
    arrow(ax,(350,114),(350,143))
    box(ax,139,145,422,95,'white',C['ink'])
    text(ax,350,175,'Labeled same-person source',18,'bold',ha='center')
    text(ax,350,204,'available?',18,'bold',ha='center')
    line(ax,[(350,240),(350,266),(174,266),(174,304)],C['orange'],1.5)
    arrow(ax,(174,294),(174,313),C['orange'])
    line(ax,[(350,266),(526,266),(526,304)],C['purple'],1.5)
    arrow(ax,(526,294),(526,313),C['purple'])
    text(ax,172,288,'YES',13,'bold',C['orange'],'center')
    text(ax,526,288,'NO / UNKNOWN DATE',13,'bold',C['purple'],'center')
    box(ax,20,317,310,105,C['orangebg'],C['orange'])
    text(ax,175,348,'Paired adapter · V114',16,'bold',ha='center')
    text(ax,175,379,'60% adapter evidence',14,ha='center')
    text(ax,175,401,'+ 40% protected source',14,ha='center')
    box(ax,370,317,310,105,C['purplebg'],C['purple'])
    text(ax,525,344,'V71 sequence route',16,'bold',ha='center')
    text(ax,525,375,'FiLM gate OFF',14,'bold',C['purple'],ha='center')
    text(ax,525,401,'Exact V57 anchor',14,ha='center')
    line(ax,[(20,452),(680,452)])
    ax.add_patch(Circle((32,481),6,facecolor=C['blue'],edgecolor='none'))
    text(ax,49,481,'Unsupported source labels → decoded V24 fallback',14)
    ax.add_patch(Circle((32,516),6,facecolor=C['gray'],edgecolor='none'))
    text(ax,49,516,'V71 FiLM is active only in paired env0, where V127 selects V114',13,color=C['muted'])
    save(fig,'figure_4_routing')

def style_plot(a,ylim,yticks):
    a.set_ylim(*ylim);a.set_yticks(yticks)
    a.tick_params(axis='both',labelsize=12,colors=C['muted'],length=0,pad=8)
    for p in a.spines.values():p.set_visible(False)
    a.grid(axis='y',color=C['light'],lw=.8,zorder=0)
    a.set_axisbelow(True)

def fig5():
    fig,ax=canvas(1440,520)
    label(ax,18,29,'a','Local development milestones')
    label(ax,900,29,'b','Deployment footprint')
    text(ax,25,79,'Early classifiers',14,'bold',C['muted'])
    text(ax,330,79,'Later decoded systems',14,'bold',C['muted'])
    early=fig.add_axes([.048,.31,.125,.43])
    style_plot(early,(50,76),[50,60,70])
    early.plot(range(3),[53.94,63.45,70.52],color=C['gray'],lw=1.7,marker='o',ms=5)
    early.set_xticks(range(3),['V1','V2','V3']); early.set_xlim(-.25,2.25)
    early.set_ylabel('Accuracy (%)',fontsize=13,labelpad=9)
    for x,v in enumerate([53.94,63.45,70.52]): early.text(x,v+1.9,f'{v:.2f}',ha='center',fontsize=11,color=C['muted'])
    later=fig.add_axes([.228,.31,.358,.43])
    vals=[86.07,87.91,88.06,88.50,88.80,89.39,90.35]
    names=['V24','V52','V57','V71','V127','V164','V166']
    style_plot(later,(85.3,92),[86,88,90])
    later.plot(range(7),vals,color=C['blue'],lw=1.6,marker='o',ms=4)
    later.set_xticks(range(7),names);later.set_xlim(-.35,6.4)
    for i in [0,1,2,3,5]:
        later.text(i,vals[i]+(.47 if i%2==0 else -.64),f'{vals[i]:.2f}',ha='center',fontsize=11,color=C['blue'])
    later.scatter([4],[88.80],s=76,marker='D',color=C['green'],edgecolor='white',lw=1.2,zorder=5)
    later.annotate('88.80',xy=(4,88.8),xytext=(4,90.2),ha='center',fontsize=13,fontweight='bold',color=C['green'],
                   arrowprops={'arrowstyle':'-','color':C['green'],'lw':.9})
    later.scatter([6],[90.35],s=84,marker='x',color=C['ink'],lw=2,zorder=5)
    later.text(6,91.1,'90.35',ha='center',fontsize=13,fontweight='bold',color=C['ink'])
    for i,t in enumerate(later.get_xticklabels()):
        if i==4:t.set_color(C['green']);t.set_fontweight('bold')
    text(ax,330,402,'◆  V127 selected',14,'bold',C['green'])
    text(ax,600,402,'×  V166 rejected',14,'bold',C['ink'])
    text(ax,25,447,'Separate metric scopes and y-axis ranges.',14,'bold',C['muted'])
    text(ax,25,476,'Development history; not a controlled ablation or independent test estimate.',13,color=C['muted'])
    line(ax,[(874,69),(874,487)])
    text(ax,923,99,'84.09',36,'bold',C['ink'])
    text(ax,1090,103,'MB used',18,'bold')
    text(ax,1420,97,'100 MB limit',14,'bold',C['muted'],'right')
    xx=922; yy=160; ww=496; hh=53
    values=[65.40,5.55,4.08,8.32,.744248,15.905752]
    cols=[C['blue'],C['teal'],C['purple'],C['orange'],C['gray'],C['light']]
    acc=0
    for v,col in zip(values,cols):
        ax.add_patch(Rectangle((xx+ww*acc/100,yy),ww*v/100,hh,facecolor=col,edgecolor='white',lw=.65));acc+=v
    used_x=xx+ww*.84094248
    line(ax,[(used_x,154),(used_x,218)],C['ink'],1)
    text(ax,923,241,'14 runtime assets',14,'bold')
    text(ax,1420,241,'15.91 MB headroom',14,'bold',C['green'],'right')
    rows=[('INT8 base','65.40',C['blue']),('YOLO cropper','5.55',C['teal']),('Retained V71 dev.','4.08',C['purple']),
          ('FP16 adapters','8.32',C['orange']),('Heads + state','0.74',C['gray'])]
    for i,(name,value,col) in enumerate(rows):
        yy=286+i*32
        ax.add_patch(Rectangle((924,yy-6),11,11,facecolor=col,edgecolor='none'))
        text(ax,948,yy,name,14)
        text(ax,1418,yy,value+' MB',14,'bold',ha='right')
    text(ax,923,476,'84,094,248 bytes total; component sizes rounded.',12,color=C['muted'])
    save(fig,'figure_5_results_budget')

def overview():
    # A contact sheet; individual files preserve their intended publication ratios.
    W=2200; pad=75; gap=55
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
    bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',27)
    entries=[('figure_1_modalities','FIGURE 1  /  MULTIMODAL INPUTS'),
             ('figure_3_architecture','FIGURE 3  /  MODEL ARCHITECTURE'),
             ('figure_5_results_budget','FIGURE 5  /  RESULTS & BUDGET')]
    fitted=[]
    for name,title in entries:
        im=Image.open(OUT/f'{name}.png').convert('RGB');im=ImageOps.contain(im,(W-2*pad,1300))
        fitted.append((im,title))
    left=Image.open(OUT/'figure_2_validation.png').convert('RGB')
    right=Image.open(OUT/'figure_4_routing.png').convert('RGB')
    fw=(W-2*pad-gap)//2
    left=ImageOps.contain(left,(fw,1000));right=ImageOps.contain(right,(fw,1000))
    H=190+sum(im.height+gap+45 for im,_ in fitted)+max(left.height,right.height)+130
    sheet=Image.new('RGB',(W,H),'#E9EDF2');d=ImageDraw.Draw(sheet)
    d.text((pad,38),'LAZYSHU  /  FIGURE REDESIGN',font=bold,fill=C['ink'])
    d.text((pad,84),'Technical report previews · synthetic illustrations · source values preserved',font=font,fill=C['muted'])
    yy=154
    # Put figures 1,3,5 first, with 2 and 4 in the final paired row.
    for im,title in fitted:
        d.text((pad,yy),title,font=bold,fill=C['muted']);yy+=43
        sheet.paste(im,(pad,yy));yy+=im.height+gap
    d.text((pad,yy),'FIGURE 2  /  VALIDATION',font=bold,fill=C['muted'])
    d.text((pad+fw+gap,yy),'FIGURE 4  /  ROUTING',font=bold,fill=C['muted']);yy+=44
    sheet.paste(left,(pad,yy));sheet.paste(right,(pad+fw+gap,yy))
    sheet=sheet.crop((0,0,W,yy+max(left.height,right.height)+pad))
    sheet.save(OUT/'Lazyshu_Figures_Overview.png')

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    for f in [fig1,fig2,fig3,fig4,fig5]: f()
    overview()
    print('Generated 5 PNG / SVG / PDF figures and a contact sheet.')
