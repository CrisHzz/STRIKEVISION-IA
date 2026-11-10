import json, io, textwrap
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import landscape, A4

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'output/annotation_audits/volkanovski_holloway_i'
G=B/'round1_geometry_masks'
report=json.loads((G/'report.json').read_text(encoding='utf8'))
source=ROOT/'data/annotations/action_strikes_direct_v1/alexander_volkanovski_vs_max_holloway_i_ufc_245__t_5G7bGTPxs__round1.json'
data=json.loads(source.read_text(encoding='utf8'))
z=np.load(G/'geometry_masks.npz',allow_pickle=False)
out=ROOT/'output/pdf/volkanovski_holloway_i_round1_poses_mascaras.pdf'
out.parent.mkdir(parents=True,exist_ok=True)
c=canvas.Canvas(str(out),pagesize=landscape(A4)); W,H=landscape(A4)
c.setTitle('Volkanovski - Holloway I | Round 1 | Poses y mascaras')
page=0
def header(title,sub=''):
    global page
    page+=1
    c.setFillColorRGB(.06,.12,.2);c.rect(0,H-65,W,65,fill=1,stroke=0)
    c.setFillColorRGB(1,1,1);c.setFont('Helvetica-Bold',18);c.drawString(28,H-28,title)
    c.setFont('Helvetica',10);c.drawString(28,H-48,sub)
    c.setFillColorRGB(.15,.2,.25);c.setFont('Helvetica',9)
    c.drawString(28,18,'UFC 245 | Round 1 | Auditoria automatica - pendiente validacion visual y adaptador v4')
    c.drawRightString(W-28,18,str(page))
def lines(txt,y,size=11,width=112):
    c.setFont('Helvetica',size)
    for line in textwrap.wrap(txt,width):
        c.drawString(30,y,line);y-=16
    return y-8
counts=Counter(a['label'] for a in data['actions'])
classes=['punch_head','punch_body','kick_leg','kick_body','kick_head','other_strike','ambiguous']
header('Round 1: acciones, poses y mascaras','Alexander Volkanovski vs Max Holloway I - 6 de noviembre de 2026')
y=H-100
y=lines('562/562 hojas RGB revisadas; 8.991 cuadros. 155 registros: 142 acciones positivas y 13 ambiguas.',y)
y=lines('107 acciones superan 65 % de cobertura articular del atacante tras filtros automaticos. Esto no constituye aprobacion para entrenamiento. training_ready=false; training_mask=false para las 155 acciones.',y)
for label in classes:
    items=[r for r in report['results'] if r['label']==label]
    y=lines(f"{label}: {counts[label]} registros | {sum(r['candidate_for_manual_validation'] for r in items)} candidatos para revision visual",y)
y=lines('Contenido: una pagina por accion, agrupada por clase; seis cuadros muestreados dentro de su intervalo. Se preservan etiquetas e identidades de la anotacion fuente. Las simultaneas aparecen como acciones independientes.',y)
c.showPage()
header('Como leer la auditoria','Las mascaras ocultan articulaciones rechazadas; no reconstruyen poses ausentes')
y=H-95
for txt in ['Verde: articulaciones retenidas del atacante. Azul: articulaciones retenidas del rival. Las lineas conectan solo pares con mascara valida. Fondo: cuadro RGB original, sin modificar.',
 'Cobertura = articulaciones retenidas del atacante / (17 articulaciones x cuadros del evento). Seis muestras visuales por accion; los porcentajes usan todos los cuadros del intervalo.',
 'Filtros: confianza >=0,50; coordenadas finitas y dentro de imagen; caja corporal con margen 15 % de altura; segmentos <=65 % de altura corporal; salto temporal <=30 % de altura cuando el track conserva ID.',
 'Estos umbrales son conservadores y no estan calibrados. La geometria plausible no garantiza que una articulacion siga al peleador correcto. Existen tramos excluidos por cambios de identidad y detecciones de arbitro/publico.',
 'Pendiente: validar articulaciones visualmente y adaptar secuencias con contexto temporal auditado al entrenamiento v4. El NPZ actual contiene solo cuadros de las acciones; no es un cache v4 listo para entrenar.',
 'Fuentes: JSON canonico round1 (revision '+str(data['revision'])+'); round1_geometry_masks/report.json y geometry_masks.npz. SHA256 poses: '+report['pose_sha256']]:
    y=lines(txt,y,11,112)
c.showPage()
edges=[(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
for label in classes:
    for ai,a in sorted(enumerate(data['actions']),key=lambda pair:pair[1]['start_frame']):
        if a['label']!=label:continue
        result=report['results'][ai]
        header(label+' | '+a['fighter'],f"Frames {a['start_frame']}-{a['end_frame']} | {a['start_frame']/data['fps']:.2f}-{a['end_frame']/data['fps']:.2f} s | ID {a['action_id']}")
        y=H-88
        y=lines(f"Cobertura atacante: {result['actor_joint_coverage']:.1%}. Candidato a revision: {'si' if result['candidate_for_manual_validation'] else 'no'}. Aprobado para entrenar: no.",y,11)
        count=a['end_frame']-a['start_frame']+1
        for k,ti in enumerate(np.rint(np.linspace(0,count-1,6)).astype(int)):
            f=a['start_frame']+int(ti); si=f//16; end=min(si*16+15,8990)
            sheet=Image.open(B/'round1_all_frames'/f'frames_{si*16:06d}_{end:06d}.jpg')
            j=f%16; ox=(j%4)*480;oy=(j//4)*270
            im=sheet.crop((ox,oy,ox+480,oy+270)).convert('RGB');d=ImageDraw.Draw(im)
            for pi,color in ((1,'#00bfff'),(0,'#00ff66')):
                pts=z['coordinates_px'][ai,ti,pi]/4;mask=z['point_mask'][ai,ti,pi]
                for p,q in edges:
                    if mask[p] and mask[q]:d.line([tuple(pts[p]),tuple(pts[q])],fill=color,width=3)
                for p,valid in zip(pts,mask):
                    if valid:d.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),fill=color)
            x=28+(k%3)*265; yy=H-300-(k//3)*170
            c.drawImage(ImageReader(im),x,yy,width=256,height=144)
            c.setFont('Helvetica',9);c.drawString(x,yy-12,f"Cuadro {f} | articulaciones atacante {int(z['point_mask'][ai,ti,0].sum())}/17")
        c.setFont('Helvetica',9)
        for k,line in enumerate(textwrap.wrap(a.get('note',''),135)[:2]):c.drawString(28,60-12*k,line)
        c.showPage()
c.save()
print(json.dumps({'pdf':str(out),'pages':page,'actions':len(data['actions'])}))
