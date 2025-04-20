import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import colorsys
import random
import json
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')
st.title('Advanced Grid Color Overlay')
st.markdown(
    'Upload images, customize grid & labels, save/load/export presets, and download results!')

# Session state
if 'presets' not in st.session_state:
    st.session_state['presets'] = {}

# Sidebar: Settings
with st.sidebar:
    st.header('Settings')
    uploaded_files = st.file_uploader('Choose images', type=['png','jpg','jpeg'], accept_multiple_files=True)
    font_file = st.file_uploader('Upload .ttf font', type=['ttf'])

    st.subheader('Grid')
    grid_type = st.selectbox('Grid Type', ['Rectangular', 'Hexagonal', 'Triangular'])
    grid_res = st.slider('Cells per axis', 5, 100, 20)
    adaptive = st.checkbox('Adaptive Resolution', value=False)

    st.subheader('Marker')
    shape = st.selectbox('Shape', ['Circle', 'Square', 'Diamond', 'Triangle', 'Cross'])
    dot_radius = st.slider('Radius (px)', 1, 30, 5)

    st.subheader('Labels')
    font_size = st.slider('Font size (px)', 8, 72, 12)
    color_format = st.selectbox('Color format', ['RGB','HEX','HSL'])
    label_pos = st.selectbox('Position', ['Above','Below','Left','Right','Center'])
    offset_x = st.slider('Offset X', -50, 50, 0)
    offset_y = st.slider('Offset Y', -50, 50, 0)
    text_rotation = st.slider('Text rotation (°)', -180, 180, 0)
    dynamic_contrast = st.checkbox('Dynamic contrast', value=True)
    avoid_overlap = st.checkbox('Avoid overlapping labels', value=False)

    st.subheader('Presets')
    preset_name = st.text_input('Save as preset')
    if st.button('Save preset') and preset_name:
        st.session_state.presets[preset_name] = {
            'grid_type': grid_type, 'grid_res': grid_res, 'adaptive': adaptive,
            'shape': shape, 'dot_radius': dot_radius,
            'font_size': font_size, 'color_format': color_format,
            'label_pos': label_pos, 'offset_x': offset_x, 'offset_y': offset_y,
            'text_rotation': text_rotation, 'dynamic_contrast': dynamic_contrast,
            'avoid_overlap': avoid_overlap
        }
        st.success(f"Preset '{preset_name}' saved.")
    choice = st.selectbox('Load preset', ['']+list(st.session_state.presets.keys()))
    if choice:
        p = st.session_state.presets[choice]
        grid_type, grid_res, adaptive = p['grid_type'], p['grid_res'], p['adaptive']
        shape, dot_radius = p['shape'], p['dot_radius']
        font_size, color_format = p['font_size'], p['color_format']
        label_pos, offset_x, offset_y = p['label_pos'], p['offset_x'], p['offset_y']
        text_rotation, dynamic_contrast = p['text_rotation'], p['dynamic_contrast']
        avoid_overlap = p['avoid_overlap']
        st.info(f"Loaded preset '{choice}'.")
    if st.button('Randomize'):
        st.session_state['rand'] = True
        st.experimental_rerun()
    # Export presets
    if st.session_state.presets:
        presets_json = json.dumps(st.session_state.presets)
        st.download_button('Export presets', data=presets_json, file_name='presets.json', mime='application/json')

# Utility functions

def get_contrast_text_color(rgb):
    r, g, b = [c/255.0 for c in rgb]
    lum = 0.299*r + 0.587*g + 0.114*b
    return (0,0,0) if lum > 0.5 else (255,255,255)

label_bboxes = []

def intersects_any(box):
    x0,y0,x1,y1 = box
    for bx0,by0,bx1,by1 in label_bboxes:
        if not (x1<bx0 or x0>bx1 or y1<by0 or y0>by1):
            return True
    return False

# Marker drawing
def draw_marker(draw, x, y, r, color, shape):
    if shape == 'Circle': draw.ellipse((x-r,y-r,x+r,y+r), fill=color)
    elif shape == 'Square': draw.rectangle((x-r,y-r,x+r,y+r), fill=color)
    elif shape == 'Diamond': pts=[(x,y-r),(x+r,y),(x,y+r),(x-r,y)]; draw.polygon(pts,fill=color)
    elif shape == 'Triangle': pts=[(x,y-r),(x+r,y+r),(x-r,y+r)]; draw.polygon(pts,fill=color)
    else: draw.line((x-r,y,x+r,y),fill=color,width=max(1,r//2)); draw.line((x,y-r,x,y+r),fill=color,width=max(1,r//2))

# Processing
if 'rand' in st.session_state or uploaded_files:
    cols = st.columns(2)
    for idx, f in enumerate(uploaded_files):
        with cols[idx%2]:
            img = Image.open(f).convert('RGBA'); w,h=img.size; arr=np.array(img)
            # resolution
            grid_res_eff = grid_res
            if adaptive:
                var = float(arr[:,:,:3].var()); grid_res_eff = min(max(int(grid_res*(1+var/10000)),5),200)
            step_x, step_y = max(1,w//grid_res_eff), max(1,h//grid_res_eff)
            # canvas & font
            result=Image.new('RGBA',(w,h),(255,255,255,0)); draw=ImageDraw.Draw(result)
            try:
                if font_file: font=ImageFont.truetype(io.BytesIO(font_file.read()), font_size)
                else: font=ImageFont.truetype('arial.ttf', font_size)
            except: font=ImageFont.load_default()
            label_bboxes.clear()
            samples=[]
            # iterate grid
            for i in range(0,w,step_x):
                for j in range(0,h,step_y):
                    cell=arr[j:min(j+step_y,h), i:min(i+step_x,w)]
                    if cell.size==0: continue
                    avg=tuple(int(c) for c in cell[:,:,:3].mean(axis=(0,1))); samples.append(avg)
                    x,y=i,j; draw_marker(draw,x,y,dot_radius,avg,shape)
                    # label
                    if color_format=='RGB': txt=str(avg)
                    elif color_format=='HEX': txt='#{:02X}{:02X}{:02X}'.format(*avg)
                    else: hls=colorsys.rgb_to_hls(avg[0]/255,avg[1]/255,avg[2]/255); txt=f"HSL({int(hls[0]*360)},{int(hls[2]*100)}%,{int(hls[1]*100)}%)"
                    text_color = get_contrast_text_color(avg) if dynamic_contrast else avg
                    tw,th=draw.textsize(txt,font=font)
                    dx,dy={'Above':(-tw//2+offset_x,-dot_radius-th+offset_y),'Below':(-tw//2+offset_x,dot_radius+offset_y),
                             'Left':(-dot_radius-tw+offset_x,-th//2+offset_y),'Right':(dot_radius+offset_x,-th//2+offset_y),
                             'Center':(-tw//2+offset_x,-th//2+offset_y)}[label_pos]
                    # render rotated text as separate layer
                    txt_img = Image.new('RGBA',(tw,th),(0,0,0,0)); txt_draw=ImageDraw.Draw(txt_img)
                    txt_draw.text((0,0),txt,font=font,fill=text_color)
                    txt_img = txt_img.rotate(text_rotation, expand=True)
                    bx0,by0 = x+dx, y+dy
                    bx1,by1 = bx0+txt_img.width, by0+txt_img.height
                    if avoid_overlap and intersects_any((bx0,by0,bx1,by1)): continue
                    label_bboxes.append((bx0,by0,bx1,by1))
                    result.alpha_composite(txt_img, (bx0,by0))
            # display & download
            st.image(result, use_container_width=True)
            buf=io.BytesIO(); result.save(buf,'PNG'); buf.seek(0)
            st.download_button(f'Download {f.name}', data=buf, file_name=f'overlay_{f.name}.png', mime='image/png')
            # histogram
            fig,ax=plt.subplots(); ax.hist(np.array(samples).flatten(),bins=30); ax.set_title('Color Distribution'); st.pyplot(fig)
    if 'rand' in st.session_state: del st.session_state['rand']
