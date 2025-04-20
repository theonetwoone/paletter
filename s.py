import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import colorsys
import random
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')
st.title('Advanced Grid Color Overlay')
st.markdown(
    'Upload one or more images, pick your font, then tweak grid, dot/shape, labels, and more. ' \
    'Save or randomize presets, view color histograms, and download batch results.')

# Session state for presets
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
    adaptive = st.checkbox('Adaptive Resolution (by variance)', value=False)

    st.subheader('Dot/Shape')
    shape = st.selectbox('Marker Shape', ['Circle', 'Square', 'Diamond', 'Triangle', 'Cross'])
    dot_radius = st.slider('Radius (px)', 1, 30, 5)

    st.subheader('Labels')
    font_size = st.slider('Font size (px)', 8, 72, 12)
    color_format = st.selectbox('Color code format', ['RGB','HEX','HSL'])
    label_pos = st.selectbox('Label Position', ['Above','Below','Left','Right','Center'])
    offset_x = st.slider('Label offset X', -50, 50, 0)
    offset_y = st.slider('Label offset Y', -50, 50, 0)
    dynamic_contrast = st.checkbox('Dynamic text contrast', value=True)

    st.subheader('Presets')
    preset_name = st.text_input('Preset name')
    if st.button('Save preset') and preset_name:
        st.session_state.presets[preset_name] = {
            'grid_type': grid_type,
            'grid_res': grid_res,
            'adaptive': adaptive,
            'shape': shape,
            'dot_radius': dot_radius,
            'font_size': font_size,
            'color_format': color_format,
            'label_pos': label_pos,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'dynamic_contrast': dynamic_contrast
        }
        st.success(f"Preset '{preset_name}' saved.")
    preset_choice = st.selectbox('Load preset', [''] + list(st.session_state.presets.keys()))
    if preset_choice:
        p = st.session_state.presets[preset_choice]
        grid_type = p['grid_type']; grid_res = p['grid_res']; adaptive = p['adaptive']
        shape = p['shape']; dot_radius = p['dot_radius']
        font_size = p['font_size']; color_format = p['color_format']
        label_pos = p['label_pos']; offset_x = p['offset_x']; offset_y = p['offset_y']
        dynamic_contrast = p['dynamic_contrast']
        st.info(f"Loaded preset '{preset_choice}'.")
    if st.button('Randomize'):
        # Randomize key settings
        grid_res = random.randint(5,100)
        dot_radius = random.randint(1,30)
        font_size = random.randint(8,72)
        color_format = random.choice(['RGB','HEX','HSL'])
        shape = random.choice(['Circle','Square','Diamond','Triangle','Cross'])
        label_pos = random.choice(['Above','Below','Left','Right','Center'])
        offset_x = random.randint(-20,20)
        offset_y = random.randint(-20,20)
        st.experimental_rerun()

# Function to compute text color based on contrast

def get_contrast_text_color(rgb):
    # luminance formula
    r, g, b = [c/255.0 for c in rgb]
    lum = 0.299*r + 0.587*g + 0.114*b
    return (0,0,0) if lum > 0.5 else (255,255,255)

# Function to draw marker

def draw_marker(draw, x, y, r, color, shape):
    if shape == 'Circle':
        draw.ellipse((x-r,y-r,x+r,y+r), fill=color)
    elif shape == 'Square':
        draw.rectangle((x-r,y-r,x+r,y+r), fill=color)
    elif shape == 'Diamond':
        pts = [(x,y-r),(x+r,y),(x,y+r),(x-r,y)]
        draw.polygon(pts, fill=color)
    elif shape == 'Triangle':
        pts = [(x,y-r),(x+r,y+r),(x-r,y+r)]
        draw.polygon(pts, fill=color)
    else:  # Cross
        draw.line((x-r,y,x+r,y), fill=color, width= max(1, r//2))
        draw.line((x,y-r,x,y+r), fill=color, width= max(1, r//2))

# Process images
if uploaded_files:
    cols = st.columns(2)
    for idx, uploaded_file in enumerate(uploaded_files):
        with cols[idx % 2]:
            img = Image.open(uploaded_file).convert('RGBA')
            w,h = img.size
            arr = np.array(img)

            # compute steps
            if adaptive:
                # use global variance to adjust resolution (simplified)
                var = float(arr[:,:,:3].var())
                grid_res_eff = min(max(int(grid_res * (1 + var/10000)),5),200)
            else:
                grid_res_eff = grid_res
            step_x = max(1, w//grid_res_eff)
            step_y = max(1, h//grid_res_eff)

            # prepare canvas and font
            result = Image.new('RGBA',(w,h),(255,255,255,0))
            draw = ImageDraw.Draw(result)
            try:
                if font_file:
                    font_bytes = io.BytesIO(font_file.read())
                    font = ImageFont.truetype(font_bytes, font_size)
                else:
                    font = ImageFont.truetype('arial.ttf', font_size)
            except:
                font = ImageFont.load_default()

            # collect colors for histogram
            samples = []

            # draw grid
            for i in range(0, w, step_x):
                for j in range(0, h, step_y):
                    cell = arr[j:min(j+step_y,h), i:min(i+step_x,w)]
                    if cell.size==0: continue
                    avg = tuple(int(c) for c in cell[:,:,:3].mean(axis=(0,1)))
                    samples.append(avg)
                    x,y = i, j
                    draw_marker(draw, x, y, dot_radius, avg, shape)

                    # format label
                    if color_format=='RGB': txt = str(avg)
                    elif color_format=='HEX': txt = '#{:02X}{:02X}{:02X}'.format(*avg)
                    else:
                        hls = colorsys.rgb_to_hls(avg[0]/255,avg[1]/255,avg[2]/255)
                        txt = f"HSL({int(hls[0]*360)},{int(hls[2]*100)}%,{int(hls[1]*100)}%)"

                    # determine text color
                    if dynamic_contrast:
                        text_color = get_contrast_text_color(avg)
                    else:
                        text_color = avg

                    tw,th = draw.textsize(txt, font=font)
                    # compute label position
                    dx,dy = {'Above':( -tw//2 + offset_x, -dot_radius-th + offset_y),
                            'Below':(-tw//2 + offset_x, dot_radius + offset_y),
                            'Left':(-dot_radius-tw + offset_x, -th//2 + offset_y),
                            'Right':(dot_radius + offset_x, -th//2 + offset_y),
                            'Center':(-tw//2 + offset_x, -th//2 + offset_y)}[label_pos]
                    draw.text((x+dx, y+dy), txt, font=font, fill=text_color)

            # display and download
            st.image(result, caption=uploaded_file.name, use_column_width=True)
            buf = io.BytesIO(); result.save(buf,'PNG'); buf.seek(0)
            st.download_button(f'Download {uploaded_file.name}', data=buf,
                               file_name=f'overlay_{uploaded_file.name}.png', mime='image/png')

            # histogram
            fig, ax = plt.subplots()
            samples_arr = np.array(samples)
            ax.hist(samples_arr.flatten(), bins=30)
            ax.set_title('Color Value Distribution')
            st.pyplot(fig)
