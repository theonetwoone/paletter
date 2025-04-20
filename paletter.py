import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import colorsys
import random
import json
import matplotlib.pyplot as plt
import math # Added for hex/tri grid
from sklearn.cluster import KMeans # Added for palette extraction

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
    grid_res = st.slider('Cells per axis', 5, 100, 20, key='grid_res')
    adaptive = st.checkbox('Adaptive Resolution', value=False, key='adaptive')

    st.subheader('Marker')
    shape = st.selectbox('Shape', ['Circle', 'Square', 'Diamond', 'Triangle', 'Cross'], key='shape')
    dot_radius = st.slider('Radius (px)', 1, 30, 5, key='dot_radius')

    st.subheader('Labels')
    font_size = st.slider('Font size (px)', 8, 72, 12, key='font_size')
    color_format = st.selectbox('Color format', ['RGB','HEX','HSL'], key='color_format')
    label_pos = st.selectbox('Position', ['Above','Below','Left','Right','Center'], key='label_pos')
    offset_x = st.slider('Offset X', -50, 50, 0, key='offset_x')
    offset_y = st.slider('Offset Y', -50, 50, 0, key='offset_y')
    text_rotation = st.slider('Text rotation (°)', -180, 180, 0, key='text_rotation')
    dynamic_contrast = st.checkbox('Dynamic contrast', value=True, key='dynamic_contrast')
    avoid_overlap = st.checkbox('Avoid overlapping labels', value=False, key='avoid_overlap')

    # Add image resize option
    st.subheader('Image Processing')
    resize_image = st.checkbox('Resize large images', key='resize_image', value=False)
    max_dimension = st.number_input('Max dimension (px)', min_value=100, max_value=4000, value=1000, key='max_dimension', disabled=not st.session_state.get('resize_image', False))

    st.subheader('Presets')
    preset_name = st.text_input('Save as preset')
    if st.button('Save preset') and preset_name:
        st.session_state.presets[preset_name] = {
            'grid_type': grid_type, 'grid_res': st.session_state.grid_res, 'adaptive': st.session_state.adaptive,
            'shape': st.session_state.shape, 'dot_radius': st.session_state.dot_radius,
            'font_size': st.session_state.font_size, 'color_format': st.session_state.color_format,
            'label_pos': st.session_state.label_pos, 'offset_x': st.session_state.offset_x, 'offset_y': st.session_state.offset_y,
            'text_rotation': st.session_state.text_rotation, 'dynamic_contrast': st.session_state.dynamic_contrast,
            'avoid_overlap': st.session_state.avoid_overlap,
            'resize_image': st.session_state.resize_image, 'max_dimension': st.session_state.max_dimension
        }
        st.success(f"Preset '{preset_name}' saved.")
    choice = st.selectbox('Load preset', ['']+list(st.session_state.presets.keys()))
    if choice:
        p = st.session_state.presets[choice]
        # Update session state directly to trigger widgets
        st.session_state.grid_res = p.get('grid_res', 20)
        st.session_state.adaptive = p.get('adaptive', False)
        st.session_state.shape = p.get('shape', 'Circle')
        st.session_state.dot_radius = p.get('dot_radius', 5)
        st.session_state.font_size = p.get('font_size', 12)
        st.session_state.color_format = p.get('color_format', 'RGB')
        st.session_state.label_pos = p.get('label_pos', 'Above')
        st.session_state.offset_x = p.get('offset_x', 0)
        st.session_state.offset_y = p.get('offset_y', 0)
        st.session_state.text_rotation = p.get('text_rotation', 0)
        st.session_state.dynamic_contrast = p.get('dynamic_contrast', True)
        st.session_state.avoid_overlap = p.get('avoid_overlap', False)
        st.session_state.resize_image = p.get('resize_image', False)
        st.session_state.max_dimension = p.get('max_dimension', 1000)
        # Note: grid_type needs special handling if we use its key, for now keep as is
        grid_type = p.get('grid_type', 'Rectangular') # Reload grid_type selection
        st.info(f"Loaded preset '{choice}'. Please re-select '{grid_type}' from the Grid Type dropdown if it didn't update automatically.") # Add note for grid_type
    if st.button('Randomize'):
        st.session_state['rand'] = True
        st.experimental_rerun()
    # Export presets
    if st.session_state.presets:
        presets_json = json.dumps(st.session_state.presets)
        st.download_button('Export presets', data=presets_json, file_name='presets.json', mime='application/json')
    # Import presets
    preset_file = st.file_uploader('Import presets (.json)', type=['json'])
    if preset_file:
        try:
            imported_presets = json.load(preset_file)
            # Basic validation could be added here
            st.session_state.presets.update(imported_presets)
            st.success(f"Imported {len(imported_presets)} presets.")
            preset_file = None # Clear the uploader
            st.experimental_rerun() # Rerun to update load list
        except json.JSONDecodeError:
            st.error("Invalid JSON file.")
        except Exception as e:
            st.error(f"Error importing presets: {e}")

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

@st.cache_data
def process_image(image_bytes, filename, grid_type, grid_res, adaptive, shape, dot_radius,
                    font_bytes, font_size, color_format, label_pos, offset_x, offset_y,
                    text_rotation, dynamic_contrast, avoid_overlap, resize_image, max_dimension):
    """Processes a single image based on the provided settings."""
    global label_bboxes # Use global list for overlap check within this image
    label_bboxes = []
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
        w, h = img.size

        # --- 1. Image Resizing ---
        if resize_image and (w > max_dimension or h > max_dimension):
            if w > h:
                new_w = max_dimension
                new_h = int(h * (max_dimension / w))
            else:
                new_h = max_dimension
                new_w = int(w * (max_dimension / h))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = img.size

        arr = np.array(img)

        # --- 2. Effective Grid Resolution ---
        grid_res_eff = grid_res
        if adaptive:
            variance = float(arr[:, :, :3].var())
            grid_res_eff = min(max(int(grid_res * (1 + variance / 10000)), 5), 200)

        # --- 3. Canvas & Font ---
        result = Image.new('RGBA', (w, h), (255, 255, 255, 0))
        draw = ImageDraw.Draw(result)
        font = None
        try:
            if font_bytes:
                font = ImageFont.truetype(io.BytesIO(font_bytes), font_size)
            else:
                # Try common system fonts as fallbacks
                try: font = ImageFont.truetype('arial.ttf', font_size)
                except IOError: 
                    try: font = ImageFont.truetype('LiberationSans-Regular.ttf', font_size)
                    except IOError: font = ImageFont.load_default()
        except Exception as e:
            st.error(f"Error loading font: {e}. Using default font.")
            font = ImageFont.load_default()

        samples = []
        coordinates = []

        # --- 4. Grid Coordinate Calculation ---
        if grid_type == 'Rectangular':
            step_x, step_y = max(1, w // grid_res_eff), max(1, h // grid_res_eff)
            for i in range(0, w, step_x):
                for j in range(0, h, step_y):
                    coordinates.append((i, j, step_x, step_y))
        elif grid_type == 'Hexagonal':
            hex_size = max(1, int(min(w, h) / (grid_res_eff * 1.5))) # Approximate size
            hex_width = hex_size * 2
            hex_height = math.sqrt(3) * hex_size
            step_x_hex = hex_width * 3/4
            step_y_hex = hex_height
            row_idx = 0
            for y in np.arange(hex_height/2, h, step_y_hex):
                start_x = (hex_width / 4) if (row_idx % 2 == 0) else (hex_width * 3 / 4)
                for x in np.arange(start_x, w, step_x_hex):
                     # Use hex_size for sampling area approximation
                    coordinates.append((int(x), int(y), int(hex_size*0.8), int(hex_size*0.8)))
                row_idx += 1
        elif grid_type == 'Triangular':
            tri_height = max(1, h / grid_res_eff)
            tri_base = tri_height * 2 / math.sqrt(3)
            step_y_tri = tri_height
            row_idx = 0
            for y in np.arange(tri_height / 2, h, step_y_tri):
                start_x = (tri_base / 4) if (row_idx % 2 == 0) else (tri_base * 3 / 4)
                for x in np.arange(start_x, w, tri_base / 2): # Check this step
                    # Use triangle height/base for sampling approx
                    coordinates.append((int(x), int(y), int(tri_base * 0.4), int(tri_height * 0.4)))
                row_idx += 1

        # --- 5. Iterate Grid Cells, Sample, Draw Marker & Label ---
        for x, y, sx, sy in coordinates:
            # Define bounds for sampling more carefully
            ix, iy = int(x), int(y)
            half_sx, half_sy = int(sx / 2), int(sy / 2)
            # Clamp bounds to image dimensions
            min_j, max_j = max(0, iy - half_sy), min(h, iy + half_sy)
            min_i, max_i = max(0, ix - half_sx), min(w, ix + half_sx)
            
            cell = arr[min_j:max_j, min_i:max_i]
            if cell.size == 0 or cell[:,:,3].mean() < 10: # Skip empty or highly transparent areas
                continue

            avg_color = tuple(int(c) for c in cell[:, :, :3].mean(axis=(0, 1)))
            samples.append(avg_color)

            draw_marker(draw, ix, iy, dot_radius, avg_color + (255,), shape)

            # --- Label Formatting ---
            if color_format == 'RGB':
                txt = str(avg_color)
            elif color_format == 'HEX':
                txt = '#{:02X}{:02X}{:02X}'.format(*avg_color)
            else: # HSL
                hls = colorsys.rgb_to_hls(avg_color[0] / 255, avg_color[1] / 255, avg_color[2] / 255)
                txt = f"HSL({int(hls[0] * 360)},{int(hls[2] * 100)}%,{int(hls[1] * 100)}%)"

            text_color = get_contrast_text_color(avg_color) if dynamic_contrast else (0,0,0) # Default to black if no dynamic

            # --- Text Rendering with getbbox ---
            try:
                # Use getbbox for layout, getmask for precise size
                bbox = font.getbbox(txt) # (left, top, right, bottom)
                text_left, text_top, text_right, text_bottom = bbox
                tw = text_right - text_left
                th = text_bottom - text_top
                # Offset based on the actual drawn text bounding box, adjusted for baseline
                offset_origin_x = -text_left
                offset_origin_y = -text_top
            except AttributeError: # Fallback for default font if getbbox fails
                tw, th = draw.textlength(txt, font=font), font.size # Approximate
                offset_origin_x, offset_origin_y = 0, 0
                bbox = (0, 0, tw, th)


            pos_offsets = {
                'Above': (-tw / 2 + offset_x, -dot_radius - th + offset_y),
                'Below': (-tw / 2 + offset_x, dot_radius + offset_y),
                'Left': (-dot_radius - tw + offset_x, -th / 2 + offset_y),
                'Right': (dot_radius + offset_x, -th / 2 + offset_y),
                'Center': (-tw / 2 + offset_x, -th / 2 + offset_y)
            }
            dx, dy = pos_offsets[label_pos]

            # Render rotated text onto a temporary transparent image
            txt_img = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
            txt_draw = ImageDraw.Draw(txt_img)
            # Draw text at adjusted origin (0,0 within its own bbox)
            txt_draw.text((offset_origin_x, offset_origin_y), txt, font=font, fill=text_color + (255,))
            
            rotated_txt_img = txt_img.rotate(text_rotation, expand=True, resample=Image.BILINEAR)
            rtw, rth = rotated_txt_img.size

            # Calculate final position, considering rotation center
            # Center of original text box relative to marker center (ix, iy)
            center_x = ix + dx + tw / 2
            center_y = iy + dy + th / 2
            # Top-left corner of rotated text box needed to center it
            final_x = int(center_x - rtw / 2)
            final_y = int(center_y - rth / 2)

            # --- Overlap Avoidance ---
            bbox_final = (final_x, final_y, final_x + rtw, final_y + rth)
            if avoid_overlap and intersects_any(bbox_final):
                continue
            label_bboxes.append(bbox_final)

            # Composite the rotated text onto the main result image
            result.alpha_composite(rotated_txt_img, (final_x, final_y))

        # --- 6. Palette Extraction ---
        palette = []
        if samples:
            try:
                # Use k-means to find 5 dominant colors
                kmeans = KMeans(n_clusters=min(5, len(set(samples))), random_state=0, n_init=10).fit(samples)
                palette = [tuple(map(int, center)) for center in kmeans.cluster_centers_]
            except Exception as e:
                st.warning(f"Could not extract palette: {e}")

        # --- 7. Final Output ---
        buf = io.BytesIO()
        result.save(buf, 'PNG')
        buf.seek(0)
        result_bytes = buf.getvalue()

        # Also generate histogram data
        hist_data = None
        if samples:
            hist_data = np.array(samples).flatten()

        return result_bytes, palette, hist_data, w, h # Return necessary data

    except Exception as e:
        st.error(f"Error processing {filename}: {e}")
        return None, [], None, None, None

# --- Main Application Logic ---

# Randomization Logic
random_settings = {}
if 'rand' in st.session_state and st.session_state['rand']:
    random_settings = {
        'grid_type': random.choice(['Rectangular', 'Hexagonal', 'Triangular']),
        'grid_res': random.randint(5, 50),
        'adaptive': random.choice([True, False]),
        'shape': random.choice(['Circle', 'Square', 'Diamond', 'Triangle', 'Cross']),
        'dot_radius': random.randint(1, 15),
        'font_size': random.randint(8, 24),
        'color_format': random.choice(['RGB', 'HEX', 'HSL']),
        'label_pos': random.choice(['Above', 'Below', 'Left', 'Right', 'Center']),
        'offset_x': random.randint(-20, 20),
        'offset_y': random.randint(-20, 20),
        'text_rotation': random.randint(-45, 45),
        'dynamic_contrast': random.choice([True, False]),
        'avoid_overlap': random.choice([True, False]),
        'resize_image': st.session_state.resize_image, # Keep resize settings
        'max_dimension': st.session_state.max_dimension
    }
    # We don't directly set widget state, just use these values for processing this run
    st.info("Using randomized settings!")
    # Ensure rand flag is removed for next run
    del st.session_state['rand']

# Retrieve settings from session state or use randomized ones
current_grid_type = random_settings.get('grid_type', grid_type) # Still need the widget value for display
current_settings = {
    'grid_res': random_settings.get('grid_res', st.session_state.grid_res),
    'adaptive': random_settings.get('adaptive', st.session_state.adaptive),
    'shape': random_settings.get('shape', st.session_state.shape),
    'dot_radius': random_settings.get('dot_radius', st.session_state.dot_radius),
    'font_size': random_settings.get('font_size', st.session_state.font_size),
    'color_format': random_settings.get('color_format', st.session_state.color_format),
    'label_pos': random_settings.get('label_pos', st.session_state.label_pos),
    'offset_x': random_settings.get('offset_x', st.session_state.offset_x),
    'offset_y': random_settings.get('offset_y', st.session_state.offset_y),
    'text_rotation': random_settings.get('text_rotation', st.session_state.text_rotation),
    'dynamic_contrast': random_settings.get('dynamic_contrast', st.session_state.dynamic_contrast),
    'avoid_overlap': random_settings.get('avoid_overlap', st.session_state.avoid_overlap),
    'resize_image': random_settings.get('resize_image', st.session_state.resize_image),
    'max_dimension': random_settings.get('max_dimension', st.session_state.max_dimension)
}

# Read font file bytes once
font_bytes_content = None
if font_file:
    font_bytes_content = font_file.getvalue()

# Processing Logic (Iterate through uploaded files)
if uploaded_files:
    if len(uploaded_files) > 4:
        st.warning("Processing more than 4 images might be slow.")
    
    cols = st.columns(min(len(uploaded_files), 2)) # Max 2 columns
    
    for idx, uploaded_file in enumerate(uploaded_files):
        with cols[idx % 2]:
            st.subheader(f"Processing: {uploaded_file.name}")
            image_bytes_content = uploaded_file.getvalue()

            # Call the cached processing function
            result_bytes, palette, hist_data, width, height = process_image(
                image_bytes=image_bytes_content,
                filename=uploaded_file.name,
                grid_type=current_grid_type,
                font_bytes=font_bytes_content,
                **current_settings # Pass all other settings
            )

            if result_bytes:
                st.image(result_bytes, caption=f'{uploaded_file.name} ({width}x{height})' if width else uploaded_file.name, use_container_width=True)
                st.download_button(
                    label=f'Download {uploaded_file.name}',
                    data=result_bytes,
                    file_name=f'overlay_{uploaded_file.name}.png',
                    mime='image/png'
                )

                # Display Palette
                if palette:
                    st.markdown("**Dominant Colors:**")
                    palette_cols = st.columns(len(palette))
                    for p_idx, color in enumerate(palette):
                        with palette_cols[p_idx]:
                            rgb_str = f"rgb({color[0]}, {color[1]}, {color[2]})"
                            st.markdown(f"<div style='background-color:{rgb_str}; width:50px; height:50px; border: 1px solid grey; margin: auto;'></div>", unsafe_allow_html=True)
                            st.caption(f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}")

                # Display Histogram
                if hist_data is not None:
                    fig, ax = plt.subplots()
                    ax.hist(hist_data, bins=30, color='gray')
                    ax.set_title('Color Channel Distribution')
                    ax.set_xlabel('Pixel Intensity (0-255)')
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
            else:
                st.error(f"Failed to process {uploaded_file.name}")

# Clear randomization flag if it wasn't used (e.g., no files uploaded)
if 'rand' in st.session_state and not random_settings:
   del st.session_state['rand']
